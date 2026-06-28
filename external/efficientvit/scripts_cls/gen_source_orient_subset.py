#!/usr/bin/env python3
import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity

# ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.classification.resnet50_cls import ResNet50Classifier
from models.classification.dataset_office import OfficeDataset
from submodlib.functions.facilityLocationMutualInformation import (
    FacilityLocationMutualInformationFunction
)


# ============================================================
# IO helpers
# ============================================================

def save_txt(path, items):
    with open(path, "w") as f:
        f.write("\n".join(items))


def load_txt(path):
    with open(path, "r") as f:
        return [x.strip() for x in f.readlines() if x.strip()]


def maybe_load_source_cache(src_cache_root):
    """
    Load cached source gradients / names / vecs / K if all required files exist.
    """
    src_grad_path = os.path.join(src_cache_root, "src_grads.npy")
    src_name_path = os.path.join(src_cache_root, "src_names.txt")
    src_vec_path = os.path.join(src_cache_root, "src_vecs.npy")
    src_id_path = os.path.join(src_cache_root, "src_ids.txt")
    src_K_path = os.path.join(src_cache_root, "src_K.npy")

    required = [
        src_grad_path,
        src_name_path,
        src_vec_path,
        src_id_path,
    ]

    if not all(os.path.exists(p) for p in required):
        return None

    out = {
        "src_grads": np.load(src_grad_path),
        "src_names": load_txt(src_name_path),
        "src_vecs": np.load(src_vec_path),
        "src_ids": load_txt(src_id_path),
        "src_K": np.load(src_K_path) if os.path.exists(src_K_path) else None,
    }
    return out


def maybe_load_target_cache(tgt_cache_root):
    """
    Load cached target gradients / names / vecs / Q if all required files exist.
    """
    tgt_grad_path = os.path.join(tgt_cache_root, "tgt_grads.npy")
    tgt_name_path = os.path.join(tgt_cache_root, "tgt_names.txt")
    tgt_vec_path = os.path.join(tgt_cache_root, "tgt_vecs.npy")
    tgt_id_path = os.path.join(tgt_cache_root, "tgt_ids.txt")
    tgt_Q_path = os.path.join(tgt_cache_root, "src_tgt_Q.npy")

    required = [
        tgt_grad_path,
        tgt_name_path,
        tgt_vec_path,
        tgt_id_path,
    ]

    if not all(os.path.exists(p) for p in required):
        return None

    out = {
        "tgt_grads": np.load(tgt_grad_path),
        "tgt_names": load_txt(tgt_name_path),
        "tgt_vecs": np.load(tgt_vec_path),
        "tgt_ids": load_txt(tgt_id_path),
        "src_tgt_Q": np.load(tgt_Q_path) if os.path.exists(tgt_Q_path) else None,
    }
    return out


# ============================================================
# Utils
# ============================================================

def detect_domain(path: str) -> str:
    p = path.lower()
    if "amazon" in p:
        return "amazon"
    if "dslr" in p:
        return "dslr"
    if "webcam" in p:
        return "webcam"
    return "unknown"


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])


# ============================================================
# Select θ′ parameters → last classifier layer
# match seg logic: final_head.parameters()
# ============================================================

def get_theta_params(model):
    theta = []
    for name, p in model.named_parameters():
        if "fc" in name:
            theta.append(p)
    return theta


# ============================================================
# ORIENT last-layer gradient approximation (classification)
# g = (p - y) * h
# dim = feature_dim (2048 for ResNet50)
# ============================================================

def compute_gradient_features(model, loader, device, theta_params, save_prefix=None):

    model.eval()

    grads = []
    names = []

    backbone = model.model

    for img, lbl, path in loader:

        # ORIENT requires batch size 1
        assert img.shape[0] == 1, "ORIENT extraction expects batch_size=1"

        img = img.to(device)
        lbl = lbl.to(device)

        # --------------------------------------------------------
        # forward backbone feature
        # --------------------------------------------------------
        x = backbone.conv1(img)
        x = backbone.bn1(x)
        x = backbone.relu(x)
        x = backbone.maxpool(x)

        x = backbone.layer1(x)
        x = backbone.layer2(x)
        x = backbone.layer3(x)
        x = backbone.layer4(x)

        x = backbone.avgpool(x)

        feat = torch.flatten(x, 1)   # (1, 2048)

        # --------------------------------------------------------
        # classifier forward
        # --------------------------------------------------------
        logits = backbone.fc(feat)

        prob = torch.softmax(logits, dim=1)

        # --------------------------------------------------------
        # build one-hot label
        # --------------------------------------------------------
        onehot = torch.zeros_like(prob)
        onehot.scatter_(1, lbl.view(-1,1), 1)

        # --------------------------------------------------------
        # ORIENT gradient embedding
        # g = (p - y) ⊗ h
        # --------------------------------------------------------
        diff = prob - onehot                # (1, C)

        g = torch.einsum("bc,bd->bcd", diff, feat)   # (1, C, 2048)

        g = g.reshape(g.size(0), -1)        # (1, C*2048)

        # --------------------------------------------------------
        # normalize for cosine similarity
        # --------------------------------------------------------
        g = g / (g.norm(dim=1, keepdim=True) + 1e-12)

        g = g.detach().cpu().numpy().astype(np.float32)

        grads.append(g.squeeze(0))
        names.append(path[0])

    grads = np.stack(grads, axis=0)

    # ------------------------------------------------------------
    # optional save
    # ------------------------------------------------------------
    if save_prefix is not None:

        np.save(save_prefix + "_grads.npy", grads)

        with open(save_prefix + "_names.txt", "w") as f:
            f.write("\n".join(names))

        print(f"💾 Saved gradients → {save_prefix}_grads.npy")

    return grads, names
# ============================================================
# Instance-level aggregation
# classification has no subject aggregation, but keep same structure
# ============================================================

def aggregate_to_instance_level(grads, names, save_prefix=None):
    ids = [os.path.basename(x) for x in names]
    vecs = grads.astype(np.float32)

    if save_prefix is not None:
        np.save(save_prefix + "_vecs.npy", vecs)
        with open(save_prefix + "_ids.txt", "w") as f:
            f.write("\n".join(ids))
        print(f"💾 Saved embeddings → {save_prefix}_vecs.npy")

    return ids, vecs


# ============================================================
# Similarity helpers
# ============================================================

def build_or_load_K(src_vecs, src_cache_root):
    K_path = os.path.join(src_cache_root, "src_K.npy")
    if os.path.exists(K_path):
        print("⚡ Loading cached SOURCE-SOURCE K matrix...")
        return np.load(K_path)

    print("🧮 Computing SOURCE-SOURCE K matrix...")
    K = np.maximum(cosine_similarity(src_vecs, src_vecs), 0).astype(np.float32)
    np.save(K_path, K)
    print(f"💾 Saved K → {K_path}")
    return K


def build_or_load_Q(src_vecs, tgt_vecs, tgt_cache_root):
    Q_path = os.path.join(tgt_cache_root, "src_tgt_Q.npy")
    if os.path.exists(Q_path):
        print("⚡ Loading cached SOURCE-TARGET Q matrix...")
        return np.load(Q_path)

    print("🧮 Computing SOURCE-TARGET Q matrix...")
    Q = np.maximum(cosine_similarity(src_vecs, tgt_vecs), 0).astype(np.float32)
    np.save(Q_path, Q)
    print(f"💾 Saved Q → {Q_path}")
    return Q


# ============================================================
# ORIENT greedy selection
# ============================================================

def orient_full_greedy_with_gain_from_KQ(K, Q, budget, eta=1.0):
    Ns = K.shape[0]
    Nt = Q.shape[1]

    print(f"Running FLMI with cached matrices (Ns={Ns}, Nt={Nt})")

    obj = FacilityLocationMutualInformationFunction(
        n=Ns,
        num_queries=Nt,
        data_sijs=K,
        query_sijs=Q,
        magnificationEta=eta,
    )

    budget = int(min(budget, Ns))

    result = obj.maximize(
        budget=budget,
        optimizer="LazyGreedy",
        stopIfNegativeGain=False,
        show_progress=True,
    )

    ordered_idx = []
    gains = []

    for elem in result:
        if isinstance(elem, tuple):
            idx, gain = elem
            ordered_idx.append(int(idx))
            gains.append(float(gain))
        else:
            ordered_idx.append(int(elem))
            gains.append(0.0)

    return ordered_idx, gains


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser("Run ORIENT source selection for Office31 target subset")
    ap.add_argument("--source_list", required=True,
                    help="e.g. data_cls/splits/office31/amazon/source_train.txt")
    ap.add_argument("--target_list", required=True,
                    help="e.g. data_cls/selections/office31/amazon/random1_2shot.txt")
    ap.add_argument("--warmup_ckpt", required=True,
                    help="source-only warmup checkpoint, e.g. experiments_cls/office31/amazon/source_only_full/best.pt")
    ap.add_argument("--budget_ratio", type=float, default=0.3)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_domain = detect_domain(args.source_list)
    target_domain = detect_domain(args.target_list)
    target_name = os.path.basename(args.target_list).replace(".txt", "")

    print("\n==============================")
    print("🚀 ORIENT SOURCE SELECTION")
    print("==============================")
    print("Source list:", args.source_list)
    print("Target list:", args.target_list)
    print("Warmup ckpt:", args.warmup_ckpt)
    print("Detected source domain:", source_domain)
    print("Detected target domain:", target_domain)

    results_root = "results_cls/orient_cache"
    src_cache_root = os.path.join(results_root, f"orient_source_cache_{source_domain}")
    tgt_cache_root = os.path.join(results_root, f"orient_target_cache_{source_domain}_{target_name}")
    os.makedirs(src_cache_root, exist_ok=True)
    os.makedirs(tgt_cache_root, exist_ok=True)

    output_root = os.path.join("data_cls", "splits_subset", "office31", source_domain)
    os.makedirs(output_root, exist_ok=True)

    # ============================================================
    # Load warmup model (source-only best.pt)
    # ============================================================
    print("\n🚀 Loading source-only warmup checkpoint...")

    model = ResNet50Classifier(
        num_classes=31,
        pretrained=False,
    ).to(device)

    state = torch.load(args.warmup_ckpt, map_location="cpu")
    sd = state["model_state"]

    # fix possible "model." prefix mismatch
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[6:]
        new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    model = model.to(device)
    model.eval()

    theta_params = get_theta_params(model)

    # ============================================================
    # Build datasets
    # ============================================================
    transform = build_transform()

    src_dataset = OfficeDataset(
        args.source_list,
        transform=transform,
        return_path=True
    )

    tgt_dataset = OfficeDataset(
        args.target_list,
        transform=transform,
        class_to_idx=src_dataset.class_to_idx,
        return_path=True
    )

    print("\n📁 Loading datasets...")
    print(f"✅ Loaded source: {len(src_dataset)} samples")
    print("Example source samples:")
    for i in range(min(5, len(src_dataset.samples))):
        print("   ", src_dataset.samples[i])

    print(f"✅ Loaded target: {len(tgt_dataset)} samples")
    print("Example target samples:")
    for i in range(min(5, len(tgt_dataset.samples))):
        print("   ", tgt_dataset.samples[i])

    src_loader = DataLoader(
        src_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    tgt_loader = DataLoader(
        tgt_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # ============================================================
    # SOURCE cache: gradients / vecs
    # ============================================================
    src_cache = maybe_load_source_cache(src_cache_root)

    if src_cache is None:
        print("\n🧠 Computing SOURCE gradients (first time only)...")
        src_grads, src_names = compute_gradient_features(
            model, src_loader, device, theta_params,
            save_prefix=os.path.join(src_cache_root, "src")
        )

        print("\n📦 Aggregating SOURCE to instance-level...")
        src_ids, src_vecs = aggregate_to_instance_level(
            src_grads, src_names,
            save_prefix=os.path.join(src_cache_root, "src")
        )

        # unified alias files
        np.save(os.path.join(src_cache_root, "src_grads.npy"), src_grads)
        save_txt(os.path.join(src_cache_root, "src_names.txt"), src_names)
        np.save(os.path.join(src_cache_root, "src_vecs.npy"), src_vecs)
        save_txt(os.path.join(src_cache_root, "src_ids.txt"), src_ids)

        src_meta = {
            "source_list": args.source_list,
            "warmup_ckpt": args.warmup_ckpt,
            "num_source_instances": len(src_ids),
        }
        with open(os.path.join(src_cache_root, "meta.json"), "w") as f:
            json.dump(src_meta, f, indent=2)

        print(f"✅ Saved SOURCE cache → {src_cache_root}")
    else:
        print("⚡ Loading cached SOURCE gradients / vecs...")
        src_grads = src_cache["src_grads"]
        src_names = src_cache["src_names"]
        src_vecs = src_cache["src_vecs"]
        src_ids = src_cache["src_ids"]

    # ============================================================
    # TARGET cache: gradients / vecs
    # ============================================================
    tgt_cache = maybe_load_target_cache(tgt_cache_root)

    if tgt_cache is None:
        print("\n🎯 Computing TARGET gradients...")
        tgt_grads, tgt_names = compute_gradient_features(
            model, tgt_loader, device, theta_params,
            save_prefix=os.path.join(tgt_cache_root, "tgt")
        )

        print("\n📦 Aggregating TARGET to instance-level...")
        tgt_ids, tgt_vecs = aggregate_to_instance_level(
            tgt_grads, tgt_names,
            save_prefix=os.path.join(tgt_cache_root, "tgt")
        )

        # unified alias files
        np.save(os.path.join(tgt_cache_root, "tgt_grads.npy"), tgt_grads)
        save_txt(os.path.join(tgt_cache_root, "tgt_names.txt"), tgt_names)
        np.save(os.path.join(tgt_cache_root, "tgt_vecs.npy"), tgt_vecs)
        save_txt(os.path.join(tgt_cache_root, "tgt_ids.txt"), tgt_ids)

        tgt_meta = {
            "target_list": args.target_list,
            "warmup_ckpt": args.warmup_ckpt,
            "num_target_instances": len(tgt_ids),
        }
        with open(os.path.join(tgt_cache_root, "meta.json"), "w") as f:
            json.dump(tgt_meta, f, indent=2)

        print(f"✅ Saved TARGET cache → {tgt_cache_root}")
    else:
        print("⚡ Loading cached TARGET gradients / vecs...")
        tgt_grads = tgt_cache["tgt_grads"]
        tgt_names = tgt_cache["tgt_names"]
        tgt_vecs = tgt_cache["tgt_vecs"]
        tgt_ids = tgt_cache["tgt_ids"]

    print(f"   SOURCE vecs: {src_vecs.shape}")
    print(f"   TARGET vecs: {tgt_vecs.shape}")

    # ============================================================
    # Build or load K and Q
    # ============================================================
    K = build_or_load_K(src_vecs, src_cache_root)
    Q = build_or_load_Q(src_vecs, tgt_vecs, tgt_cache_root)
    # ============================================================
    # DEBUG: check Q collapse
    # ============================================================
    from collections import defaultdict
    print("\n🔍 Checking SOURCE→TARGET similarity (Q) collapse...")

    cls_scores = defaultdict(list)

    for i, name in enumerate(src_names):
        cls = name.split("/")[-2]
        cls_scores[cls].append(Q[i].mean())

    cls_mean = {k: float(np.mean(v)) for k, v in cls_scores.items()}
    cls_mean = dict(sorted(cls_mean.items(), key=lambda x: x[1], reverse=True))

    print("\nTop classes by mean Q similarity:")
    for k, v in list(cls_mean.items())[:10]:
        print(f"{k:20s} {v:.4f}")

    # ============================================================
    # DEBUG: check K structure
    # ============================================================
    print("\n🔍 Checking SOURCE-SOURCE similarity (K)...")

    from collections import defaultdict

    src_cls = [n.split("/")[-2] for n in src_names]

    intra = defaultdict(list)
    inter = []

    N = len(src_names)

    for i in range(N):
        for j in range(i + 1, N):

            if src_cls[i] == src_cls[j]:
                intra[src_cls[i]].append(K[i, j])
            else:
                inter.append(K[i, j])

    print("\nMean inter-class K:", float(np.mean(inter)))

    for c in ["trash_can", "tape_dispenser", "stapler", "speaker"]:
        if c in intra:
            print(f"{c:20s} intra-class K:", float(np.mean(intra[c])))
    # ============================================================
    # ORIENT greedy ordering
    # ============================================================
    budget = int(len(src_ids) * args.budget_ratio)

    print("\n🧭 Running ORIENT greedy selection...")
    ordered_idx, gains = orient_full_greedy_with_gain_from_KQ(
        K, Q, budget=budget, eta=args.eta
    )

    ordered_names = [src_names[i] for i in ordered_idx]
    # ============================================================
    # DEBUG: class distribution of selected subset
    # ============================================================
    from collections import Counter

    cls = [p.split("/")[-2] for p in ordered_names[:budget]]

    print("\n📊 Selected class distribution:")
    print(Counter(cls))
    
    selected_names = []
    for p in ordered_names[:budget]:
        label = p.split("/")[-2]   # class name from path
        selected_names.append(f"{p} {label}")
    
    # ============================================================
    # Save selected subset
    # ============================================================
    output_path = os.path.join(
        output_root,
        f"source_orient_given_{target_name}.txt"
    )
    save_txt(output_path, selected_names)

    print(f"✅ Saved ORIENT subset: {output_path} ({len(selected_names)} instances)")

    # ============================================================
    # Save ranking / gains / score dict
    # ============================================================
    with open(os.path.join(tgt_cache_root, "orient_sorted_names.txt"), "w") as f:
        f.write("\n".join(ordered_names))

    np.save(os.path.join(tgt_cache_root, "orient_gains.npy"), np.array(gains, dtype=np.float32))

    orient_score_dict = {name: 0.0 for name in src_names}
    L = len(ordered_idx)
    for r, idx in enumerate(ordered_idx):
        name = src_names[idx]
        orient_score_dict[name] = 1.0 - r / max(L - 1, 1)

    np.save(
        os.path.join(tgt_cache_root, "orient_score_dict.npy"),
        orient_score_dict,
        allow_pickle=True
    )

    meta = {
        "source_list": args.source_list,
        "target_list": args.target_list,
        "warmup_ckpt": args.warmup_ckpt,
        "budget_ratio": args.budget_ratio,
        "budget": budget,
        "eta": args.eta,
        "src_cache_root": src_cache_root,
        "tgt_cache_root": tgt_cache_root,
        "num_source_instances": len(src_ids),
        "num_target_instances": len(tgt_ids),
    }
    with open(os.path.join(tgt_cache_root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\n🎉 ORIENT active-source selection completed.")


if __name__ == "__main__":
    main()