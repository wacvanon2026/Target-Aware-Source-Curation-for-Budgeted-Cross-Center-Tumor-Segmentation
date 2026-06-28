#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import random

# -------------------------------------------------
# Local imports
# -------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from models.efficientvit_seg.losses import DiceCELoss


# -------------------------------------------------
# Dataset Loader (UNCHANGED)
# -------------------------------------------------
def load_datasets(data_cfg, prebuilt=None):
    
    if prebuilt is not None:
        print("⚡ Using prebuilt datasets")
        return prebuilt["train"], prebuilt["val"]
    
    train_datasets = []

    for dom in data_cfg["domains"]:
        ds = BraTSSliceDataset(
            root_dir=dom["path"],
            split=dom["split"],
            img_size=data_cfg["img_size"],
            split_txt_dir=dom.get("split_txt"),
            skip_empty=data_cfg.get(
                "skip_empty_train" if dom["split"] == "train" else "skip_empty_val",
                False
            ),
        )
        train_datasets.append(ds)
        print(f"✅ Loaded train domain: {dom['name']} ({len(ds)} slices)")

    train_dataset = ConcatDataset(train_datasets)
    print(f"📊 Total train dataset: {len(train_dataset)} slices")

    val_cfg = data_cfg["val"]
    val_dataset = BraTSSliceDataset(
        root_dir=val_cfg["path"],
        split=val_cfg["split"],
        img_size=data_cfg["img_size"],
        split_txt_dir=val_cfg.get("split_txt"),
        skip_empty=data_cfg.get("skip_empty_val", False),
    )

    return train_dataset, val_dataset


# -------------------------------------------------
# DataLoader helper
# -------------------------------------------------
def make_loader(dataset, cfg, seed, shuffle):
    g = torch.Generator()
    g.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )


# -------------------------------------------------
# One SHORT training loop (CONTINUATION AWARE)
# -------------------------------------------------
def train_short(model, loader, optimizer, criterion,
                device, scaler, start_iter, max_iters):

    model.train()
    losses = []
    global_it = start_iter

    data_iter = iter(loader)

    while global_it < max_iters:
        try:
            batch = next(data_iter)
        except StopIteration:
            # 1 epoch ends → regenerate iterator
            data_iter = iter(loader)
            batch = next(data_iter)

        imgs = batch[0].to(device)
        masks = batch[1].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
        global_it += 1

    return np.array(losses), global_it


@torch.no_grad()
def eval_short(model, loader, criterion, device):
    model.eval()
    losses = []

    for batch in loader:
        imgs = batch[0].to(device)
        masks = batch[1].to(device)
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        losses.append(loss.item())

    return np.array(losses)

def run_short_training_with_dataset(
    cfg: dict,
    seeds: List[int],
    train_dataset,
    val_dataset,
) -> Dict[str, Any]:
    """
    供 StageA2 直接 import 调用的入口：
    - 不会构建/扫描数据集（train_dataset, val_dataset 由外部传入）
    - 仍然按你现有的逻辑：warmup ckpt -> train_short -> eval_short -> 保存 latest.pt
    - 返回 loss 数组，方便你之后做 fitness proxy（如果需要）
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")
    print(f"🎲 Running short-eval with seeds = {seeds}")

    max_iters = cfg["trainer"]["max_iters"]
    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    print(f"⚡ Short evaluation target iters = {max_iters}")

    all_train_losses = []
    all_val_losses = []

    for seed in seeds:
        print(f"\n🌱 Seed {seed} -------------------------------")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        start_iter = 0
        ckpt = None

        # ===== Resume (Warmup checkpoint) =====
        if "warmup" in cfg and "checkpoint" in cfg["warmup"]:
            ckpt_path = cfg["warmup"]["checkpoint"]
            assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
            print(f"🔥 Resuming from checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            start_iter = ckpt.get("global_iter", 0)

        # ---- model ----
        mcfg = cfg["model"]
        model = EfficientViT_Seg(
            backbone=mcfg["name"],
            in_channels=mcfg["in_channels"],
            num_classes=mcfg["num_classes"],
            pretrained=(ckpt is None),
        ).to(device)

        if ckpt is not None:
            model.load_state_dict(ckpt["model_state"])

        # ---- loaders ----
        train_loader = make_loader(train_dataset, cfg, seed=seed, shuffle=True)
        val_loader = make_loader(val_dataset, cfg, seed=0, shuffle=False)

        # ---- optim ----
        criterion = DiceCELoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )
        scaler = torch.cuda.amp.GradScaler()
        # ---- run ----
        train_losses, global_it = train_short(
            model, train_loader, optimizer,
            criterion, device, scaler,
            start_iter=start_iter,
            max_iters=max_iters,
        )
        val_losses = eval_short(model, val_loader, criterion, device)

        print(f"📉 Train loss: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
        print(f"📉 Val   loss: {val_losses[0]:.4f} → {val_losses[-1]:.4f}")

        # ---- save latest checkpoint ----
        torch.save(
            {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "global_iter": global_it,
            },
            os.path.join(save_dir, "latest.pt"),
        )

        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)

    # ---------------- Save losses ----------------
    np.save(os.path.join(save_dir, "train_losses_seeds.npy"), np.stack(all_train_losses))
    np.save(os.path.join(save_dir, "val_losses_seeds.npy"), np.stack(all_val_losses))
    np.save(os.path.join(save_dir, "train_losses_mean.npy"),
            np.mean(np.stack(all_train_losses), axis=0))
    np.save(os.path.join(save_dir, "val_losses_mean.npy"),
            np.mean(np.stack(all_val_losses), axis=0))

    print("\n✅ Multi-seed short evaluation finished.")

    return {
        "train_losses_seeds": all_train_losses,
        "val_losses_seeds": all_val_losses,
    }
# -------------------------------------------------
# Main
# -------------------------------------------------
def main(config_path, seeds):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ---------------- Data (CLI mode uses load_datasets) ----------------
    print("📁 Loading datasets...")
    train_dataset, val_dataset = load_datasets(cfg["data"])

    # delegate
    run_short_training_with_dataset(
        cfg=cfg,
        seeds=seeds,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="0")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    main(args.config, seeds)
