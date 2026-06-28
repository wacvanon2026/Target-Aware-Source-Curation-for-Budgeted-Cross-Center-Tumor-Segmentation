#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader, ConcatDataset

# ensure local imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.efficientvit_seg.losses import DiceCELoss
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset

import torch.nn.functional as F
import torch.nn as nn


# =====================================================
# Dice + CE loss (for ∇θ L)
# =====================================================
class DiceCELoss(nn.Module):
    def __init__(self, smooth=1e-5, weight_ce=0.5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)

        preds_soft = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()

        intersection = (preds_soft * targets_onehot).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (
            preds_soft.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3)) + self.smooth
        )
        dice_loss = 1 - dice.mean()

        return self.weight_ce * ce_loss + (1 - self.weight_ce) * dice_loss


# =====================================================
# Select θ′ parameters → last layer
# =====================================================
def get_theta_params(model):
    return list(model.final_head.parameters())


# =====================================================
# Per-slice ∇θ′ L
# =====================================================
def compute_gradient_features(model, loader, device, theta_params, save_prefix=None):
    loss_fn = DiceCELoss()
    model.train()

    grads = []
    names = []

    for img, lbl, subject_ids, slice_idxs in tqdm(loader, desc="ORIENT: computing gradients"):
        img = img.to(device)
        lbl = lbl.to(device)

        logits = model(img)
        loss = loss_fn(logits, lbl)

        model.zero_grad()
        loss.backward()

        g_list = [p.grad.view(-1) for p in theta_params if p.grad is not None]
        g = torch.cat(g_list).detach().cpu().numpy()

        grads.append(g)
        for sid, sidx in zip(subject_ids, slice_idxs):
            names.append(f"{sid}_slice{sidx.item()}")

    grads = np.stack(grads, axis=0)

    # ---------- OPTIONAL SAVE ----------
    if save_prefix is not None:
        np.save(save_prefix + "_slice_grads.npy", grads)
        with open(save_prefix + "_slice_names.txt", "w") as f:
            f.write("\n".join(names))
        print(f"💾 Saved slice gradients → {save_prefix}_slice_grads.npy")

    return grads, names


# =====================================================
# Slice → Subject
# =====================================================
def aggregate_to_subject_level(slice_grads, slice_names, save_prefix=None):
    subj_to_vecs = defaultdict(list)

    for g, name in zip(slice_grads, slice_names):
        subj = name.split("_slice")[0]
        subj_to_vecs[subj].append(g)

    subj_ids = sorted(subj_to_vecs.keys())
    subj_vecs = []

    for sid in subj_ids:
        arr = np.stack(subj_to_vecs[sid], axis=0)
        subj_vecs.append(arr.sum(axis=0))

    subj_vecs = np.stack(subj_vecs, axis=0)

    # ---------- OPTIONAL SAVE ----------
    if save_prefix is not None:
        np.save(save_prefix + "_case_vecs.npy", subj_vecs)
        with open(save_prefix + "_case_ids.txt", "w") as f:
            f.write("\n".join(subj_ids))
        print(f"💾 Saved case embeddings → {save_prefix}_case_vecs.npy")

    return subj_ids, subj_vecs


# =====================================================
# ORIENT FLMI selection
# =====================================================
def orient_select_subset(src_subj_vecs, tgt_subj_vecs,
                         src_subj_ids, budget, eta=1.0):

    from submodlib.functions.facilityLocationMutualInformation import (
        FacilityLocationMutualInformationFunction
    )
    from sklearn.metrics.pairwise import cosine_similarity

    Ns = src_subj_vecs.shape[0]
    Nt = tgt_subj_vecs.shape[0]

    raw_K = cosine_similarity(src_subj_vecs, src_subj_vecs)
    raw_Q = cosine_similarity(src_subj_vecs, tgt_subj_vecs)

    K = np.maximum(raw_K, 0).astype(np.float64)
    Q = np.maximum(raw_Q, 0).astype(np.float64)

    obj = FacilityLocationMutualInformationFunction(
        n=Ns,
        num_queries=Nt,
        data_sijs=K,
        query_sijs=Q,
        magnificationEta=eta,
    )

    # ----------- FIX: support both old/new return formats -----------
    result = obj.maximize(
        budget=min(budget, Ns),
        optimizer="LazyGreedy",
        stopIfNegativeGain=True,
        show_progress=True,
    )
    # ============================================================
    # 🔥 Fix: your submodlib returns list[ (idx, gain) ]
    # ============================================================
    selected_idx = []
    for elem in result:
        if isinstance(elem, tuple):
            selected_idx.append(elem[0])      # (idx, gain)
        else:
            selected_idx.append(elem)         # idx

    selected_idx = sorted(selected_idx)
    return [src_subj_ids[i] for i in selected_idx]


# =====================================================
# High-level wrapper
# =====================================================
def orient_select_source_subjects(
    model, device,
    base_dir,
    src_split_dir,
    tgt_split_dir,
    subset_size,
    img_size,
    batch_size,
    num_workers,
    eta=1.0,
    use_val_for_target=True,
    save_dir="results/orient_gradient"
):
    os.makedirs(save_dir, exist_ok=True)

    theta_params = get_theta_params(model)

    # ------------------------------
    # Load datasets
    # ------------------------------
    src_dataset = BraTSSliceDataset(
        root_dir=os.path.join(base_dir, "002_BraTS21"),
        split="train",
        img_size=img_size,
        split_txt_dir=src_split_dir,
    )
    src_dataset.return_meta = True

    tgt_train_dataset = BraTSSliceDataset(
        root_dir=os.path.join(base_dir, "002_BraTS21"),
        split="train",
        img_size=img_size,
        split_txt_dir=tgt_split_dir,
    )
    tgt_train_dataset.return_meta = True

    if use_val_for_target:
        tgt_val_dataset = BraTSSliceDataset(
            root_dir=os.path.join(base_dir, "002_BraTS21"),
            split="val",
            img_size=img_size,
            split_txt_dir=tgt_split_dir,
        )
        tgt_val_dataset.return_meta = True

        tgt_dataset = ConcatDataset([tgt_train_dataset, tgt_val_dataset])
    else:
        tgt_dataset = tgt_train_dataset

    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    # ------------------------------
    # Compute + save gradients
    # ------------------------------
    src_slice_g, src_slice_n = compute_gradient_features(
        model, src_loader, device, theta_params,
        save_prefix=os.path.join(save_dir, "src")
    )

    tgt_slice_g, tgt_slice_n = compute_gradient_features(
        model, tgt_loader, device, theta_params,
        save_prefix=os.path.join(save_dir, "tgt")
    )

    # ------------------------------
    # Aggregate + save
    # ------------------------------
    src_subj_ids, src_subj_vecs = aggregate_to_subject_level(
        src_slice_g, src_slice_n,
        save_prefix=os.path.join(save_dir, "src")
    )
    tgt_subj_ids, tgt_subj_vecs = aggregate_to_subject_level(
        tgt_slice_g, tgt_slice_n,
        save_prefix=os.path.join(save_dir, "tgt")
    )

    # ------------------------------
    # ORIENT selection
    # ------------------------------
    selected_ids = orient_select_subset(
        src_subj_vecs, tgt_subj_vecs, src_subj_ids,
        budget=subset_size, eta=eta
    )

    return selected_ids
