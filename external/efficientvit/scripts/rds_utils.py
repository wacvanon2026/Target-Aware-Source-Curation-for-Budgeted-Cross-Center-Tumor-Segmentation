#!/usr/bin/env python3
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 1. Slice-level embedding extraction (FeatureExtractor)
# ============================================================
def extract_slice_embeddings(model, loader, device):
    """
    Returns:
        feats: (N_slices, D)
        names: list of "subjid_sliceX"
    """
    feats = []
    names = []

    model.eval()
    with torch.no_grad():
        for img, lbl, subj_ids, slice_idxs in tqdm(loader, desc="RDS+: extracting embeddings"):

            img = img.to(device)
            emb = model(img)                      # (B,C,H,W)
            emb = torch.nn.functional.adaptive_avg_pool2d(emb, (1,1))  # (B,C,1,1)
            emb = emb.view(emb.size(0), -1)      # (B,C)

            feats.append(emb.cpu().numpy())

            for sid, sidx in zip(subj_ids, slice_idxs):
                names.append(f"{sid}_slice{sidx.item()}")

    feats = np.concatenate(feats, axis=0)
    return feats, names


# ============================================================
# 2. Aggregate to subject-level
# ============================================================
def aggregate_to_subject(feats, names, agg="mean"):
    """
    feats: (N_slices, D)
    names: ["UPENN-GBM-xxx_slice0", ...]
    Return:
        subj_ids (sorted list)
        subj_vecs (N_subjects, D)
    """
    subj_to_vecs = defaultdict(list)

    for f, name in zip(feats, names):
        subj = name.split("_slice")[0]
        subj_to_vecs[subj].append(f)

    subj_ids = sorted(subj_to_vecs.keys())
    subj_vecs = []

    for sid in subj_ids:
        arr = np.stack(subj_to_vecs[sid], axis=0)
        if agg == "mean":
            subj_vecs.append(arr.mean(axis=0))
        elif agg == "max":
            subj_vecs.append(arr.max(axis=0))
        elif agg == "sum":
            subj_vecs.append(arr.sum(axis=0))
        else:
            raise ValueError("unknown aggregation")

    subj_vecs = np.stack(subj_vecs, axis=0)
    return subj_ids, subj_vecs


# ============================================================
# 3. RDS+ similarity scoring
# ============================================================
def compute_rds_scores(src_subj_vecs, tgt_subj_vecs, batch=1024):
    """
    RDS+ standard max similarity on subject-level embeddings:

        score_i = max_j cos(src_i, tgt_j)

    Returns:
        scores: (N_src_subjects,)
    """
    N = src_subj_vecs.shape[0]
    D = src_subj_vecs.shape[1]

    scores = np.zeros(N, dtype=np.float32)

    # normalize
    src_norm = src_subj_vecs / (np.linalg.norm(src_subj_vecs, axis=1, keepdims=True) + 1e-8)
    tgt_norm = tgt_subj_vecs / (np.linalg.norm(tgt_subj_vecs, axis=1, keepdims=True) + 1e-8)

    print("🔍 Computing subject-level RDS+ scores…")

    for start in tqdm(range(0, N, batch)):
        end = min(start + batch, N)
        S = src_norm[start:end]                # (B, D)

        sim = cosine_similarity(S, tgt_norm)   # (B, Nt)
        scores[start:end] = np.max(sim, axis=1)

    return scores


# ============================================================
# 4. High-level subject subset selection
# ============================================================
def rds_select_subjects(src_subj_ids, scores, budget):
    """
    Sort by descending RDS+ score and return top K subjects.
    """
    idx = np.argsort(scores)[::-1]              # descending
    idx = idx[:budget]
    return [src_subj_ids[i] for i in idx]
