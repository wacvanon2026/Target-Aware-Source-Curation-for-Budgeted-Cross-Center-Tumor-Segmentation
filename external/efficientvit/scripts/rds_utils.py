#!/usr/bin/env python3
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity

def extract_slice_embeddings(model, loader, device):
    feats = []
    names = []
    model.eval()
    with torch.no_grad():
        for img, lbl, subj_ids, slice_idxs in tqdm(loader, desc='RDS+: extracting embeddings'):
            img = img.to(device)
            emb = model(img)
            emb = torch.nn.functional.adaptive_avg_pool2d(emb, (1, 1))
            emb = emb.view(emb.size(0), -1)
            feats.append(emb.cpu().numpy())
            for sid, sidx in zip(subj_ids, slice_idxs):
                names.append(f'{sid}_slice{sidx.item()}')
    feats = np.concatenate(feats, axis=0)
    return (feats, names)

def aggregate_to_subject(feats, names, agg='mean'):
    subj_to_vecs = defaultdict(list)
    for f, name in zip(feats, names):
        subj = name.split('_slice')[0]
        subj_to_vecs[subj].append(f)
    subj_ids = sorted(subj_to_vecs.keys())
    subj_vecs = []
    for sid in subj_ids:
        arr = np.stack(subj_to_vecs[sid], axis=0)
        if agg == 'mean':
            subj_vecs.append(arr.mean(axis=0))
        elif agg == 'max':
            subj_vecs.append(arr.max(axis=0))
        elif agg == 'sum':
            subj_vecs.append(arr.sum(axis=0))
        else:
            raise ValueError('unknown aggregation')
    subj_vecs = np.stack(subj_vecs, axis=0)
    return (subj_ids, subj_vecs)

def compute_rds_scores(src_subj_vecs, tgt_subj_vecs, batch=1024):
    N = src_subj_vecs.shape[0]
    D = src_subj_vecs.shape[1]
    scores = np.zeros(N, dtype=np.float32)
    src_norm = src_subj_vecs / (np.linalg.norm(src_subj_vecs, axis=1, keepdims=True) + 1e-08)
    tgt_norm = tgt_subj_vecs / (np.linalg.norm(tgt_subj_vecs, axis=1, keepdims=True) + 1e-08)
    print(' Computing subject-level RDS+ scores...')
    for start in tqdm(range(0, N, batch)):
        end = min(start + batch, N)
        S = src_norm[start:end]
        sim = cosine_similarity(S, tgt_norm)
        scores[start:end] = np.max(sim, axis=1)
    return scores

def rds_select_subjects(src_subj_ids, scores, budget):
    idx = np.argsort(scores)[::-1]
    idx = idx[:budget]
    return [src_subj_ids[i] for i in idx]
