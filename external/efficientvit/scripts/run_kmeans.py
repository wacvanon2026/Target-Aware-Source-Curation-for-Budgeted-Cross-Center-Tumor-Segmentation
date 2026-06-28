#!/usr/bin/env python3
import os
import argparse
import numpy as np
from sklearn.cluster import KMeans

def kmeans_full_ranking(X, max_rank=500, normalize=True, seed=0, eps=1e-12):
    N, D = X.shape
    K = min(max_rank, N)
    X = X.astype(np.float64)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    print(f'🔹 Running KMeans clustering (K={K})...')
    kmeans = KMeans(n_clusters=K, random_state=seed, n_init=10, max_iter=300)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    print('🔹 Selecting cluster representatives...')
    selected = []
    cluster_sizes = []
    for c in range(K):
        cluster_idx = np.where(labels == c)[0]
        size = len(cluster_idx)
        if size == 0:
            continue
        cluster_points = X[cluster_idx]
        center = centers[c]
        dists = np.linalg.norm(cluster_points - center, axis=1)
        best_local = cluster_idx[np.argmin(dists)]
        selected.append(best_local)
        cluster_sizes.append(size)
    return (selected, cluster_sizes)

def main():
    parser = argparse.ArgumentParser('KMeans for UPENN / IVYGAP / C5 / TCGA_LGG / TCGA_GBM')
    parser.add_argument('--target', required=True, choices=['UPENN', 'IVYGAP', 'C5', 'TCGA_LGG', 'TCGA_GBM'])
    parser.add_argument('--T', type=int, required=True)
    parser.add_argument('--max_rank', type=int, default=500)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    embed_root = f'././data/splits_{args.target}_rds'
    out_root = f'././data/splits_{args.target}_kmeans'
    os.makedirs(out_root, exist_ok=True)
    print(f'\n📂 Loading embeddings from: {embed_root}')
    src_vecs = np.load(os.path.join(embed_root, 'src_subject_vecs.npy'))
    with open(os.path.join(embed_root, 'src_subject_ids.txt')) as f:
        src_ids = [line.strip() for line in f]
    print(f'Source subjects: {len(src_ids)}')
    selected_order, cluster_sizes = kmeans_full_ranking(src_vecs, max_rank=args.max_rank, normalize=args.normalize, seed=args.seed)
    pairs = list(zip(selected_order, cluster_sizes))
    pairs.sort(key=lambda x: x[1], reverse=True)
    selected_order = [p[0] for p in pairs]
    score = np.zeros(len(src_ids))
    N = len(selected_order)
    for rank, idx in enumerate(selected_order):
        score[idx] = 1.0 - rank / (N - 1)
    score_dict = {src_ids[i]: float(score[i]) for i in range(len(src_ids))}
    np.save(os.path.join(out_root, 'kmeans_score_dict.npy'), score_dict, allow_pickle=True)
    print('💾 Saved kmeans_score_dict.npy')
    ordered_ids = [src_ids[i] for i in selected_order]
    with open(os.path.join(out_root, 'kmeans_sorted_ids.txt'), 'w') as f:
        f.write('\n'.join(ordered_ids))
    budgets_T = [1, 5, 10]
    for k in budgets_T:
        budget = k * args.T
        subset_ids = ordered_ids[:budget]
        subset_dir = os.path.join(out_root, f'kmeans_{k}T')
        os.makedirs(subset_dir, exist_ok=True)
        with open(os.path.join(subset_dir, 'train_subjects.txt'), 'w') as f:
            f.write('\n'.join(subset_ids))
        print(f'✅ Saved kmeans_{k}T ({budget})')
    print('\n🎉 KMeans completed.')
if __name__ == '__main__':
    main()
