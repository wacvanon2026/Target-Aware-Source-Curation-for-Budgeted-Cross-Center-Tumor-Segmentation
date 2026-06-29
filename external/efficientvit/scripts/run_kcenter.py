#!/usr/bin/env python3
import os
import argparse
import numpy as np

def kcenter_full_ranking(X, max_rank=750, normalize=True, seed=0, eps=1e-12):
    np.random.seed(seed)
    N, D = X.shape
    K = min(max_rank, N)
    X = X.astype(np.float64)
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
    print(f' Running KCenter greedy (max_rank={K})...')
    selected = []
    gains = []
    first = np.random.randint(N)
    selected.append(first)
    dists = np.linalg.norm(X - X[first], axis=1)
    gains.append(float(np.max(dists)))
    for step in range(1, K):
        idx = np.argmax(dists)
        max_dist = float(dists[idx])
        selected.append(idx)
        gains.append(max_dist)
        new_dist = np.linalg.norm(X - X[idx], axis=1)
        dists = np.minimum(dists, new_dist)
        if step % 50 == 0:
            print(f'Step {step}/{K}')
    return (selected, gains)

def main():
    parser = argparse.ArgumentParser('KCenter for UPENN / IVYGAP / C5 / TCGA_LGG / TCGA_GBM')
    parser.add_argument('--target', required=True, choices=['UPENN', 'IVYGAP', 'C5', 'TCGA_LGG', 'TCGA_GBM'])
    parser.add_argument('--T', type=int, required=True)
    parser.add_argument('--max_rank', type=int, default=750)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    embed_root = f'external/efficientvit/data/splits_{args.target}_rds'
    out_root = f'external/efficientvit/data/splits_{args.target}_kcenter'
    os.makedirs(out_root, exist_ok=True)
    print(f'\n Loading embeddings from: {embed_root}')
    src_vecs = np.load(os.path.join(embed_root, 'src_subject_vecs.npy'))
    with open(os.path.join(embed_root, 'src_subject_ids.txt')) as f:
        src_ids = [line.strip() for line in f]
    print(f'Source subjects: {len(src_ids)}')
    selected_order, gains = kcenter_full_ranking(src_vecs, max_rank=args.max_rank, normalize=args.normalize, seed=args.seed)
    score = np.zeros(len(src_ids))
    for idx, gain in zip(selected_order, gains):
        score[idx] = float(gain)
    score_dict = {src_ids[i]: float(score[i]) for i in range(len(src_ids))}
    score_path = os.path.join(out_root, 'kcenter_score_dict.npy')
    np.save(score_path, score_dict, allow_pickle=True)
    print(f' Saved kcenter_score_dict.npy -> {score_path}')
    ordered_ids = [src_ids[i] for i in selected_order]
    with open(os.path.join(out_root, 'kcenter_sorted_ids.txt'), 'w') as f:
        f.write('\n'.join(ordered_ids))
    print(' Saved greedy order.')
    budgets_T = [1, 5, 10, 15]
    for k in budgets_T:
        budget = k * args.T
        subset_ids = [src_ids[i] for i in selected_order[:budget]]
        subset_dir = os.path.join(out_root, f'kcenter_{k}T')
        os.makedirs(subset_dir, exist_ok=True)
        with open(os.path.join(subset_dir, 'train_subjects.txt'), 'w') as f:
            f.write('\n'.join(subset_ids))
        print(f'OK Saved kcenter_{k}T ({budget})')
    print('\n KCenter completed.')
if __name__ == '__main__':
    main()
