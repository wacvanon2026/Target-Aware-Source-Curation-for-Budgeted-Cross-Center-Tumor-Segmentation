#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from scripts.orient_utils import compute_gradient_features, aggregate_to_subject_level, get_theta_params
from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
from sklearn.metrics.pairwise import cosine_similarity

def orient_full_greedy_with_gain(src_vecs, tgt_vecs, budget, eta=1.0):
    Ns = src_vecs.shape[0]
    Nt = tgt_vecs.shape[0]
    print(f'Building similarity matrices (Ns={Ns}, Nt={Nt})')
    K = np.maximum(cosine_similarity(src_vecs, src_vecs), 0).astype(np.float32)
    Q = np.maximum(cosine_similarity(src_vecs, tgt_vecs), 0).astype(np.float32)
    obj = FacilityLocationMutualInformationFunction(n=Ns, num_queries=Nt, data_sijs=K, query_sijs=Q, magnificationEta=eta)
    budget = int(min(budget, Ns))
    result = obj.maximize(budget=budget, optimizer='LazyGreedy', stopIfNegativeGain=False, show_progress=True)
    ordered_idx = []
    gains = []
    for idx, gain in result:
        ordered_idx.append(int(idx))
        gains.append(float(gain))
    return (ordered_idx, gains)

def main():
    ap = argparse.ArgumentParser('Run ORIENT (single greedy version)')
    ap.add_argument('--target', required=True, choices=['UPENN', 'IVYGAP', 'C5', 'TCGA_LGG', 'TCGA_GBM'])
    ap.add_argument('--warmup_ckpt', required=True)
    ap.add_argument('--T', type=int, required=True)
    ap.add_argument('--eta', type=float, default=1.0)
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--skip_empty', action='store_true', default=True)
    args = ap.parse_args()
    base_dir = './data'
    src_split_dir = os.path.join(base_dir, f'splits_{args.target}_source')
    if args.target == 'UPENN':
        tgt_split_dir = os.path.join(base_dir, 'split_UPENN_T150')
    elif args.target == 'IVYGAP':
        tgt_split_dir = os.path.join(base_dir, 'split_IVYGAP_T15')
    elif args.target == 'C5':
        tgt_split_dir = os.path.join(base_dir, 'split_C5_T22')
    elif args.target == 'TCGA_LGG':
        tgt_split_dir = os.path.join(base_dir, 'split_TCGA_LGG_T25')
    elif args.target == 'TCGA_GBM':
        tgt_split_dir = os.path.join(base_dir, 'split_TCGA_GBM_T40')
    else:
        raise ValueError('Unknown target')
    output_root = os.path.join(base_dir, f'splits_{args.target}_orient')
    embed_root = os.path.join('./results', f'orient_embeddings_{args.target}')
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(embed_root, exist_ok=True)
    budgets_T = [1, 5, 10, 15]
    max_budget = max(budgets_T) * args.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('🚀 Loading warmup checkpoint…')
    model = EfficientViT_Seg(backbone='efficientvit_l1', in_channels=4, num_classes=4, pretrained=False).to(device)
    state = torch.load(args.warmup_ckpt, map_location=device)
    model.load_state_dict(state['model_state'], strict=False)
    model.eval()
    theta_params = get_theta_params(model)
    src_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='train', img_size=args.img_size, split_txt_dir=src_split_dir, skip_empty=args.skip_empty)
    src_dataset.return_meta = True
    tgt_train_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='train', img_size=args.img_size, split_txt_dir=tgt_split_dir, skip_empty=args.skip_empty)
    tgt_train_dataset.return_meta = True
    tgt_val_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='val', img_size=args.img_size, split_txt_dir=tgt_split_dir, skip_empty=args.skip_empty)
    tgt_val_dataset.return_meta = True
    tgt_dataset = ConcatDataset([tgt_train_dataset, tgt_val_dataset])
    src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    tgt_loader = DataLoader(tgt_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    print('\n🧠 Computing SOURCE gradients …')
    src_slice_g, src_slice_n = compute_gradient_features(model, src_loader, device, theta_params, save_prefix=os.path.join(embed_root, 'src'))
    print('\n🎯 Computing TARGET gradients …')
    tgt_slice_g, tgt_slice_n = compute_gradient_features(model, tgt_loader, device, theta_params, save_prefix=os.path.join(embed_root, 'tgt'))
    print('\n📦 Aggregating to subject-level …')
    src_subj_ids, src_subj_vecs = aggregate_to_subject_level(src_slice_g, src_slice_n, save_prefix=os.path.join(embed_root, 'src'))
    tgt_subj_ids, tgt_subj_vecs = aggregate_to_subject_level(tgt_slice_g, tgt_slice_n, save_prefix=os.path.join(embed_root, 'tgt'))
    print('\n🧭 Running ORIENT greedy ordering (once)…')
    ordered_idx, gains = orient_full_greedy_with_gain(src_subj_vecs, tgt_subj_vecs, budget=max_budget, eta=args.eta)
    ordered_ids = [src_subj_ids[i] for i in ordered_idx]
    for k in budgets_T:
        budget = min(k * args.T, len(ordered_ids))
        selected_ids = ordered_ids[:budget]
        out_dir = os.path.join(output_root, f'orient_{k}T')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'train_subjects.txt'), 'w') as f:
            f.write('\n'.join(selected_ids))
        print(f'✅ Saved orient_{k}T ({budget})')
    orient_score_dict = {sid: 0.0 for sid in src_subj_ids}
    L = len(ordered_idx)
    for r, idx in enumerate(ordered_idx):
        sid = src_subj_ids[idx]
        orient_score_dict[sid] = 1 - r / max(L - 1, 1)
    np.save(os.path.join(output_root, 'orient_score_dict.npy'), orient_score_dict, allow_pickle=True)
    with open(os.path.join(output_root, 'orient_sorted_ids.txt'), 'w') as f:
        f.write('\n'.join(ordered_ids))
    print('\n🎉 ORIENT completed (single greedy version).')
if __name__ == '__main__':
    main()
