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
from models.feature_extractor import FeatureExtractor
from scripts.rds_utils import extract_slice_embeddings, aggregate_to_subject, compute_rds_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=['UPENN', 'IVYGAP', 'C5', 'TCGA_LGG', 'TCGA_GBM'])
    parser.add_argument('--warmup_ckpt', required=True)
    parser.add_argument('--T', type=int, required=True)
    args = parser.parse_args()
    base_dir = 'external/efficientvit/data'
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
    output_root = os.path.join(base_dir, f'splits_{args.target}_rds')
    os.makedirs(output_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' Loading warmup model...')
    model = EfficientViT_Seg(backbone='efficientvit_l1', in_channels=4, num_classes=4, pretrained=False).to(device)
    state = torch.load(args.warmup_ckpt, map_location=device)
    model.load_state_dict(state['model_state'], strict=False)
    model.eval()
    model = FeatureExtractor(model).to(device)
    print('OK Warmup loaded.')
    img_size = 512
    batch_size = 4
    num_workers = 4
    SKIP_EMPTY = True
    src_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='train', img_size=img_size, split_txt_dir=src_split_dir, skip_empty=SKIP_EMPTY)
    src_dataset.return_meta = True
    tgt_train_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='train', img_size=img_size, split_txt_dir=tgt_split_dir, skip_empty=SKIP_EMPTY)
    tgt_train_dataset.return_meta = True
    tgt_val_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='val', img_size=img_size, split_txt_dir=tgt_split_dir, skip_empty=SKIP_EMPTY)
    tgt_val_dataset.return_meta = True
    tgt_dataset = ConcatDataset([tgt_train_dataset, tgt_val_dataset])
    src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    tgt_loader = DataLoader(tgt_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(' Extracting SOURCE embeddings...')
    src_slice_emb, src_slice_names = extract_slice_embeddings(model, src_loader, device)
    print(' Extracting TARGET embeddings...')
    tgt_slice_emb, tgt_slice_names = extract_slice_embeddings(model, tgt_loader, device)
    np.save(os.path.join(output_root, 'src_slice_emb.npy'), src_slice_emb)
    np.save(os.path.join(output_root, 'tgt_slice_emb.npy'), tgt_slice_emb)
    with open(os.path.join(output_root, 'src_slice_names.txt'), 'w') as f:
        f.write('\n'.join(src_slice_names))
    with open(os.path.join(output_root, 'tgt_slice_names.txt'), 'w') as f:
        f.write('\n'.join(tgt_slice_names))
    print(' Saved slice embeddings + names.')
    src_ids, src_vecs = aggregate_to_subject(src_slice_emb, src_slice_names)
    tgt_ids, tgt_vecs = aggregate_to_subject(tgt_slice_emb, tgt_slice_names)
    np.save(os.path.join(output_root, 'src_subject_vecs.npy'), src_vecs)
    np.save(os.path.join(output_root, 'tgt_subject_vecs.npy'), tgt_vecs)
    with open(os.path.join(output_root, 'src_subject_ids.txt'), 'w') as f:
        f.write('\n'.join(src_ids))
    print(' Computing RDS scores...')
    scores = compute_rds_scores(src_vecs, tgt_vecs)
    np.save(os.path.join(output_root, 'rds_scores.npy'), scores)
    sorted_idx = np.argsort(scores)[::-1]
    sorted_ids = [src_ids[i] for i in sorted_idx]
    np.save(os.path.join(output_root, 'rds_sorted_idx.npy'), sorted_idx)
    with open(os.path.join(output_root, 'rds_sorted_ids.txt'), 'w') as f:
        f.write('\n'.join(sorted_ids))
    score_dict = {src_ids[i]: float(scores[i]) for i in range(len(src_ids))}
    np.save(os.path.join(output_root, 'rds_score_dict.npy'), score_dict, allow_pickle=True)
    print(' Saved full ranking + score dict.')
    budgets = [1, 5, 10, 15]
    for k in budgets:
        n = min(k * args.T, len(sorted_ids))
        selected = sorted_ids[:n]
        out_dir = os.path.join(output_root, f'rds_{k}T')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'train_subjects.txt'), 'w') as f:
            f.write('\n'.join(selected))
        print(f'OK Saved rds_{k}T ({n} subjects)')
    print('\n RDS generation complete.')
if __name__ == '__main__':
    main()
