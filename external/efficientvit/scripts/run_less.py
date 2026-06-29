#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
torch.multiprocessing.set_sharing_strategy('file_system')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
import torch.nn.functional as F
import torch.nn as nn

class DiceCELoss(nn.Module):

    def __init__(self, smooth=1e-05, weight_ce=0.5):
        super().__init__()
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        preds_soft = F.softmax(preds, dim=1)
        targets_onehot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (preds_soft * targets_onehot).sum(dim=(2, 3))
        dice = (2.0 * intersection + self.smooth) / (preds_soft.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3)) + self.smooth)
        dice_loss = 1 - dice.mean()
        return self.weight_ce * ce_loss + (1 - self.weight_ce) * dice_loss

def get_theta_params(model):
    return list(model.final_head.parameters())

def compute_gradient_features(model, loader, device, theta_params):
    loss_fn = DiceCELoss()
    model.eval()
    grads = []
    names = []
    for img, lbl, subject_ids, slice_idxs in tqdm(loader):
        img = img.to(device)
        lbl = lbl.to(device)
        logits = model(img)
        loss = loss_fn(logits, lbl)
        model.zero_grad()
        loss.backward()
        g = []
        for p in theta_params:
            if p.grad is not None:
                g.append(p.grad.view(-1))
        g = torch.cat(g).detach().cpu().numpy()
        grads.append(g)
        for sid, sidx in zip(subject_ids, slice_idxs):
            names.append(f'{sid}_slice{sidx.item()}')
    grads = np.stack(grads, axis=0)
    return (grads, names)

def compute_less_slice_scores(src_grads, tgt_grads, beta=20.0, batch=1000):
    src_norm = src_grads / (np.linalg.norm(src_grads, axis=1, keepdims=True) + 1e-08)
    tgt_norm = tgt_grads / (np.linalg.norm(tgt_grads, axis=1, keepdims=True) + 1e-08)
    N = src_norm.shape[0]
    scores = np.zeros(N, dtype=np.float32)
    for start in tqdm(range(0, N, batch)):
        end = min(start + batch, N)
        sim = cosine_similarity(src_norm[start:end], tgt_norm)
        W = np.exp(beta * sim)
        W /= W.sum(axis=1, keepdims=True) + 1e-08
        scores[start:end] = (W * sim).sum(axis=1)
    return scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=['UPENN', 'IVYGAP', 'C5', 'TCGA_LGG', 'TCGA_GBM'])
    parser.add_argument('--warmup_ckpt', required=True)
    parser.add_argument('--T', type=int, required=True)
    parser.add_argument('--beta', type=float, default=20.0)
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
    out_root = os.path.join(base_dir, f'splits_{args.target}_less')
    os.makedirs(out_root, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EfficientViT_Seg(backbone='efficientvit_l1', in_channels=4, num_classes=4, pretrained=False)
    state = torch.load(args.warmup_ckpt, map_location=device)
    model.load_state_dict(state['model_state'], strict=False)
    model = model.to(device)
    theta_params = get_theta_params(model)
    img_size = 512
    SKIP_EMPTY = True
    src_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='train', img_size=img_size, split_txt_dir=src_split_dir, skip_empty=SKIP_EMPTY)
    tgt_train_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='train', img_size=img_size, split_txt_dir=tgt_split_dir, skip_empty=SKIP_EMPTY)
    tgt_val_dataset = BraTSSliceDataset(root_dir=os.path.join(base_dir, '002_BraTS21'), split='val', img_size=img_size, split_txt_dir=tgt_split_dir, skip_empty=SKIP_EMPTY)
    src_loader = DataLoader(src_dataset, batch_size=1, shuffle=False, num_workers=4)
    tgt_train_loader = DataLoader(tgt_train_dataset, batch_size=1, shuffle=False, num_workers=4)
    tgt_val_loader = DataLoader(tgt_val_dataset, batch_size=1, shuffle=False, num_workers=4)
    print(' Extracting SOURCE gradients...')
    src_grads, src_names = compute_gradient_features(model, src_loader, device, theta_params)
    print(' Extracting TARGET gradients...')
    tgt_train_grads, tgt_train_names = compute_gradient_features(model, tgt_train_loader, device, theta_params)
    tgt_val_grads, tgt_val_names = compute_gradient_features(model, tgt_val_loader, device, theta_params)
    tgt_grads = np.concatenate([tgt_train_grads, tgt_val_grads], axis=0)
    tgt_names = tgt_train_names + tgt_val_names
    print(f'SRC slice count: {len(src_names)}')
    print(f'TGT slice count: {len(tgt_names)}')
    np.save(os.path.join(out_root, 'src_slice_grads.npy'), src_grads)
    np.save(os.path.join(out_root, 'tgt_slice_grads.npy'), tgt_grads)
    print(' Computing LESS scores...')
    slice_scores = compute_less_slice_scores(src_grads, tgt_grads, beta=args.beta)
    pid_to_scores = defaultdict(list)
    for name, s in zip(src_names, slice_scores):
        pid = name.split('_slice')[0]
        pid_to_scores[pid].append(s)
    less_score_dict = {pid: float(np.mean(scores)) for pid, scores in pid_to_scores.items()}
    print(f'SRC subject count (after aggregation): {len(less_score_dict)}')
    np.save(os.path.join(out_root, 'less_score_dict.npy'), less_score_dict, allow_pickle=True)
    ranked = sorted(less_score_dict.items(), key=lambda x: x[1], reverse=True)
    ranked_ids = [pid for pid, _ in ranked]
    with open(os.path.join(out_root, 'less_sorted_ids.txt'), 'w') as f:
        f.write('\n'.join(ranked_ids))
    budgets = [1, 5, 10, 15]
    for k in budgets:
        n = min(k * args.T, len(ranked_ids))
        selected = ranked_ids[:n]
        out_dir = os.path.join(out_root, f'less_{k}T')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, 'train_subjects.txt'), 'w') as f:
            f.write('\n'.join(selected))
        print(f'OK Saved LESS {k}T ({n} subjects)')
    print('\n LESS pipeline completed successfully.')
if __name__ == '__main__':
    main()
