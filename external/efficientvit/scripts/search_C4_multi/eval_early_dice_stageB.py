#!/usr/bin/env python3
import os
import sys
import json
import yaml
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
import sys
EFFICIENTVIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EFFICIENTVIT_ROOT))
sys.path.insert(0, str(EFFICIENTVIT_ROOT))
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset

def dice_3d(pred, gt):
    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    inter = (pred & gt).sum()
    return 2 * inter / (pred.sum() + gt.sum() + 1e-08)

@torch.no_grad()
def evaluate_early_dice_2d_to_3d(config_path: str, checkpoint_path: str, max_subjects: int=20, val_dataset=None):
    print(' Loading config:', config_path)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' Using device:', device)
    model_cfg = cfg['model']
    model = EfficientViT_Seg(backbone=model_cfg['name'], in_channels=model_cfg['in_channels'], num_classes=model_cfg['num_classes'], pretrained=False).to(device)
    print(' Loading checkpoint:', checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    model.eval()
    data_cfg = cfg['data']['val']
    img_size = cfg['data']['img_size']
    if val_dataset is not None:
        print(' Using prebuilt validation dataset')
        dataset = val_dataset
    else:
        print(' Preparing validation dataset...')
        dataset = BraTSSliceDataset(root_dir=data_cfg['path'], split=data_cfg['split'], img_size=img_size, split_txt_dir=data_cfg.get('split_txt'), skip_empty=False)
    subject_to_slices = defaultdict(list)
    for i in range(len(dataset)):
        _, _, subject_id, slice_idx = dataset.samples[i]
        subject_to_slices[subject_id].append((i, slice_idx))
    val_size = len(subject_to_slices)
    all_subjects = list(subject_to_slices.items())
    rng = np.random.default_rng(0)
    rng.shuffle(all_subjects)
    if max_subjects is not None:
        all_subjects = all_subjects[:max_subjects]
    used_subjects = len(all_subjects)
    is_full_val = used_subjects >= val_size
    print(f' Evaluating {used_subjects}/{val_size} subjects (early Dice)')
    dice_ET = dice_TC = dice_WT = 0.0
    N = 0
    for subject_id, slice_list in tqdm(all_subjects, desc='Early Dice Eval', ncols=100):
        max_slice = max((z for _, z in slice_list))
        vol_pred = np.zeros((240, 240, max_slice + 1), dtype=np.uint8)
        vol_gt = np.zeros_like(vol_pred)
        for dataset_idx, z in slice_list:
            img, lbl, _, _ = dataset[dataset_idx]
            img = img.unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(enabled=False):
                out = model(img)
            pred = torch.argmax(out, dim=1)[0].cpu().numpy()
            pred = F.interpolate(torch.from_numpy(pred)[None, None].float(), size=(240, 240), mode='nearest')[0, 0].numpy()
            lbl_np = lbl.numpy()
            lbl_np = F.interpolate(torch.from_numpy(lbl_np)[None, None].float(), size=(240, 240), mode='nearest')[0, 0].numpy()
            vol_pred[:, :, z] = pred
            vol_gt[:, :, z] = lbl_np
        et_pred = vol_pred == 3
        et_gt = vol_gt == 3
        tc_pred = np.isin(vol_pred, [2, 3])
        tc_gt = np.isin(vol_gt, [2, 3])
        wt_pred = vol_pred > 0
        wt_gt = vol_gt > 0
        dice_ET += dice_3d(et_pred, et_gt)
        dice_TC += dice_3d(tc_pred, tc_gt)
        dice_WT += dice_3d(wt_pred, wt_gt)
        N += 1
        del vol_pred, vol_gt
        torch.cuda.empty_cache()
    dET = dice_ET / max(N, 1)
    dTC = dice_TC / max(N, 1)
    dWT = dice_WT / max(N, 1)
    macro_avg = (dET + dTC + dWT) / 3.0
    return {'dice_ET': float(dET), 'dice_TC': float(dTC), 'dice_WT': float(dWT), 'dice_macro': float(macro_avg), 'num_subjects': int(N), 'val_size': int(val_size), 'max_subjects': int(max_subjects), 'is_full_val': bool(is_full_val)}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Early Dice Evaluation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_subjects', type=int, default=15)
    parser.add_argument('--out_json', type=str, required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(' Running early Dice evaluation...')
    results = evaluate_early_dice_2d_to_3d(args.config, args.checkpoint, max_subjects=args.max_subjects)
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(results, indent=2))
    print('\n================== EARLY DICE ==================')
    print(f"Dice_ET    : {results['dice_ET']:.4f}")
    print(f"Dice_TC    : {results['dice_TC']:.4f}")
    print(f"Dice_WT    : {results['dice_WT']:.4f}")
    print(f"Macro Avg  : {results['dice_macro']:.4f}")
    print(f"Subjects   : {results['num_subjects']}")
    print('================================================')
    print(f' Saved to {out_path}')
