#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import ndimage
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
REGIONS = {'ET': (3,), 'TC': (2, 3), 'WT': (1, 2, 3)}
METRICS = ('dice', 'hd95', 'precision', 'recall')

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--metrics-dir', type=Path, default=None, help='Directory that will receive per_case_metrics.csv, summary_metrics.json, and run_metadata.json.')
    parser.add_argument('--target-domain', default=None)
    parser.add_argument('--method', default=None)
    parser.add_argument('--budget', type=int, default=None)
    parser.add_argument('--train-seed', default=None)
    parser.add_argument('--search-seed', default=None)
    parser.add_argument('--subset-seed', default=None)
    parser.add_argument('--checkpoint-name', default=None)
    parser.add_argument('--legacy-results', type=Path, default=None)
    parser.add_argument('--save-pred', action='store_true')
    parser.add_argument('--pred-dir', type=Path, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    return parser.parse_args()

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r') as f:
        return yaml.safe_load(f)

def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        return ckpt
    return {'model_state': ckpt}

def infer_target_domain(*parts: str) -> str | None:
    joined = ' '.join([p for p in parts if p])
    for name in ('TCGA_LGG', 'TCGA_GBM', 'C4', 'C5', 'UPENN', 'IVYGAP', 'TCGA'):
        if re.search(f'(^|[^A-Za-z0-9]){re.escape(name)}([^A-Za-z0-9]|$)', joined):
            return name
    return None

def infer_budget(*parts: str) -> int | None:
    joined = ' '.join([p for p in parts if p])
    m = re.search('(?<![A-Za-z0-9])K(\\d+)(?![A-Za-z0-9])', joined, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search('(?<![A-Za-z0-9])(\\d+)T(?![A-Za-z0-9])', joined, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) * 50
    return None

def infer_method(*parts: str) -> str | None:
    joined = ' '.join([p.lower() for p in parts if p])
    if '8d' in joined and ('cma' in joined or 'stagea2' in joined):
        return 'TAVO_8D'
    if '3d' in joined and ('cma' in joined or 'stagea2' in joined):
        return 'TAVO_3D'
    checks = [('rdsplus_kmeans', 'RDSPlus_KMeans'), ('rdsplus', 'RDSPlus'), ('gradmatch', 'GradMatch'), ('diversity', 'Diversity'), ('kcenter', 'KCenter'), ('kmeans', 'KMeans'), ('orient', 'ORIENT'), ('random3', 'Random3'), ('random2', 'Random2'), ('random1', 'Random1'), ('random', 'Random'), ('craig', 'CRAIG'), ('less', 'LESS'), ('source_plus_target', 'FullSourceTarget'), ('s_and_t', 'FullSourceTarget'), ('target_train', 'TargetOnly'), ('source', 'SourceOnly')]
    for needle, method in checks:
        if needle in joined:
            return method
    return None

def infer_seed(pattern: str, *parts: str) -> str | None:
    joined = ' '.join([p for p in parts if p])
    m = re.search(pattern, joined, flags=re.IGNORECASE)
    return m.group(1) if m else None

def as_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    return obj

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return 1.0
    denom = pred_sum + gt_sum
    if denom == 0:
        return 0.0
    tp = int(np.logical_and(pred, gt).sum())
    return float(2.0 * tp / denom)

def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    pred_sum = tp + fp
    gt_sum = tp + fn
    if pred_sum == 0 and gt_sum == 0:
        precision = 1.0
        recall = 1.0
    else:
        precision = float(tp / pred_sum) if pred_sum > 0 else 0.0
        recall = float(tp / gt_sum) if gt_sum > 0 else 1.0
    return (precision, recall, tp, fp, fn)

def surface_mask(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(mask.ndim, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)

def hd95_score(pred: np.ndarray, gt: np.ndarray) -> tuple[float, bool, str]:
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    if pred_sum == 0 and gt_sum == 0:
        return (0.0, True, 'both_empty')
    if pred_sum == 0 or gt_sum == 0:
        diagonal = float(np.linalg.norm(np.maximum(np.asarray(pred.shape) - 1, 0)))
        if pred_sum == 0:
            return (diagonal, False, 'pred_empty')
        return (diagonal, False, 'gt_empty')
    pred_surface = surface_mask(pred)
    gt_surface = surface_mask(gt)
    pred_to_gt = ndimage.distance_transform_edt(np.logical_not(gt_surface))[pred_surface]
    gt_to_pred = ndimage.distance_transform_edt(np.logical_not(pred_surface))[gt_surface]
    distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64, copy=False)
    if distances.size == 0:
        return (0.0, True, 'surface_empty')
    return (float(np.percentile(distances, 95)), True, 'non_empty')

def region_mask(volume: np.ndarray, labels: tuple[int, ...]) -> np.ndarray:
    if len(labels) == 1:
        return volume == labels[0]
    return np.isin(volume, labels)

def compute_region_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    precision, recall, tp, fp, fn = precision_recall(pred, gt)
    hd95, hd95_valid, empty_case = hd95_score(pred, gt)
    return {'dice': dice_score(pred, gt), 'hd95': hd95, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn, 'pred_voxels': int(pred.sum()), 'gt_voxels': int(gt.sum()), 'hd95_valid': bool(hd95_valid), 'empty_case': empty_case}

def build_model(cfg: dict[str, Any], device: torch.device) -> EfficientViT_Seg:
    model_cfg = cfg['model']
    model = EfficientViT_Seg(backbone=model_cfg['name'], in_channels=model_cfg['in_channels'], num_classes=model_cfg['num_classes'], pretrained=model_cfg.get('pretrained', False)).to(device)
    return model

def split_file_exists(split_txt_dir: str | None, split: str) -> bool:
    if not split_txt_dir:
        return False
    return (Path(split_txt_dir) / f'{split}_subjects.txt').exists()

def resolve_split_txt_dir(split_txt_dir: str | None, split: str) -> str | None:
    if not split_txt_dir:
        return split_txt_dir
    if split_file_exists(split_txt_dir, split):
        return split_txt_dir
    candidates = []
    text = str(split_txt_dir)
    marker = '/EfficientVit/data/'
    if marker in text:
        candidates.append(text.replace(marker, '/EfficientVit/data/TAVO/', 1))
    path = Path(text)
    parts = path.parts
    if 'data' in parts:
        idx = parts.index('data')
        if idx + 1 < len(parts) and parts[idx + 1] != 'TAVO':
            candidates.append(str(Path(*parts[:idx + 1]) / 'TAVO' / Path(*parts[idx + 1:])))
    for candidate in candidates:
        if split_file_exists(candidate, split):
            print(f'Remapped split path for {split}: {split_txt_dir} -> {candidate}')
            return candidate
    return split_txt_dir

def build_test_dataset(cfg: dict[str, Any]) -> BraTSSliceDataset:
    data_cfg = cfg['data']['test']
    split = data_cfg['split']
    split_txt_dir = resolve_split_txt_dir(data_cfg.get('split_txt'), split)
    if not split_file_exists(split_txt_dir, split):
        missing = Path(split_txt_dir or '') / f'{split}_subjects.txt'
        raise FileNotFoundError(f'Required test split file is missing: {missing}. Refusing to evaluate all subjects accidentally.')
    data_cfg['resolved_split_txt'] = split_txt_dir
    return BraTSSliceDataset(root_dir=data_cfg['path'], split=split, img_size=cfg['data']['img_size'], split_txt_dir=split_txt_dir, skip_empty=False)

def group_subject_slices(dataset: BraTSSliceDataset) -> dict[str, list[tuple[int, int]]]:
    subject_to_slices: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for i in range(len(dataset)):
        _, _, subject_id, slice_idx = dataset.samples[i]
        subject_to_slices[subject_id].append((i, int(slice_idx)))
    for subject_id in subject_to_slices:
        subject_to_slices[subject_id].sort(key=lambda x: x[1])
    return dict(subject_to_slices)

def reconstruct_subject(subject_id: str, slice_list: list[tuple[int, int]], dataset: BraTSSliceDataset, model: EfficientViT_Seg, device: torch.device, use_amp: bool) -> tuple[np.ndarray, np.ndarray]:
    max_slice = max((z for _, z in slice_list))
    vol_pred = np.zeros((240, 240, max_slice + 1), dtype=np.uint8)
    vol_gt = np.zeros_like(vol_pred)
    for dataset_idx, z in slice_list:
        img, lbl, _, _ = dataset[dataset_idx]
        img = img.unsqueeze(0).to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=use_amp):
            out = model(img)
        pred = torch.argmax(out, dim=1)[0].detach().cpu().numpy()
        pred = F.interpolate(torch.from_numpy(pred)[None, None].float(), size=(240, 240), mode='nearest')[0, 0].numpy()
        lbl_np = F.interpolate(torch.from_numpy(lbl.numpy())[None, None].float(), size=(240, 240), mode='nearest')[0, 0].numpy()
        vol_pred[:, :, z] = pred.astype(np.uint8)
        vol_gt[:, :, z] = lbl_np.astype(np.uint8)
    return (vol_pred, vol_gt)

def make_case_row(subject_id: str, vol_pred: np.ndarray, vol_gt: np.ndarray, metadata: dict[str, Any], num_slices: int) -> dict[str, Any]:
    row: dict[str, Any] = {'case_id': subject_id, 'target_domain': metadata.get('target_domain'), 'method': metadata.get('method'), 'budget': metadata.get('budget'), 'train_seed': metadata.get('train_seed'), 'search_seed': metadata.get('search_seed'), 'subset_seed': metadata.get('subset_seed'), 'checkpoint': metadata.get('checkpoint_name'), 'epoch': metadata.get('epoch'), 'num_slices': num_slices, 'volume_shape': 'x'.join((str(x) for x in vol_pred.shape))}
    region_metrics: dict[str, dict[str, Any]] = {}
    for region_name, labels in REGIONS.items():
        pred_mask = region_mask(vol_pred, labels)
        gt_mask = region_mask(vol_gt, labels)
        metrics = compute_region_metrics(pred_mask, gt_mask)
        region_metrics[region_name] = metrics
        for metric_name in METRICS:
            row[f'{metric_name}_{region_name}'] = metrics[metric_name]
        for count_key in ('tp', 'fp', 'fn', 'pred_voxels', 'gt_voxels'):
            row[f'{count_key}_{region_name}'] = metrics[count_key]
        row[f'hd95_valid_{region_name}'] = metrics['hd95_valid']
        row[f'empty_case_{region_name}'] = metrics['empty_case']
    for metric_name in METRICS:
        row[f'{metric_name}_macro'] = float(np.mean([region_metrics[name][metric_name] for name in REGIONS]))
    row['hd95_units'] = 'voxel'
    return row

def csv_columns() -> list[str]:
    base = ['case_id', 'target_domain', 'method', 'budget', 'train_seed', 'search_seed', 'subset_seed', 'checkpoint', 'epoch', 'num_slices', 'volume_shape']
    metric_cols = []
    for metric_name in METRICS:
        metric_cols.extend([f'{metric_name}_{name}' for name in REGIONS])
        metric_cols.append(f'{metric_name}_macro')
    count_cols = []
    for count_key in ('tp', 'fp', 'fn', 'pred_voxels', 'gt_voxels'):
        count_cols.extend([f'{count_key}_{name}' for name in REGIONS])
    flag_cols = []
    for name in REGIONS:
        flag_cols.append(f'hd95_valid_{name}')
        flag_cols.append(f'empty_case_{name}')
    return base + metric_cols + count_cols + flag_cols + ['hd95_units']

def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = csv_columns()
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {'num_cases': len(rows), 'metrics': {}, 'hd95_valid_counts': {}}
    for metric_name in METRICS:
        summary['metrics'][metric_name] = {}
        for region_name in list(REGIONS) + ['macro']:
            key = f'{metric_name}_{region_name}'
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            summary['metrics'][metric_name][region_name] = {'mean': float(values.mean()) if values.size else None, 'std': float(values.std(ddof=1)) if values.size > 1 else 0.0, 'min': float(values.min()) if values.size else None, 'max': float(values.max()) if values.size else None}
    for region_name in REGIONS:
        summary['hd95_valid_counts'][region_name] = int(sum((bool(row[f'hd95_valid_{region_name}']) for row in rows)))
    return summary

def parse_legacy_results(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    mapping = {'Dice_ET': 'ET', 'Dice_TC': 'TC', 'Dice_WT': 'WT', 'Macro Avg': 'macro'}
    results: dict[str, float] = {}
    for line in path.read_text().splitlines():
        if ':' not in line:
            continue
        name, value = [part.strip() for part in line.split(':', 1)]
        if name in mapping:
            try:
                results[mapping[name]] = float(value)
            except ValueError:
                pass
    return results

def attach_legacy_comparison(summary: dict[str, Any], legacy: dict[str, float] | None) -> None:
    if not legacy:
        return
    comparison = {}
    dice_summary = summary['metrics']['dice']
    for region_name, old_value in legacy.items():
        if region_name not in dice_summary:
            continue
        new_value = dice_summary[region_name]['mean']
        comparison[region_name] = {'legacy': old_value, 'current': new_value, 'abs_diff': abs(float(new_value) - float(old_value))}
    summary['legacy_dice_comparison'] = comparison

def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(as_jsonable(payload), f, indent=2)

def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    requested_device = args.device
    if requested_device == 'cuda' and (not torch.cuda.is_available()):
        requested_device = 'cpu'
    device = torch.device(requested_device)
    use_amp = device.type == 'cuda' and (not args.no_amp)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = build_model(cfg, device)
    model_state = checkpoint.get('model_state', checkpoint)
    model.load_state_dict(model_state, strict=False)
    model.eval()
    dataset = build_test_dataset(cfg)
    subject_to_slices = group_subject_slices(dataset)
    config_text = str(args.config)
    checkpoint_text = str(args.checkpoint)
    save_dir = str(cfg.get('training', {}).get('save_dir', ''))
    ckpt_parent = str(args.checkpoint.parent)
    checkpoint_name = args.checkpoint_name or args.checkpoint.stem
    metadata = {'config_path': str(args.config), 'checkpoint_path': str(args.checkpoint), 'checkpoint_name': checkpoint_name, 'target_domain': args.target_domain or infer_target_domain(config_text, checkpoint_text, save_dir, ckpt_parent), 'method': args.method or infer_method(config_text, checkpoint_text, save_dir, ckpt_parent), 'budget': args.budget if args.budget is not None else infer_budget(config_text, checkpoint_text, save_dir, ckpt_parent), 'train_seed': args.train_seed, 'search_seed': args.search_seed, 'subset_seed': args.subset_seed or infer_seed('(?:repeat|random)(\\d+)', config_text, checkpoint_text, save_dir), 'epoch': checkpoint.get('epoch', checkpoint.get('global_iter')), 'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'device': str(device), 'use_amp': use_amp, 'img_size': cfg['data']['img_size'], 'test_split': cfg['data']['test'], 'test_split_resolved': {**cfg['data']['test'], 'split_txt': cfg['data']['test'].get('resolved_split_txt', cfg['data']['test'].get('split_txt'))}, 'num_subjects': len(subject_to_slices), 'hd95_units': 'voxel', 'empty_mask_convention': {'dice': 'both empty=1; one empty=0', 'hd95': 'both empty=0; one empty=image diagonal penalty', 'precision': 'both empty=1; pred empty with nonempty GT=0', 'recall': 'both empty=1; GT empty with nonempty prediction=1'}}
    metrics_dir = args.metrics_dir or args.checkpoint.parent / 'metrics' / checkpoint_name
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = args.pred_dir or metrics_dir / 'predictions'
    if args.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    print(f'Evaluating {len(subject_to_slices)} subjects on {device}')
    for subject_id, slice_list in tqdm(subject_to_slices.items(), desc='Evaluating subjects', ncols=100):
        vol_pred, vol_gt = reconstruct_subject(subject_id=subject_id, slice_list=slice_list, dataset=dataset, model=model, device=device, use_amp=use_amp)
        rows.append(make_case_row(subject_id=subject_id, vol_pred=vol_pred, vol_gt=vol_gt, metadata=metadata, num_slices=len(slice_list)))
        if args.save_pred:
            np.savez_compressed(pred_dir / f'{subject_id}.npz', pred=vol_pred, gt=vol_gt)
        del vol_pred, vol_gt
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summary = summarize(rows)
    summary['metadata'] = metadata
    legacy = parse_legacy_results(args.legacy_results)
    attach_legacy_comparison(summary, legacy)
    write_csv(metrics_dir / 'per_case_metrics.csv', rows)
    save_json(metrics_dir / 'summary_metrics.json', summary)
    save_json(metrics_dir / 'run_metadata.json', metadata)
    dice = summary['metrics']['dice']
    hd95 = summary['metrics']['hd95']
    print('\n================== SUMMARY ==================')
    for region_name in list(REGIONS) + ['macro']:
        print(f"{region_name:5s} Dice={dice[region_name]['mean']:.4f} HD95={hd95[region_name]['mean']:.4f}")
    if 'legacy_dice_comparison' in summary:
        print('\nLegacy Dice comparison:')
        for region_name, comp in summary['legacy_dice_comparison'].items():
            print(f"{region_name:5s} old={comp['legacy']:.4f} new={comp['current']:.4f} diff={comp['abs_diff']:.6f}")
    print(f'\nSaved metrics to: {metrics_dir}')
if __name__ == '__main__':
    main()
