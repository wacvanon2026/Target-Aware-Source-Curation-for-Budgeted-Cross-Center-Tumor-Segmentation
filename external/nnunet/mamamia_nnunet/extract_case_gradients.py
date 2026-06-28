#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import project_root
PROJECT_ROOT = project_root()
MAMAMIA_CLEAN = PROJECT_ROOT / 'mamamia_clean'
sys.path.insert(0, str(MAMAMIA_CLEAN))
from meta.fitness.nnunet_proxy import DCandCELoss, create_nnunet_2d_model

def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def create_countsketch(grad_dim: int, proj_dim: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    hash_indices = rng.randint(0, proj_dim, size=grad_dim).astype(np.int32)
    signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=grad_dim)
    return (hash_indices, signs)

def countsketch_project(grad_vec: np.ndarray, hash_indices: np.ndarray, signs: np.ndarray, proj_dim: int) -> np.ndarray:
    projected = np.zeros(proj_dim, dtype=np.float32)
    np.add.at(projected, hash_indices, grad_vec * signs)
    return projected

def load_model(checkpoint: Path, device: torch.device) -> nn.Module:
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = create_nnunet_2d_model(in_channels=3, num_classes=2)
    model.load_state_dict(ckpt['network_weights'])
    model.to(device)
    model.eval()
    return model

def gradient_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    named = list(model.named_parameters())
    for name, param in named:
        if any((key in name for key in ('seg_layers', 'decoder.stages.0', 'decoder.stages.1'))):
            params.append(param)
    if params:
        return params
    return [param for _, param in named[-max(4, len(named) // 5):]]

def find_case_dir(case_id: str, data_dirs: list[Path]) -> Path | None:
    for data_dir in data_dirs:
        if (data_dir / f'{case_id}.npy').exists() and (data_dir / f'{case_id}_seg.npy').exists():
            return data_dir
        if (data_dir / f'{case_id}.npz').exists():
            return data_dir
    return None

def load_case_arrays(case_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
    data_path = case_dir / f'{case_id}.npy'
    seg_path = case_dir / f'{case_id}_seg.npy'
    if data_path.exists() and seg_path.exists():
        return (np.load(data_path), np.load(seg_path))
    npz_path = case_dir / f'{case_id}.npz'
    if npz_path.exists():
        with np.load(npz_path) as arrays:
            return (arrays['data'], arrays['seg'])
    raise FileNotFoundError(f'No nnUNet preprocessed arrays found for {case_id} in {case_dir}')

def extract_case_gradient(model: nn.Module, criterion: nn.Module, params: list[nn.Parameter], case_dir: Path, case_id: str, device: torch.device, hash_indices: np.ndarray, signs: np.ndarray, proj_dim: int, crop_size: int, min_tumor_pixels: int, max_slices: int) -> np.ndarray | None:
    data, seg = load_case_arrays(case_dir, case_id)
    if seg.ndim == 4:
        seg = seg[0]
    tumor_slices = [z for z in range(seg.shape[0]) if int((seg[z] > 0).sum()) >= min_tumor_pixels]
    if not tumor_slices:
        return None
    if len(tumor_slices) > max_slices:
        rng = np.random.RandomState(abs(hash(case_id)) % 2 ** 31)
        tumor_slices = sorted(rng.choice(tumor_slices, max_slices, replace=False))
    projected = np.zeros(proj_dim, dtype=np.float32)
    for z in tumor_slices:
        image = torch.from_numpy(data[:, z]).float().unsqueeze(0)
        mask = torch.from_numpy(seg[z]).float().unsqueeze(0).unsqueeze(0)
        image = F.interpolate(image, size=(crop_size, crop_size), mode='bilinear', align_corners=False)
        mask = F.interpolate(mask, size=(crop_size, crop_size), mode='nearest')
        mask = (mask.squeeze(0).squeeze(0) > 0).long().unsqueeze(0)
        for channel in range(image.shape[1]):
            values = image[0, channel]
            vmin, vmax = (values.min(), values.max())
            if vmax > vmin:
                values = (values - vmin) / (vmax - vmin)
            image[0, channel] = (values - 0.485) / 0.229
        image = image.to(device)
        mask = mask.to(device)
        model.zero_grad(set_to_none=True)
        output = model(image)
        if output.shape[-2:] != mask.shape[-2:]:
            output = F.interpolate(output, size=mask.shape[-2:], mode='bilinear', align_corners=False)
        loss = criterion(output, mask)
        loss.backward()
        grad_parts = []
        for param in params:
            if param.grad is None:
                grad_parts.append(torch.zeros(param.numel()))
            else:
                grad_parts.append(param.grad.detach().cpu().flatten())
        grad_vec = torch.cat(grad_parts).numpy()
        projected += countsketch_project(grad_vec, hash_indices, signs, proj_dim)
    return projected / max(len(tumor_slices), 1)

def extract_set(label: str, case_ids: list[str], data_dirs: list[Path], model: nn.Module, criterion: nn.Module, params: list[nn.Parameter], device: torch.device, hash_indices: np.ndarray, signs: np.ndarray, args: argparse.Namespace) -> tuple[list[str], np.ndarray]:
    vectors: list[np.ndarray] = []
    valid: list[str] = []
    start = time.time()
    missing = 0
    no_tumor = 0
    for idx, case_id in enumerate(case_ids, start=1):
        case_dir = find_case_dir(case_id, data_dirs)
        if case_dir is None:
            missing += 1
            continue
        grad = extract_case_gradient(model, criterion, params, case_dir, case_id, device, hash_indices, signs, args.proj_dim, args.crop_size, args.min_tumor_pixels, args.max_slices)
        if grad is None:
            no_tumor += 1
            continue
        vectors.append(grad)
        valid.append(case_id)
        if idx % args.log_every == 0:
            elapsed = time.time() - start
            print(f'{label}: {idx}/{len(case_ids)} scanned, {len(valid)} valid, {elapsed / 60:.1f} min', flush=True)
    print(f'{label}: valid={len(valid)} missing={missing} no_tumor={no_tumor}', flush=True)
    return (valid, np.asarray(vectors, dtype=np.float32))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--dataset-dirs', type=Path, nargs='+', required=True)
    parser.add_argument('--pool-cases-file', type=Path, required=True)
    parser.add_argument('--query-cases-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--proj-dim', type=int, default=4096)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--min-tumor-pixels', type=int, default=50)
    parser.add_argument('--max-slices', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log-every', type=int, default=50)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dirs = [path.resolve() for path in args.dataset_dirs if path.exists()]
    if not data_dirs:
        raise SystemExit('No existing dataset dirs were provided')
    model = load_model(args.checkpoint, device)
    criterion = DCandCELoss()
    params = gradient_params(model)
    grad_dim = sum((param.numel() for param in params))
    hash_indices, signs = create_countsketch(grad_dim, args.proj_dim, args.seed + 1000)
    pool_cases = read_list(args.pool_cases_file)
    query_cases = read_list(args.query_cases_file)
    pool_ids, pool_gradients = extract_set('pool', pool_cases, data_dirs, model, criterion, params, device, hash_indices, signs, args)
    query_ids, query_gradients = extract_set('query', query_cases, data_dirs, model, criterion, params, device, hash_indices, signs, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'pool_gradients.npy', pool_gradients)
    np.save(args.output_dir / 'query_gradients.npy', query_gradients)
    (args.output_dir / 'pool_ids.json').write_text(json.dumps(pool_ids, indent=2) + '\n')
    (args.output_dir / 'query_ids.json').write_text(json.dumps(query_ids, indent=2) + '\n')
    (args.output_dir / 'info.json').write_text(json.dumps({'checkpoint': str(args.checkpoint), 'dataset_dirs': [str(path) for path in data_dirs], 'grad_dim_raw': grad_dim, 'proj_dim': args.proj_dim, 'projection': 'countsketch', 'n_pool': len(pool_ids), 'n_query': len(query_ids), 'max_slices': args.max_slices}, indent=2) + '\n')
    print(f'Saved gradients to {args.output_dir}', flush=True)
if __name__ == '__main__':
    main()
