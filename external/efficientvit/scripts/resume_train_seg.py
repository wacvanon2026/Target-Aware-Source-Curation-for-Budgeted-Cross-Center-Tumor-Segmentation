#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.losses import DiceCELoss

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--target-epochs', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--weight-decay', type=float, default=None)
    parser.add_argument('--eta-min', type=float, default=None)
    parser.add_argument('--resume-optimizer', action='store_true', help='Load optimizer state from checkpoint. By default optimizer/scheduler are reset.')
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--delete-epoch-checkpoints', action='store_true', help='Delete per-epoch checkpoints after creating last.pt to save storage.')
    return parser.parse_args()

def load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r') as f:
        return yaml.safe_load(f)

def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(payload, f, sort_keys=False)

def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(payload, f, indent=2)

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

def resolve_config_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg['data']
    for dom in data_cfg.get('domains', []):
        split = dom.get('split', 'train')
        dom['split_txt'] = resolve_split_txt_dir(dom.get('split_txt'), split)
        if not split_file_exists(dom.get('split_txt'), split):
            raise FileNotFoundError(f"Missing split file for domain {dom.get('name')}: {dom.get('split_txt')}/{split}_subjects.txt")
    for key in ('val', 'test'):
        if key in data_cfg:
            split = data_cfg[key].get('split', key)
            data_cfg[key]['split_txt'] = resolve_split_txt_dir(data_cfg[key].get('split_txt'), split)
            if not split_file_exists(data_cfg[key].get('split_txt'), split):
                raise FileNotFoundError(f"Missing {key} split file: {data_cfg[key].get('split_txt')}/{split}_subjects.txt")
    return cfg

def load_datasets(data_cfg: dict[str, Any]) -> tuple[ConcatDataset, BraTSSliceDataset]:
    train_datasets = []
    skip_empty_train = data_cfg.get('skip_empty_train', True)
    skip_empty_val = data_cfg.get('skip_empty_val', False)
    for dom in data_cfg['domains']:
        ds = BraTSSliceDataset(root_dir=dom['path'], split=dom['split'], img_size=data_cfg['img_size'], split_txt_dir=dom.get('split_txt'), skip_empty=skip_empty_train)
        train_datasets.append(ds)
        print(f"Loaded train domain: {dom['name']} ({len(ds)} slices)")
    val_cfg = data_cfg['val']
    val_dataset = BraTSSliceDataset(root_dir=val_cfg['path'], split=val_cfg['split'], img_size=data_cfg['img_size'], split_txt_dir=val_cfg.get('split_txt'), skip_empty=skip_empty_val)
    return (ConcatDataset(train_datasets), val_dataset)

def make_loader(dataset, cfg: dict[str, Any], shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg['data']['batch_size'], shuffle=shuffle, generator=generator if shuffle else None, num_workers=cfg['data']['num_workers'], pin_memory=True)

def build_model(cfg: dict[str, Any], device: torch.device) -> EfficientViT_Seg:
    model_cfg = cfg['model']
    return EfficientViT_Seg(backbone=model_cfg['name'], in_channels=model_cfg['in_channels'], num_classes=model_cfg['num_classes'], pretrained=False).to(device)

def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Training', ncols=100):
        imgs = batch[0].to(device)
        masks = batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.item()) * imgs.size(0)
    return total_loss / max(len(loader.dataset), 1)

@torch.no_grad()
def evaluate_loss(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc='Validation', ncols=100):
        imgs = batch[0].to(device)
        masks = batch[1].to(device)
        with torch.amp.autocast('cuda', enabled=False):
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        total_loss += float(loss.item()) * imgs.size(0)
    return total_loss / max(len(loader.dataset), 1)

def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f'Saved checkpoint: {path}')

def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'lr'])
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    cfg = resolve_config_splits(load_yaml(args.config))
    cfg['training']['save_dir'] = str(args.output_dir)
    cfg['training']['epochs'] = int(args.target_epochs)
    if args.num_workers is not None:
        cfg['data']['num_workers'] = int(args.num_workers)
    set_seed(args.seed)
    device_name = args.device
    if device_name == 'cuda' and (not torch.cuda.is_available()):
        device_name = 'cpu'
    device = torch.device(device_name)
    ckpt = load_checkpoint(args.checkpoint, device)
    start_epoch = int(ckpt.get('epoch', 0))
    if args.target_epochs <= start_epoch:
        raise ValueError(f'target-epochs ({args.target_epochs}) must be > checkpoint epoch ({start_epoch})')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(args.output_dir / 'resume_config.yaml', cfg)
    model = build_model(cfg, device)
    model.load_state_dict(ckpt['model_state'], strict=False)
    train_dataset, val_dataset = load_datasets(cfg['data'])
    for ds in train_dataset.datasets:
        ds.return_meta = False
    train_loader = make_loader(train_dataset, cfg, shuffle=True, seed=args.seed)
    val_loader = make_loader(val_dataset, cfg, shuffle=False, seed=0)
    criterion = DiceCELoss()
    lr = args.lr if args.lr is not None else float(cfg['optimizer']['lr'])
    weight_decay = args.weight_decay if args.weight_decay is not None else float(cfg['optimizer']['weight_decay'])
    eta_min = args.eta_min if args.eta_min is not None else float(cfg['scheduler'].get('eta_min', 1e-06))
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_mode = 'reset'
    if args.resume_optimizer and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
        optimizer_mode = 'loaded_from_checkpoint'
    remaining_epochs = args.target_epochs - start_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(remaining_epochs, 1), eta_min=eta_min)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    metadata = {'config': str(args.config), 'resume_checkpoint': str(args.checkpoint), 'output_dir': str(args.output_dir), 'start_epoch': start_epoch, 'target_epochs': args.target_epochs, 'seed': args.seed, 'optimizer_mode': optimizer_mode, 'initial_lr': lr, 'effective_start_lr': optimizer.param_groups[0]['lr'], 'eta_min': eta_min, 'created_at_utc': datetime.now(timezone.utc).isoformat(), 'note': 'Default mode resets optimizer/scheduler for continued fine-tuning; use --resume-optimizer for conservative continuation.'}
    save_json(args.output_dir / 'resume_metadata.json', metadata)
    history: list[dict[str, Any]] = []
    best_val = float('inf')
    best_epoch = None
    print(f'Resuming from epoch {start_epoch} to {args.target_epochs} on {device}')
    print(f"Optimizer mode: {optimizer_mode}; lr={optimizer.param_groups[0]['lr']:.6g}; remaining_epochs={remaining_epochs}")
    for epoch in range(start_epoch + 1, args.target_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        lr_now = float(optimizer.param_groups[0]['lr'])
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'lr': lr_now}
        history.append(row)
        write_history(args.output_dir / 'resume_history.csv', history)
        print(f'Epoch {epoch}/{args.target_epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={lr_now:.3e}')
        state = {'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_loss': val_loss, 'config': cfg, 'resume_metadata': metadata}
        epoch_path = args.output_dir / f'epoch_{epoch:03d}.pt'
        save_checkpoint(epoch_path, state)
        save_checkpoint(args.output_dir / 'last.pt', state)
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            save_checkpoint(args.output_dir / 'best.pt', state)
        if args.delete_epoch_checkpoints and epoch_path.exists():
            epoch_path.unlink()
    metadata['best_val_loss'] = best_val
    metadata['best_epoch'] = best_epoch
    save_json(args.output_dir / 'resume_metadata.json', metadata)
    print(f'Done. Best val loss={best_val:.4f} at epoch {best_epoch}')
if __name__ == '__main__':
    main()
