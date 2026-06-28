#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import sys
from itertools import cycle
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.losses import DiceCELoss

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class GradientReversalFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (-ctx.alpha * grad_output, None)

class DomainDiscriminator(nn.Module):

    def __init__(self, in_dim: int, hidden_dim: int=256, dropout: float=0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor, alpha: float=1.0) -> torch.Tensor:
        return self.net(GradientReversalFn.apply(x, alpha))

class PixelDiscriminator(nn.Module):

    def __init__(self, in_channels: int=1, ndf: int=64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def dann_alpha(epoch: int, step: int, epochs: int, steps_per_epoch: int, lambda_max: float, schedule: str) -> float:
    if schedule == 'fixed':
        return float(lambda_max)
    total_steps = max(epochs * steps_per_epoch, 1)
    p = (epoch * steps_per_epoch + step) / total_steps
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, multipliers: list[float], fixed_sigma: float | None) -> torch.Tensor:
    z = torch.cat([x, y], dim=0).float()
    dist = torch.cdist(z, z, p=2).pow(2)
    if fixed_sigma is None:
        with torch.no_grad():
            sigma = dist.detach()
            sigma = sigma[sigma > 0].median().clamp_min(1e-06)
    else:
        sigma = torch.tensor(float(fixed_sigma), device=z.device).clamp_min(1e-06)
    return sum((torch.exp(-dist / (2.0 * sigma * float(m))) for m in multipliers))

def mmd_loss(x: torch.Tensor, y: torch.Tensor, multipliers: list[float], fixed_sigma: float | None) -> torch.Tensor:
    n = min(x.size(0), y.size(0))
    x = x[:n]
    y = y[:n]
    k = gaussian_kernel(x, y, multipliers, fixed_sigma)
    return k[:n, :n].mean() + k[n:, n:].mean() - 2.0 * k[:n, n:].mean()

def entropy_map(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    ent = -torch.sum(probs * torch.log(probs + 1e-06), dim=1, keepdim=True)
    return ent / math.log(float(logits.size(1)))

def bce_domain(logits: torch.Tensor, domain_value: float) -> torch.Tensor:
    labels = torch.full_like(logits, float(domain_value))
    return F.binary_cross_entropy_with_logits(logits, labels)

def forward_with_features(model: EfficientViT_Seg, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = x.shape[2:]
    feats = model.model.backbone(x)
    if isinstance(feats, dict):
        feats = list(feats.values())[-1]
    out = model.decoder(feats)
    out = model.final_head(out)
    if out.shape[2:] != (h, w):
        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
    pooled = F.adaptive_avg_pool2d(feats, 1).flatten(1)
    return (out, pooled)

def make_dataset(data_cfg: dict[str, Any], key: str, default_skip: bool) -> BraTSSliceDataset:
    item = data_cfg[key]
    ds = BraTSSliceDataset(root_dir=item['path'], split=item['split'], img_size=data_cfg['img_size'], split_txt_dir=item.get('split_txt'), skip_empty=bool(data_cfg.get(f'skip_empty_{key}', default_skip)))
    ds.return_meta = False
    print(f"Loaded {key}: {len(ds)} slices from {item.get('name', item['path'])}")
    return ds

def make_loader(dataset, cfg: dict[str, Any], shuffle: bool, drop_last: bool, generator: torch.Generator) -> DataLoader:
    return DataLoader(dataset, batch_size=int(cfg['data']['batch_size']), shuffle=shuffle, num_workers=int(cfg['data']['num_workers']), pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker, generator=generator)

def evaluate(model: EfficientViT_Seg, loader: DataLoader, criterion: nn.Module, device: torch.device, desc: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc=desc, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, masks)
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
    return {'loss': total_loss / max(total_samples, 1)}

def train_one_epoch(*, model: EfficientViT_Seg, feature_disc: DomainDiscriminator | None, output_disc: PixelDiscriminator | None, source_loader: DataLoader, target_loader: DataLoader, optimizer: optim.Optimizer, output_optimizer: optim.Optimizer | None, criterion: nn.Module, device: torch.device, scaler: torch.cuda.amp.GradScaler, cfg: dict[str, Any], epoch: int) -> dict[str, float]:
    model.train()
    if feature_disc is not None:
        feature_disc.train()
    if output_disc is not None:
        output_disc.train()
    da_cfg = cfg['da']
    method = str(da_cfg['method']).lower()
    epochs = int(cfg['training']['epochs'])
    steps_cfg = da_cfg.get('steps_per_epoch', 'auto')
    steps = max(len(source_loader), len(target_loader)) if steps_cfg == 'auto' else int(steps_cfg)
    lambda_max = float(da_cfg.get('lambda_max', 0.1))
    schedule = str(da_cfg.get('lambda_schedule', 'dann_logistic'))
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)
    totals = {'loss': 0.0, 'source_seg': 0.0, 'target_seg': 0.0, 'da_loss': 0.0, 'da_weight': 0.0, 'samples': 0}
    for step in tqdm(range(steps), desc=f'Training {method}', ncols=100):
        src_imgs, src_masks = next(source_iter)
        tgt_imgs, tgt_masks = next(target_iter)
        src_imgs = src_imgs.to(device, non_blocking=True)
        src_masks = src_masks.to(device, non_blocking=True)
        tgt_imgs = tgt_imgs.to(device, non_blocking=True)
        tgt_masks = tgt_masks.to(device, non_blocking=True)
        da_weight = dann_alpha(epoch, step, epochs, steps, lambda_max, schedule)
        optimizer.zero_grad(set_to_none=True)
        if output_optimizer is not None:
            output_optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            src_logits, src_feats = forward_with_features(model, src_imgs)
            tgt_logits, tgt_feats = forward_with_features(model, tgt_imgs)
            source_seg = criterion(src_logits, src_masks)
            target_seg = criterion(tgt_logits, tgt_masks)
            supervised = source_seg + float(da_cfg.get('target_seg_weight', 1.0)) * target_seg
            if method == 'dann':
                assert feature_disc is not None
                feats = torch.cat([src_feats, tgt_feats], dim=0)
                labels = torch.cat([torch.zeros(src_feats.size(0), device=device), torch.ones(tgt_feats.size(0), device=device)])
                da_loss = F.binary_cross_entropy_with_logits(feature_disc(feats, da_weight).view(-1), labels)
                loss = supervised + da_loss
            elif method in {'mmd', 'dan', 'dan_mmd'}:
                da_loss = mmd_loss(src_feats, tgt_feats, [float(x) for x in da_cfg.get('kernel_multipliers', [0.25, 0.5, 1.0, 2.0, 4.0])], None if da_cfg.get('fixed_sigma', 'auto') == 'auto' else float(da_cfg['fixed_sigma']))
                loss = supervised + da_weight * da_loss
            elif method in {'advent', 'advent_advent', 'se_asa', 'seasa'}:
                assert output_disc is not None and output_optimizer is not None
                src_ent = entropy_map(src_logits.detach())
                tgt_ent = entropy_map(tgt_logits.detach())
                disc_loss = 0.5 * (bce_domain(output_disc(src_ent), 0.0) + bce_domain(output_disc(tgt_ent), 1.0))
                scaler.scale(disc_loss).backward()
                scaler.step(output_optimizer)
                output_optimizer.zero_grad(set_to_none=True)
                tgt_ent_for_model = entropy_map(tgt_logits)
                adv_loss = bce_domain(output_disc(tgt_ent_for_model), 0.0)
                if method in {'se_asa', 'seasa'}:
                    entropy_regularizer = tgt_ent_for_model.mean()
                    consistency = F.mse_loss(F.softmax(tgt_logits, dim=1), F.softmax(tgt_logits + torch.randn_like(tgt_logits) * float(da_cfg.get('seasa_noise_std', 0.03)), dim=1))
                    adv_loss = adv_loss + float(da_cfg.get('seasa_lambda_class', 0.1)) * entropy_regularizer + float(da_cfg.get('seasa_lambda_selective', 0.01)) * consistency
                da_loss = adv_loss
                loss = supervised + da_weight * adv_loss
            else:
                raise ValueError(f'Unknown DA method: {method}')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        bs = src_imgs.size(0) + tgt_imgs.size(0)
        totals['loss'] += loss.item() * bs
        totals['source_seg'] += source_seg.item() * src_imgs.size(0)
        totals['target_seg'] += target_seg.item() * tgt_imgs.size(0)
        totals['da_loss'] += da_loss.item() * bs
        totals['da_weight'] += da_weight
        totals['samples'] += bs
    source_seen = steps * int(cfg['data']['batch_size'])
    target_seen = steps * int(cfg['data']['batch_size'])
    return {'loss': totals['loss'] / max(totals['samples'], 1), 'source_seg': totals['source_seg'] / max(source_seen, 1), 'target_seg': totals['target_seg'] / max(target_seen, 1), 'da_loss': totals['da_loss'] / max(totals['samples'], 1), 'da_weight': totals['da_weight'] / max(steps, 1)}

def save_checkpoint(state: dict[str, Any], save_dir: Path, filename: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / filename)
    print(f'Saved checkpoint: {filename}')

def main(config_path: str) -> None:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg['training']['seed']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('Config:', config_path)
    source_ds = make_dataset(cfg['data'], 'source', bool(cfg['data'].get('skip_empty_train', True)))
    target_ds = make_dataset(cfg['data'], 'target', bool(cfg['data'].get('skip_empty_align', True)))
    val_ds = make_dataset(cfg['data'], 'val', bool(cfg['data'].get('skip_empty_val', False)))
    test_ds = make_dataset(cfg['data'], 'test', bool(cfg['data'].get('skip_empty_val', False)))
    g = torch.Generator()
    g.manual_seed(int(cfg['training']['seed']))
    source_loader = make_loader(source_ds, cfg, shuffle=True, drop_last=True, generator=g)
    target_loader = make_loader(target_ds, cfg, shuffle=True, drop_last=True, generator=g)
    val_loader = make_loader(val_ds, cfg, shuffle=False, drop_last=False, generator=g)
    test_loader = make_loader(test_ds, cfg, shuffle=False, drop_last=False, generator=g)
    model_cfg = cfg['model']
    model = EfficientViT_Seg(backbone=model_cfg['name'], in_channels=int(model_cfg['in_channels']), num_classes=int(model_cfg['num_classes']), pretrained=bool(model_cfg.get('pretrained', True))).to(device)
    criterion = DiceCELoss()
    da_method = str(cfg['da']['method']).lower()
    with torch.no_grad():
        sample_imgs, _ = next(iter(source_loader))
        _, sample_feats = forward_with_features(model, sample_imgs[:1].to(device))
        feature_dim = sample_feats.size(1)
    feature_disc: DomainDiscriminator | None = None
    output_disc: PixelDiscriminator | None = None
    if da_method == 'dann':
        feature_disc = DomainDiscriminator(feature_dim, int(cfg['da'].get('domain_hidden_dim', 256)), float(cfg['da'].get('domain_dropout', 0.5))).to(device)
    if da_method in {'advent', 'advent_advent', 'se_asa', 'seasa'}:
        output_disc = PixelDiscriminator(1, int(cfg['da'].get('output_discriminator_ndf', 64))).to(device)
    params = list(model.parameters())
    if feature_disc is not None:
        params += list(feature_disc.parameters())
    optimizer = optim.AdamW(params, lr=float(cfg['optimizer']['lr']), weight_decay=float(cfg['optimizer']['weight_decay']))
    output_optimizer = None
    if output_disc is not None:
        output_optimizer = optim.Adam(output_disc.parameters(), lr=float(cfg['da'].get('lr_d', cfg['optimizer']['lr'])), betas=(float(cfg['da'].get('beta1_d', 0.9)), float(cfg['da'].get('beta2_d', 0.99))))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg['scheduler']['T_max']), eta_min=float(cfg['scheduler'].get('eta_min', 0.0)))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    save_dir = Path(cfg['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'run_metadata.json', 'w', encoding='utf-8') as f:
        json.dump({'config_path': os.path.abspath(config_path), 'method': da_method, 'seed': cfg['training']['seed']}, f, indent=2)
    best_loss = float('inf')
    history: list[dict[str, Any]] = []
    epochs = int(cfg['training']['epochs'])
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}/{epochs}')
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_stats = train_one_epoch(model=model, feature_disc=feature_disc, output_disc=output_disc, source_loader=source_loader, target_loader=target_loader, optimizer=optimizer, output_optimizer=output_optimizer, criterion=criterion, device=device, scaler=scaler, cfg=cfg, epoch=epoch)
        val_stats = evaluate(model, val_loader, criterion, device, 'Validating target_val')
        scheduler.step()
        print(f"Train loss {train_stats['loss']:.4f} | Source seg {train_stats['source_seg']:.4f} | Target seg {train_stats['target_seg']:.4f} | DA {train_stats['da_loss']:.4f} | DA weight {train_stats['da_weight']:.4f} | Val loss {val_stats['loss']:.4f}")
        checkpoint = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'feature_domain_state': feature_disc.state_dict() if feature_disc is not None else None, 'output_domain_state': output_disc.state_dict() if output_disc is not None else None, 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'train_stats': train_stats, 'val_stats': val_stats, 'val_loss': val_stats['loss'], 'config': cfg}
        if bool(cfg['training'].get('keep_epoch_checkpoints', True)):
            save_checkpoint(checkpoint, save_dir, f'epoch_{epoch + 1:03d}.pt')
        save_checkpoint(checkpoint, save_dir, 'last.pt')
        if val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            save_checkpoint(checkpoint, save_dir, 'best.pt')
        history.append({'epoch': epoch + 1, 'train': train_stats, 'val': val_stats})
        with open(save_dir / 'train_history.json', 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)
    recent = history[-5:]
    last_best_epoch = sorted(recent, key=lambda x: x['val']['loss'])[0]['epoch']
    if bool(cfg['training'].get('keep_epoch_checkpoints', True)) and (save_dir / f'epoch_{last_best_epoch:03d}.pt').exists():
        shutil.copy(save_dir / f'epoch_{last_best_epoch:03d}.pt', save_dir / 'best_last.pt')
    else:
        shutil.copy(save_dir / 'best.pt', save_dir / 'best_last.pt')
    print(f'Best-last epoch = {last_best_epoch}')
    if bool(cfg['training'].get('auto_eval', False)):
        final_results = {}
        for ckpt_name in ['best.pt', 'last.pt', 'best_last.pt']:
            print(f'\nEvaluating {ckpt_name} on target_test')
            ckpt = torch.load(save_dir / ckpt_name, map_location=device)
            model.load_state_dict(ckpt['model_state'])
            final_results[ckpt_name] = evaluate(model, test_loader, criterion, device, f'Testing {ckpt_name}')
        with open(save_dir / 'final_results.json', 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    main(args.config)
