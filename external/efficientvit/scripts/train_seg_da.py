#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_checkpoint(state: dict[str, Any], save_dir: str, filename: str) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    print(f'Saved checkpoint: {filename}')

def load_existing_history(save_dir: str, start_epoch: int) -> tuple[list[dict[str, float]], list[float]]:
    train_history: list[dict[str, float]] = []
    val_history: list[float] = []
    history_path = os.path.join(save_dir, 'train_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        train_history = list(history.get('train', []))[:start_epoch]
        val_history = [float(x) for x in history.get('val_loss', [])[:start_epoch]]
    for epoch in range(len(val_history) + 1, start_epoch + 1):
        ckpt_path = os.path.join(save_dir, f'epoch_{epoch:03d}.pt')
        if not os.path.exists(ckpt_path):
            break
        ckpt = torch.load(ckpt_path, map_location='cpu')
        val_history.append(float(ckpt['val_loss']))
        train_history.append(ckpt.get('train_stats', {}))
    return (train_history, val_history)

def restore_scheduler_position(scheduler: torch.optim.lr_scheduler.CosineAnnealingLR, optimizer: optim.Optimizer, checkpoint: dict[str, Any], start_epoch: int) -> None:
    if 'scheduler_state' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        return
    scheduler.last_epoch = start_epoch
    scheduler._step_count = start_epoch + 1
    scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]

class GradientReversalFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (grad_output.neg() * ctx.alpha, None)

class DomainDiscriminator(nn.Module):

    def __init__(self, in_channels: int, hidden_dim: int=256, dropout: float=0.5) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(in_channels, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, feats: torch.Tensor, alpha: float=1.0) -> torch.Tensor:
        x = GradientReversalFn.apply(feats, alpha)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

class OutputEntropyDiscriminator(nn.Module):

    def __init__(self, num_classes: int, ndf: int=64) -> None:
        super().__init__()
        self.model = nn.Sequential(nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def dann_lambda(epoch: int, step: int, epochs: int, steps_per_epoch: int, lambda_max: float) -> float:
    total_steps = max(epochs * steps_per_epoch, 1)
    current = epoch * steps_per_epoch + step
    p = current / total_steps
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

def set_requires_grad(module: nn.Module | None, requires_grad: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = requires_grad

def bce_loss_with_label(logits: torch.Tensor, label: float) -> torch.Tensor:
    target = torch.empty_like(logits).fill_(float(label))
    return F.binary_cross_entropy_with_logits(logits, target)

def prob_2_entropy(prob: torch.Tensor) -> torch.Tensor:
    num_classes = prob.size(1)
    return -prob * torch.log2(prob + 1e-30) / math.log2(num_classes)

def normalized_entropy_map(logits: torch.Tensor) -> torch.Tensor:
    prob = F.softmax(logits, dim=1)
    return -torch.sum(prob * torch.log(prob + 1e-30), dim=1) / math.log(prob.size(1))

def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    return normalized_entropy_map(logits).mean()

def extract_backbone_features(model: EfficientViT_Seg, x: torch.Tensor) -> torch.Tensor:
    feats = model.model.backbone(x)
    if isinstance(feats, dict):
        feats = list(feats.values())[-1]
    return feats

def decode_features(model: EfficientViT_Seg, feats: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    out = model.decoder(feats)
    out = model.final_head(out)
    if out.shape[2:] != hw:
        out = F.interpolate(out, size=hw, mode='bilinear', align_corners=False)
    return out

def pooled_features(feats: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(feats, 1).flatten(1).float()

def fourier_augment_target(source: torch.Tensor, target: torch.Tensor, beta: float=0.01) -> torch.Tensor:
    if beta <= 0:
        return target
    with torch.cuda.amp.autocast(enabled=False):
        source = source.float()
        target = target.float()
        idx = torch.randint(0, source.size(0), (target.size(0),), device=target.device)
        source = source[idx]
        src_fft = torch.fft.fft2(source, dim=(-2, -1))
        tgt_fft = torch.fft.fft2(target, dim=(-2, -1))
        src_amp = torch.abs(src_fft)
        tgt_amp = torch.abs(tgt_fft)
        tgt_phase = torch.angle(tgt_fft)
        src_amp = torch.fft.fftshift(src_amp, dim=(-2, -1))
        tgt_amp = torch.fft.fftshift(tgt_amp, dim=(-2, -1))
        _, _, h, w = target.shape
        radius = max(1, int(min(h, w) * beta))
        h0, h1 = (h // 2 - radius, h // 2 + radius)
        w0, w1 = (w // 2 - radius, w // 2 + radius)
        tgt_amp[:, :, h0:h1, w0:w1] = src_amp[:, :, h0:h1, w0:w1]
        mixed_amp = torch.fft.ifftshift(tgt_amp, dim=(-2, -1))
        mixed_fft = mixed_amp * torch.exp(1j * tgt_phase)
        mixed = torch.fft.ifft2(mixed_fft, dim=(-2, -1)).real
        return mixed.clamp(0.0, 1.0).to(dtype=target.dtype)

def intensity_augment_target(target: torch.Tensor, noise_std: float=0.03) -> torch.Tensor:
    scale = torch.empty(target.size(0), 1, 1, 1, device=target.device).uniform_(0.9, 1.1)
    shift = torch.empty(target.size(0), 1, 1, 1, device=target.device).uniform_(-0.05, 0.05)
    noise = torch.randn_like(target) * noise_std
    return (target * scale + shift + noise).clamp(0.0, 1.0)

def downsample_labels(labels: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(labels.unsqueeze(1).float(), size=size, mode='nearest').squeeze(1).long()

def batch_class_centers(feats: torch.Tensor, labels: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor]:
    labels_ds = downsample_labels(labels, feats.shape[2:])
    centers = []
    valid = []
    for cls_idx in range(num_classes):
        mask = (labels_ds == cls_idx).unsqueeze(1)
        count = mask.sum().clamp_min(1)
        center = (feats * mask).sum(dim=(0, 2, 3)) / count
        centers.append(center)
        valid.append((mask.sum() > 0).to(feats.device))
    return (torch.stack(centers, dim=0), torch.stack(valid).bool())

def init_seasa_state(num_classes: int, channels: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {'src_centers': torch.zeros(num_classes, channels, device=device), 'tgt_centers': torch.zeros(num_classes, channels, device=device), 'src_valid': torch.zeros(num_classes, dtype=torch.bool, device=device), 'tgt_valid': torch.zeros(num_classes, dtype=torch.bool, device=device)}

def update_center_memory(memory: torch.Tensor, memory_valid: torch.Tensor, batch_centers: torch.Tensor, batch_valid: torch.Tensor, momentum: float) -> None:
    with torch.no_grad():
        for cls_idx in range(memory.size(0)):
            if not bool(batch_valid[cls_idx]):
                continue
            if bool(memory_valid[cls_idx]):
                memory[cls_idx].mul_(momentum).add_(batch_centers[cls_idx].detach(), alpha=1.0 - momentum)
            else:
                memory[cls_idx].copy_(batch_centers[cls_idx].detach())
                memory_valid[cls_idx] = True

def semantic_alignment_loss(state: dict[str, torch.Tensor], src_feats: torch.Tensor, src_masks: torch.Tensor, tgt_feats: torch.Tensor, tgt_logits: torch.Tensor, num_classes: int, momentum: float) -> torch.Tensor:
    src_centers, src_valid = batch_class_centers(src_feats, src_masks, num_classes)
    pseudo_target = F.interpolate(tgt_logits.detach().argmax(dim=1, keepdim=True).float(), size=tgt_feats.shape[2:], mode='nearest').squeeze(1).long()
    tgt_centers, tgt_valid = batch_class_centers(tgt_feats, pseudo_target, num_classes)
    update_center_memory(state['src_centers'], state['src_valid'], src_centers, src_valid, momentum)
    update_center_memory(state['tgt_centers'], state['tgt_valid'], tgt_centers, tgt_valid, momentum)
    valid = src_valid & tgt_valid
    if not bool(valid.any()):
        return torch.zeros((), device=src_feats.device)
    return F.mse_loss(src_centers[valid], tgt_centers[valid])

def selective_entropy_loss(model: EfficientViT_Seg, source_imgs: torch.Tensor, target_imgs: torch.Tensor, target_logits: torch.Tensor, da_cfg: dict[str, Any]) -> tuple[torch.Tensor, float, float]:
    original_preds = target_logits.detach().argmax(dim=1)
    aug_logits: list[torch.Tensor] = []
    consistent_count = torch.zeros_like(original_preds, dtype=torch.int16)
    inconsistent_count = torch.zeros_like(original_preds, dtype=torch.int16)
    num_aug = int(da_cfg.get('seasa_num_aug', 3))
    fourier_beta = float(da_cfg.get('seasa_fourier_beta', 0.01))
    for aug_idx in range(num_aug):
        if aug_idx % 2 == 0:
            aug_imgs = fourier_augment_target(source_imgs, target_imgs, beta=fourier_beta)
        else:
            aug_imgs = intensity_augment_target(target_imgs, noise_std=float(da_cfg.get('seasa_noise_std', 0.03)))
        logits_aug = model(aug_imgs)
        aug_logits.append(logits_aug)
        aug_preds = logits_aug.detach().argmax(dim=1)
        consistent = aug_preds == original_preds
        consistent_count += consistent.to(torch.int16)
        inconsistent_count += (~consistent).to(torch.int16)
    threshold = int(da_cfg.get('seasa_consistency_threshold', max(1, (num_aug + 1) // 2)))
    correct_mask = consistent_count >= threshold
    incorrect_mask = inconsistent_count >= threshold
    correct_ratio = float(correct_mask.float().mean().item())
    incorrect_ratio = float(incorrect_mask.float().mean().item())
    losses = []
    for logits_aug in aug_logits:
        ent = normalized_entropy_map(logits_aug)
        if bool(correct_mask.any()):
            losses.append(correct_ratio * ent[correct_mask].mean())
        if bool(incorrect_mask.any()):
            losses.append(-incorrect_ratio * ent[incorrect_mask].mean())
    if not losses:
        return (torch.zeros((), device=target_imgs.device), correct_ratio, incorrect_ratio)
    return (torch.stack(losses).mean(), correct_ratio, incorrect_ratio)

def pairwise_sq_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True).t()
    return (x_norm + y_norm - 2.0 * x @ y.t()).clamp_min(0.0)

def multi_kernel_mmd(source: torch.Tensor, target: torch.Tensor, kernel_multipliers: list[float], fixed_sigma: float | None=None) -> torch.Tensor:
    source = source.float()
    target = target.float()
    combined = torch.cat([source, target], dim=0)
    dists = pairwise_sq_dists(combined, combined)
    if fixed_sigma is None or fixed_sigma <= 0:
        with torch.no_grad():
            positive = dists.detach()[dists.detach() > 0]
            if positive.numel() == 0:
                base_sigma = torch.tensor(1.0, device=combined.device)
            else:
                base_sigma = torch.median(positive)
    else:
        base_sigma = torch.tensor(float(fixed_sigma), device=combined.device)
    kernels = 0.0
    for mult in kernel_multipliers:
        sigma = (base_sigma * float(mult)).clamp_min(1e-06)
        kernels = kernels + torch.exp(-dists / (2.0 * sigma))
    n_source = source.size(0)
    k_xx = kernels[:n_source, :n_source]
    k_yy = kernels[n_source:, n_source:]
    k_xy = kernels[:n_source, n_source:]
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

def infinite_iter(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

def build_dataset(data_cfg: dict[str, Any], entry: dict[str, Any], split: str, skip_empty: bool) -> BraTSSliceDataset:
    return BraTSSliceDataset(root_dir=entry['path'], split=split, img_size=data_cfg['img_size'], split_txt_dir=entry.get('split_txt'), skip_empty=skip_empty)

def build_loader(dataset: BraTSSliceDataset, batch_size: int, num_workers: int, shuffle: bool, seed: int | None=None) -> DataLoader:
    dataset.return_meta = False
    generator = None
    if seed is not None and shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, worker_init_fn=seed_worker if seed is not None and shuffle else None, generator=generator)

def evaluate_loss(model: EfficientViT_Seg, loader: DataLoader | None, criterion: nn.Module, device: torch.device) -> float:
    if loader is None:
        return 0.0
    model.eval()
    total = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='Validating', ncols=100):
            imgs = batch[0].to(device)
            masks = batch[1].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total += float(loss.item()) * imgs.size(0)
    return total / max(len(loader.dataset), 1)

def infer_backbone_channels(model: EfficientViT_Seg) -> int:
    last_block = list(model.model.backbone.stages[-1].op_list)[-1]
    return int(last_block.context_module.main.proj.conv.out_channels)

def train_one_epoch_da(model: EfficientViT_Seg, discriminator: DomainDiscriminator | None, source_loader: DataLoader, target_loader: DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, device: torch.device, scaler: torch.amp.GradScaler, da_cfg: dict[str, Any], epoch: int, epochs: int, steps_per_epoch: int) -> dict[str, float]:
    model.train()
    if discriminator is not None:
        discriminator.train()
    method = da_cfg['method'].lower()
    lambda_max = float(da_cfg.get('lambda_max', 0.1))
    target_seg_weight = float(da_cfg.get('target_seg_weight', 1.0))
    src_iter = infinite_iter(source_loader)
    tgt_iter = infinite_iter(target_loader)
    totals = {'loss': 0.0, 'seg_source': 0.0, 'seg_target': 0.0, 'da': 0.0}
    domain_criterion = nn.BCEWithLogitsLoss()
    kernel_multipliers = [float(x) for x in da_cfg.get('kernel_multipliers', [0.25, 0.5, 1.0, 2.0, 4.0])]
    fixed_sigma = da_cfg.get('fixed_sigma')
    fixed_sigma = None if fixed_sigma in (None, 'auto') else float(fixed_sigma)
    pbar = tqdm(range(steps_per_epoch), desc=f'{method.upper()} epoch {epoch + 1}/{epochs}', ncols=120)
    for step in pbar:
        src_batch = next(src_iter)
        tgt_batch = next(tgt_iter)
        src_imgs = src_batch[0].to(device)
        src_masks = src_batch[1].to(device)
        tgt_imgs = tgt_batch[0].to(device)
        tgt_masks = tgt_batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        current_lambda = dann_lambda(epoch, step, epochs, steps_per_epoch, lambda_max)
        with torch.cuda.amp.autocast():
            src_feats = extract_backbone_features(model, src_imgs)
            tgt_feats = extract_backbone_features(model, tgt_imgs)
            src_logits = decode_features(model, src_feats, src_imgs.shape[2:])
            tgt_logits = decode_features(model, tgt_feats, tgt_imgs.shape[2:])
            seg_source = criterion(src_logits, src_masks)
            seg_target = criterion(tgt_logits, tgt_masks)
            seg_loss = seg_source + target_seg_weight * seg_target
            if method == 'dann':
                if discriminator is None:
                    raise RuntimeError('DANN requires a domain discriminator')
                src_domain = discriminator(src_feats, current_lambda)
                tgt_domain = discriminator(tgt_feats, current_lambda)
                src_labels = torch.ones(src_domain.shape, device=device)
                tgt_labels = torch.zeros(tgt_domain.shape, device=device)
                da_loss = domain_criterion(src_domain, src_labels) + domain_criterion(tgt_domain, tgt_labels)
                loss = seg_loss + da_loss
            elif method in {'mmd', 'dan_mmd'}:
                da_loss = multi_kernel_mmd(pooled_features(src_feats), pooled_features(tgt_feats), kernel_multipliers=kernel_multipliers, fixed_sigma=fixed_sigma)
                loss = seg_loss + current_lambda * da_loss
            else:
                raise ValueError(f'Unsupported DA method: {method}')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        totals['loss'] += float(loss.item())
        totals['seg_source'] += float(seg_source.item())
        totals['seg_target'] += float(seg_target.item())
        totals['da'] += float(da_loss.item())
        pbar.set_postfix(loss=f'{loss.item():.3f}', src=f'{seg_source.item():.3f}', tgt=f'{seg_target.item():.3f}', da=f'{da_loss.item():.3f}', lam=f'{current_lambda:.3f}')
    return {key: val / max(steps_per_epoch, 1) for key, val in totals.items()}

def train_one_epoch_output_da(model: EfficientViT_Seg, output_discriminator: OutputEntropyDiscriminator, source_loader: DataLoader, target_loader: DataLoader, optimizer: optim.Optimizer, optimizer_d: optim.Optimizer, criterion: nn.Module, device: torch.device, scaler: torch.amp.GradScaler, da_cfg: dict[str, Any], epoch: int, epochs: int, steps_per_epoch: int, seasa_state: dict[str, torch.Tensor] | None=None) -> dict[str, float]:
    model.train()
    output_discriminator.train()
    method = da_cfg['method'].lower()
    lambda_max = float(da_cfg.get('lambda_max', 0.001))
    target_seg_weight = float(da_cfg.get('target_seg_weight', 1.0))
    num_classes = int(da_cfg.get('num_classes', 4))
    src_iter = infinite_iter(source_loader)
    tgt_iter = infinite_iter(target_loader)
    totals = {'loss': 0.0, 'seg_source': 0.0, 'seg_target': 0.0, 'da': 0.0, 'disc': 0.0, 'seasa_selective': 0.0, 'seasa_class': 0.0, 'seasa_correct_ratio': 0.0, 'seasa_incorrect_ratio': 0.0}
    pbar = tqdm(range(steps_per_epoch), desc=f'{method.upper()} epoch {epoch + 1}/{epochs}', ncols=140)
    for step in pbar:
        src_batch = next(src_iter)
        tgt_batch = next(tgt_iter)
        src_imgs = src_batch[0].to(device)
        src_masks = src_batch[1].to(device)
        tgt_imgs = tgt_batch[0].to(device)
        tgt_masks = tgt_batch[1].to(device)
        if da_cfg.get('lambda_schedule', 'fixed') == 'fixed':
            current_lambda = lambda_max
        else:
            current_lambda = dann_lambda(epoch, step, epochs, steps_per_epoch, lambda_max)
        optimizer.zero_grad(set_to_none=True)
        optimizer_d.zero_grad(set_to_none=True)
        set_requires_grad(output_discriminator, False)
        with torch.cuda.amp.autocast():
            if method in {'se_asa', 'seasa'}:
                src_feats = extract_backbone_features(model, src_imgs)
                tgt_feats = extract_backbone_features(model, tgt_imgs)
                src_logits = decode_features(model, src_feats, src_imgs.shape[2:])
                tgt_logits = decode_features(model, tgt_feats, tgt_imgs.shape[2:])
            else:
                src_feats = tgt_feats = None
                src_logits = model(src_imgs)
                tgt_logits = model(tgt_imgs)
            seg_source = criterion(src_logits, src_masks)
            seg_target = criterion(tgt_logits, tgt_masks)
            seg_loss = seg_source + target_seg_weight * seg_target
            tgt_entropy = prob_2_entropy(F.softmax(tgt_logits, dim=1))
            adv_loss = bce_loss_with_label(output_discriminator(tgt_entropy), 0.0)
            loss = seg_loss + current_lambda * adv_loss
            selective_loss = torch.zeros((), device=device)
            class_loss = torch.zeros((), device=device)
            correct_ratio = incorrect_ratio = 0.0
            if method in {'se_asa', 'seasa'}:
                if seasa_state is None:
                    raise RuntimeError('SE-ASA requires a state dictionary')
                class_loss = semantic_alignment_loss(seasa_state, src_feats, src_masks, tgt_feats, tgt_logits, num_classes=num_classes, momentum=float(da_cfg.get('seasa_class_center_momentum', 0.01)))
                selective_loss, correct_ratio, incorrect_ratio = selective_entropy_loss(model, src_imgs, tgt_imgs, tgt_logits, da_cfg)
                loss = loss + float(da_cfg.get('seasa_lambda_class', 0.1)) * class_loss + float(da_cfg.get('seasa_lambda_selective', 0.01)) * selective_loss
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        set_requires_grad(output_discriminator, True)
        optimizer_d.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            src_entropy_detached = prob_2_entropy(F.softmax(src_logits.detach(), dim=1))
            tgt_entropy_detached = prob_2_entropy(F.softmax(tgt_logits.detach(), dim=1))
            d_src = output_discriminator(src_entropy_detached)
            d_tgt = output_discriminator(tgt_entropy_detached)
            disc_loss = 0.5 * (bce_loss_with_label(d_src, 0.0) + bce_loss_with_label(d_tgt, 1.0))
        scaler.scale(disc_loss).backward()
        scaler.step(optimizer_d)
        scaler.update()
        totals['loss'] += float(loss.item())
        totals['seg_source'] += float(seg_source.item())
        totals['seg_target'] += float(seg_target.item())
        totals['da'] += float(adv_loss.item())
        totals['disc'] += float(disc_loss.item())
        totals['seasa_selective'] += float(selective_loss.item())
        totals['seasa_class'] += float(class_loss.item())
        totals['seasa_correct_ratio'] += correct_ratio
        totals['seasa_incorrect_ratio'] += incorrect_ratio
        pbar.set_postfix(loss=f'{loss.item():.3f}', src=f'{seg_source.item():.3f}', tgt=f'{seg_target.item():.3f}', adv=f'{adv_loss.item():.3f}', d=f'{disc_loss.item():.3f}', lam=f'{current_lambda:.4f}')
    return {key: val / max(steps_per_epoch, 1) for key, val in totals.items()}

def run_auto_eval(config_path: str, save_dir: str) -> None:
    print('\nStarting legacy 3D Dice evaluation for DA checkpoints...')
    from eval.evaluate_2d_to_3d import run_evaluation_2d_to_3d
    for name in ['best', 'last', 'best_last']:
        ckpt = os.path.join(save_dir, f'{name}.pt')
        if not os.path.exists(ckpt):
            continue
        print(f'\nEvaluating {name.upper()} checkpoint...')
        run_evaluation_2d_to_3d(config_path=config_path, checkpoint_path=ckpt, output_dir=save_dir)

def main(config_path: str, seed: int | None=None, no_auto_eval: bool=False, resume_checkpoint: str | None=None) -> None:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if seed is None:
        seed = cfg.get('training', {}).get('seed')
    if seed is not None:
        seed = int(seed)
        set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model_cfg = cfg['model']
    model = EfficientViT_Seg(backbone=model_cfg['name'], in_channels=model_cfg['in_channels'], num_classes=model_cfg['num_classes'], pretrained=model_cfg.get('pretrained', True)).to(device)
    data_cfg = cfg['data']
    batch_size = int(data_cfg['batch_size'])
    num_workers = int(data_cfg['num_workers'])
    skip_empty_train = bool(data_cfg.get('skip_empty_train', True))
    skip_empty_val = bool(data_cfg.get('skip_empty_val', False))
    skip_empty_align = bool(data_cfg.get('skip_empty_align', skip_empty_train))
    print('Loading DA datasets...')
    source_ds = build_dataset(data_cfg, data_cfg['source'], split=data_cfg['source'].get('split', 'train'), skip_empty=skip_empty_train)
    target_ds = build_dataset(data_cfg, data_cfg['target'], split=data_cfg['target'].get('split', 'train'), skip_empty=skip_empty_train)
    target_align_entry = data_cfg.get('target_align', data_cfg['target'])
    target_align_ds = build_dataset(data_cfg, target_align_entry, split=target_align_entry.get('split', 'train'), skip_empty=skip_empty_align)
    val_loader = None
    if 'val' in data_cfg:
        val_ds = build_dataset(data_cfg, data_cfg['val'], split=data_cfg['val'].get('split', 'val'), skip_empty=skip_empty_val)
        val_loader = build_loader(val_ds, batch_size, num_workers, shuffle=False)
    source_loader = build_loader(source_ds, batch_size, num_workers, shuffle=True, seed=seed)
    target_loader = build_loader(target_ds, batch_size, num_workers, shuffle=True, seed=None if seed is None else seed + 1000)
    target_align_loader = build_loader(target_align_ds, batch_size, num_workers, shuffle=True, seed=None if seed is None else seed + 2000)
    da_cfg = cfg['da']
    method = da_cfg['method'].lower()
    discriminator = None
    output_discriminator = None
    optimizer_d = None
    seasa_state = None
    params: list[nn.Parameter] = list(model.parameters())
    if method == 'dann':
        discriminator = DomainDiscriminator(infer_backbone_channels(model), hidden_dim=int(da_cfg.get('domain_hidden_dim', 256)), dropout=float(da_cfg.get('domain_dropout', 0.5))).to(device)
        params += list(discriminator.parameters())
    elif method not in {'mmd', 'dan_mmd'}:
        if method not in {'advent_advent', 'advent', 'advent_advent_output', 'se_asa', 'seasa'}:
            raise ValueError(f'Unsupported DA method: {method}')
        output_discriminator = OutputEntropyDiscriminator(num_classes=int(model_cfg['num_classes']), ndf=int(da_cfg.get('output_discriminator_ndf', 64))).to(device)
        if method in {'se_asa', 'seasa'}:
            seasa_state = init_seasa_state(num_classes=int(model_cfg['num_classes']), channels=infer_backbone_channels(model), device=device)
    optimizer_cfg = cfg['optimizer']
    optimizer = optim.AdamW(params, lr=float(optimizer_cfg['lr']), weight_decay=float(optimizer_cfg['weight_decay']))
    if output_discriminator is not None:
        optimizer_d = optim.Adam(output_discriminator.parameters(), lr=float(da_cfg.get('lr_d', optimizer_cfg['lr'])), betas=(float(da_cfg.get('beta1_d', 0.9)), float(da_cfg.get('beta2_d', 0.99))))
    train_cfg = cfg['training']
    epochs = int(train_cfg['epochs'])
    steps_per_epoch_cfg = da_cfg.get('steps_per_epoch', 'auto')
    if steps_per_epoch_cfg == 'auto':
        steps_per_epoch = max(len(source_loader), len(target_align_loader))
    else:
        steps_per_epoch = int(steps_per_epoch_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg['scheduler'].get('T_max', epochs)), eta_min=float(cfg['scheduler'].get('eta_min', 1e-06)))
    criterion = DiceCELoss()
    scaler = torch.amp.GradScaler('cuda')
    save_dir = train_cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    metadata = {'config_path': os.path.abspath(config_path), 'save_dir': os.path.abspath(save_dir), 'seed': seed, 'created_at': datetime.now().isoformat(timespec='seconds'), 'epochs': epochs, 'steps_per_epoch': steps_per_epoch, 'da': da_cfg, 'lr': float(optimizer_cfg['lr']), 'weight_decay': float(optimizer_cfg['weight_decay']), 'source_slices': len(source_ds), 'target_slices': len(target_ds), 'target_align_slices': len(target_align_ds), 'adaptation_notes': {'advent_advent': 'ADVENT AdvEnt output entropy adversarial loss adapted to EfficientViT single-output BraTS segmentation.', 'se_asa': 'SE-ASA core losses adapted to EfficientViT/BraTS: output entropy adversarial loss, prediction-consistency selective entropy, and online class-center semantic alignment.'}.get(method)}
    if resume_checkpoint is None:
        resume_checkpoint = train_cfg.get('resume_checkpoint')
    if resume_checkpoint:
        resume_checkpoint = os.path.abspath(resume_checkpoint)
    metadata['resume_checkpoint'] = resume_checkpoint
    with open(os.path.join(save_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f'Starting DA training: method={method}, epochs={epochs}, steps_per_epoch={steps_per_epoch}')
    start_epoch = 0
    best_loss = float('inf')
    val_history: list[float] = []
    train_history: list[dict[str, float]] = []
    if resume_checkpoint:
        print(f'Resuming from checkpoint: {resume_checkpoint}')
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        if discriminator is not None:
            if 'domain_discriminator_state' not in checkpoint:
                raise RuntimeError('Resume checkpoint is missing domain_discriminator_state')
            discriminator.load_state_dict(checkpoint['domain_discriminator_state'])
        if output_discriminator is not None:
            if 'output_discriminator_state' not in checkpoint:
                raise RuntimeError('Resume checkpoint is missing output_discriminator_state')
            output_discriminator.load_state_dict(checkpoint['output_discriminator_state'])
            if optimizer_d is not None and 'optimizer_d_state' in checkpoint:
                optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
        if seasa_state is not None and 'seasa_state' in checkpoint:
            for key, value in checkpoint['seasa_state'].items():
                seasa_state[key] = value.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = int(checkpoint['epoch'])
        restore_scheduler_position(scheduler, optimizer, checkpoint, start_epoch)
        train_history, val_history = load_existing_history(save_dir, start_epoch)
        if len(val_history) < start_epoch:
            val_history.extend([float('inf')] * (start_epoch - len(val_history)))
        best_loss = min(val_history) if val_history else float(checkpoint.get('val_loss', 'inf'))
        print(f"Resume state: completed_epochs={start_epoch}, next_epoch={start_epoch + 1}, best_val={best_loss:.4f}, lr={[group['lr'] for group in optimizer.param_groups]}")
        if start_epoch >= epochs:
            print(f'Checkpoint already reached requested epochs={epochs}; skipping training loop.')
    for epoch in range(start_epoch, epochs):
        if output_discriminator is None:
            stats = train_one_epoch_da(model=model, discriminator=discriminator, source_loader=source_loader, target_loader=target_align_loader, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler, da_cfg=da_cfg, epoch=epoch, epochs=epochs, steps_per_epoch=steps_per_epoch)
        else:
            if optimizer_d is None:
                raise RuntimeError('Output-level DA requires discriminator optimizer')
            stats = train_one_epoch_output_da(model=model, output_discriminator=output_discriminator, source_loader=source_loader, target_loader=target_align_loader, optimizer=optimizer, optimizer_d=optimizer_d, criterion=criterion, device=device, scaler=scaler, da_cfg=da_cfg, epoch=epoch, epochs=epochs, steps_per_epoch=steps_per_epoch, seasa_state=seasa_state)
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_history.append(val_loss)
        train_history.append(stats)
        scheduler.step()
        print(f"Epoch {epoch + 1}/{epochs}: loss={stats['loss']:.4f}, src={stats['seg_source']:.4f}, tgt={stats['seg_target']:.4f}, da={stats['da']:.4f}, val={val_loss:.4f}")
        checkpoint = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(), 'val_loss': val_loss, 'train_stats': stats, 'config': cfg, 'seed': seed}
        if discriminator is not None:
            checkpoint['domain_discriminator_state'] = discriminator.state_dict()
        if output_discriminator is not None:
            checkpoint['output_discriminator_state'] = output_discriminator.state_dict()
        if optimizer_d is not None:
            checkpoint['optimizer_d_state'] = optimizer_d.state_dict()
        if seasa_state is not None:
            checkpoint['seasa_state'] = {key: value.detach().cpu() for key, value in seasa_state.items()}
        save_checkpoint(checkpoint, save_dir, f'epoch_{epoch + 1:03d}.pt')
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpoint, save_dir, 'best.pt')
        with open(os.path.join(save_dir, 'train_history.json'), 'w') as f:
            json.dump({'train': train_history, 'val_loss': val_history}, f, indent=2)
    save_checkpoint(checkpoint, save_dir, 'last.pt')
    if len(val_history) >= 5:
        last_k = 5
        last_losses = val_history[-last_k:]
        min_idx = last_losses.index(min(last_losses))
        best_last_epoch = len(val_history) - last_k + min_idx + 1
        best_last_ckpt = os.path.join(save_dir, f'epoch_{best_last_epoch:03d}.pt')
        if os.path.exists(best_last_ckpt):
            torch.save(torch.load(best_last_ckpt, map_location='cpu'), os.path.join(save_dir, 'best_last.pt'))
            print(f'Saved best_last.pt (epoch {best_last_epoch})')
    if not bool(train_cfg.get('keep_epoch_checkpoints', False)):
        removed = 0
        for path in Path(save_dir).glob('epoch_*.pt'):
            path.unlink()
            removed += 1
        print(f'Removed {removed} epoch checkpoints; kept best.pt, last.pt, best_last.pt')
    with open(os.path.join(save_dir, 'train_history.json'), 'w') as f:
        json.dump({'train': train_history, 'val_loss': val_history}, f, indent=2)
    print(f'Training complete. Best Val Loss: {best_loss:.4f}')
    auto_eval = bool(train_cfg.get('auto_eval', True)) and (not no_auto_eval)
    if auto_eval:
        run_auto_eval(config_path, save_dir)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-auto-eval', action='store_true')
    parser.add_argument('--resume-checkpoint', default=None)
    args = parser.parse_args()
    main(args.config, seed=args.seed, no_auto_eval=args.no_auto_eval, resume_checkpoint=args.resume_checkpoint)
