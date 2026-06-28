#!/usr/bin/env python3
import json
import os
import random
import sys
from datetime import datetime
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.efficientvit_seg.efficientvit_seg import EfficientViT_Seg
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from models.efficientvit_seg.losses import DiceCELoss

def save_checkpoint(state, save_dir, filename='latest.pt'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, os.path.join(save_dir, filename))
    print(f'Saved checkpoint: {filename}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc='🧠 Training (AMP)', ncols=100):
        imgs = batch[0].to(device)
        masks = batch[1].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc='🔍 Evaluating', ncols=100):
            imgs = batch[0].to(device)
            masks = batch[1].to(device)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)

def load_datasets(data_cfg):
    train_datasets = []
    skip_empty_train = data_cfg.get('skip_empty_train', True)
    skip_empty_val = data_cfg.get('skip_empty_val', False)
    if 'domains' in data_cfg:
        for dom in data_cfg['domains']:
            ds = BraTSSliceDataset(root_dir=dom['path'], split=dom['split'], img_size=data_cfg['img_size'], split_txt_dir=dom.get('split_txt'), skip_empty=skip_empty_train)
            train_datasets.append(ds)
            print(f"✅ Loaded train domain: {dom['name']} ({len(ds)} slices)")
    else:
        ds = BraTSSliceDataset(root_dir=data_cfg['train_dir'], split='train', img_size=data_cfg['img_size'], split_txt_dir=data_cfg.get('split_txt_dir'), skip_empty=skip_empty_train)
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)
    print(f'📊 Total train dataset: {len(train_dataset)} slices')
    val_dataset = None
    if 'val' in data_cfg:
        val_cfg = data_cfg['val']
        val_dataset = BraTSSliceDataset(root_dir=val_cfg['path'], split=val_cfg['split'], img_size=data_cfg['img_size'], split_txt_dir=val_cfg.get('split_txt'), skip_empty=skip_empty_val)
    test_dataset = None
    if 'test' in data_cfg:
        test_cfg = data_cfg['test']
        test_dataset = BraTSSliceDataset(root_dir=test_cfg['path'], split=test_cfg['split'], img_size=data_cfg['img_size'], split_txt_dir=test_cfg.get('split_txt'), skip_empty=skip_empty_val)
    return (train_dataset, val_dataset, test_dataset)

def main(config_path, seed=None):
    cfg_path = config_path
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if seed is None:
        seed = cfg.get('training', {}).get('seed')
    if seed is not None:
        seed = int(seed)
        set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'🚀 Using device: {device}')
    model_cfg = cfg['model']
    model = EfficientViT_Seg(backbone=model_cfg['name'], in_channels=model_cfg['in_channels'], num_classes=model_cfg['num_classes'], pretrained=model_cfg.get('pretrained', True)).to(device)
    print('📁 Loading datasets...')
    train_dataset, val_dataset, test_dataset = load_datasets(cfg['data'])
    for ds in train_dataset.datasets:
        ds.return_meta = False
    loader_generator = None
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True, worker_init_fn=seed_worker if seed is not None else None, generator=loader_generator)
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True) if val_dataset else None
    criterion = DiceCELoss()
    optim_cfg = cfg['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=optim_cfg['lr'], weight_decay=optim_cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler']['T_max'], eta_min=cfg['scheduler']['eta_min'])
    scaler = torch.amp.GradScaler('cuda')
    train_cfg = cfg['training']
    save_dir = train_cfg['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    metadata = {'config_path': os.path.abspath(cfg_path), 'save_dir': os.path.abspath(save_dir), 'seed': seed, 'created_at': datetime.now().isoformat(timespec='seconds'), 'epochs': train_cfg['epochs'], 'lr': cfg['optimizer']['lr'], 'weight_decay': cfg['optimizer']['weight_decay'], 'scheduler_T_max': cfg['scheduler']['T_max'], 'scheduler_eta_min': cfg['scheduler']['eta_min']}
    with open(os.path.join(save_dir, 'run_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n🔥 Starting training for {train_cfg['epochs']} epochs...")
    best_loss = float('inf')
    val_history = []
    for epoch in range(train_cfg['epochs']):
        print(f"\n🟢 Epoch [{epoch + 1}/{train_cfg['epochs']}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device) if val_loader else 0
        val_history.append(val_loss)
        print(f'📊 Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        scheduler.step()
        checkpoint = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'val_loss': val_loss, 'config': cfg, 'seed': seed}
        save_checkpoint(checkpoint, save_dir, f'epoch_{epoch + 1:03d}.pt')
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(checkpoint, save_dir, 'best.pt')
    save_checkpoint(checkpoint, save_dir, 'last.pt')
    if len(val_history) >= 5:
        last_k = 5
        last_losses = val_history[-last_k:]
        min_idx = last_losses.index(min(last_losses))
        best_last_epoch = len(val_history) - last_k + min_idx + 1
        best_last_ckpt = os.path.join(save_dir, f'epoch_{best_last_epoch:03d}.pt')
        target_path = os.path.join(save_dir, 'best_last.pt')
        if os.path.exists(best_last_ckpt):
            torch.save(torch.load(best_last_ckpt), target_path)
            print(f'✅ Saved best_last.pt (epoch {best_last_epoch})')
    print(f'\n🏁 Training complete! Best Val Loss: {best_loss:.4f}')
    print('\n🚀 Starting 3D case-level evaluation...')
    from eval.evaluate_2d_to_3d import run_evaluation_2d_to_3d
    best_ckpt = os.path.join(save_dir, 'best.pt')
    print('\n🔎 Evaluating BEST checkpoint...')
    run_evaluation_2d_to_3d(config_path=cfg_path, checkpoint_path=best_ckpt, output_dir=save_dir)
    last_ckpt = os.path.join(save_dir, 'last.pt')
    print('\n🔎 Evaluating LAST checkpoint...')
    run_evaluation_2d_to_3d(config_path=cfg_path, checkpoint_path=last_ckpt, output_dir=save_dir)
    best_last_ckpt = os.path.join(save_dir, 'best_last.pt')
    if os.path.exists(best_last_ckpt):
        print('\n🔎 Evaluating BEST_LAST checkpoint...')
        run_evaluation_2d_to_3d(config_path=cfg_path, checkpoint_path=best_last_ckpt, output_dir=save_dir)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    main(args.config, seed=args.seed)
