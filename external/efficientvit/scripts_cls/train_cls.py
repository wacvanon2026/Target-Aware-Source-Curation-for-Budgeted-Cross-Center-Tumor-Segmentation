#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import shutil
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from torchvision import transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classification.resnet50_cls import ResNet50Classifier
from models.classification.dataset_office import OfficeDataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, save_dir, filename='latest.pt'):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    torch.save(state, path)
    print(f'OK Saved checkpoint: {path}')

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for imgs, labels in tqdm(loader, desc=' Training', ncols=100):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return (avg_loss, acc)

def evaluate(model, loader, criterion, device, desc=' Evaluating'):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size
    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return (avg_loss, acc)

def build_transforms():
    train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return (train_transform, test_transform)

def load_datasets(cfg):
    train_transform, test_transform = build_transforms()
    source_train_txt = cfg['data'].get('source_train', None)
    source_val_txt = cfg['data'].get('source_val', None)
    target_selected_txt = cfg['data'].get('target_selected', None)
    target_test_txt = cfg['data']['target_test']
    train_datasets = []
    class_to_idx = None
    if source_train_txt is not None:
        source_train = OfficeDataset(source_train_txt, transform=train_transform, return_path=False)
        class_to_idx = source_train.class_to_idx
        train_datasets.append(source_train)
        print(f'OK Loaded source_train: {len(source_train)} samples')
        print('Example source samples:')
        for i in range(min(5, len(source_train.samples))):
            print('   ', source_train.samples[i])
    if target_selected_txt is not None:
        if class_to_idx is None:
            target_train = OfficeDataset(target_selected_txt, transform=train_transform, return_path=False)
            class_to_idx = target_train.class_to_idx
        else:
            target_train = OfficeDataset(target_selected_txt, transform=train_transform, class_to_idx=class_to_idx, return_path=False)
        train_datasets.append(target_train)
        print(f'OK Loaded target_selected: {len(target_train)} samples')
        print('Example target_selected samples:')
        for i in range(min(5, len(target_train.samples))):
            print('   ', target_train.samples[i])
        from collections import Counter
        labels = [x[1] for x in target_train.samples]
        print('Target class distribution:')
        print(Counter(labels))
    if len(train_datasets) == 0:
        raise ValueError('At least one of source_train or target_selected must be provided.')
    if len(train_datasets) == 1:
        train_dataset = train_datasets[0]
    else:
        train_dataset = ConcatDataset(train_datasets)
    val_dataset = None
    if source_val_txt is not None:
        val_dataset = OfficeDataset(source_val_txt, transform=test_transform, class_to_idx=class_to_idx, return_path=False)
        print(f'OK Loaded source_val: {len(val_dataset)} samples')
        print('Example source_val samples:')
        for i in range(min(5, len(val_dataset.samples))):
            print('   ', val_dataset.samples[i])
    else:
        print(' No validation set provided.')
    test_dataset = OfficeDataset(target_test_txt, transform=test_transform, class_to_idx=class_to_idx, return_path=False)
    print(f'OK Loaded target_test: {len(test_dataset)} samples')
    print('Example target_test samples:')
    for i in range(min(5, len(test_dataset.samples))):
        print('   ', test_dataset.samples[i])
    return (train_dataset, val_dataset, test_dataset)

def main(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg['training']['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(' Device:', device)
    model = ResNet50Classifier(num_classes=cfg['model']['num_classes'], pretrained=cfg['model']['pretrained']).to(device)
    print(' Loading datasets...')
    source_train_txt = cfg['data'].get('source_train', None)
    source_val_txt = cfg['data'].get('source_val', None)
    target_selected_txt = cfg['data'].get('target_selected', None)
    target_test_txt = cfg['data']['target_test']
    print('----- Dataset Paths -----')
    print('source_train:', source_train_txt)
    print('source_val:', source_val_txt)
    print('target_selected:', target_selected_txt)
    print('target_test:', target_test_txt)
    print('-------------------------')
    train_dataset, val_dataset, test_dataset = load_datasets(cfg)
    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True, num_workers=cfg['data']['num_workers'], pin_memory=True)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False, num_workers=cfg['data']['num_workers'], pin_memory=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg['optimizer']['lr'], weight_decay=cfg['optimizer']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['scheduler']['T_max'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    epochs = cfg['training']['epochs']
    save_dir = cfg['experiment']['save_dir']
    best_metric = -1
    use_val = val_loader is not None
    history = []
    print('\n Starting training...')
    for epoch in range(epochs):
        print(f'\n Epoch {epoch + 1}/{epochs}')
        lr = optimizer.param_groups[0]['lr']
        print(f'LR: {lr:.6f}')
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        if use_val:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device, ' Validating')
            metric_for_best = val_acc
            print(f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f}')
        else:
            val_loss, val_acc = (None, None)
            metric_for_best = train_acc
            print(f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | No Val Set')
        scheduler.step()
        checkpoint = {'epoch': epoch + 1, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict(), 'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc, 'config': cfg}
        save_checkpoint(checkpoint, save_dir, f'epoch_{epoch + 1:03d}.pt')
        if metric_for_best > best_metric:
            best_metric = metric_for_best
            save_checkpoint(checkpoint, save_dir, 'best.pt')
        history.append({'epoch': epoch + 1, 'val_acc': val_acc, 'val_loss': val_loss})
    print('\n Training finished')
    last_epoch = history[-1]['epoch']
    shutil.copy(os.path.join(save_dir, f'epoch_{last_epoch:03d}.pt'), os.path.join(save_dir, 'last.pt'))
    K = 5
    recent = history[-K:]
    if use_val:
        best_recent = sorted(recent, key=lambda x: (-x['val_acc'], x['val_loss']))[0]
        last_best_epoch = best_recent['epoch']
    else:
        last_best_epoch = last_epoch
    shutil.copy(os.path.join(save_dir, f'epoch_{last_best_epoch:03d}.pt'), os.path.join(save_dir, 'last_best.pt'))
    print(f'* Last-best epoch = {last_best_epoch}')

    def run_test(ckpt_name):
        print(f'\n Evaluating {ckpt_name}')
        ckpt = torch.load(os.path.join(save_dir, ckpt_name), map_location=device)
        model.load_state_dict(ckpt['model_state'])
        loss, acc = evaluate(model, test_loader, criterion, device, ' Testing')
        print(f'{ckpt_name} Target Acc: {acc:.4f}')
        return acc
    best_acc = run_test('best.pt')
    last_acc = run_test('last.pt')
    last_best_acc = run_test('last_best.pt')
    print('\n===== FINAL RESULTS =====')
    print('best.pt      :', best_acc)
    print('last.pt      :', last_acc)
    print('last_best.pt :', last_best_acc)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    main(args.config)
