#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.classification.dataset_office import OfficeDataset
from models.classification.resnet50_cls import ResNet50Classifier
TARGETS = ['Art', 'Clipart', 'Product', 'RealWorld']
STANDARD_METHODS = [('NoBudget', 'Target-only', 'officehome/{target}/Target_only/NoBudget/split00/train_seed{seed:02d}/last_best.pt'), ('Full', 'Source-only Full', 'officehome/{target}/Source_onlyFull/Full/split00/train_seed{seed:02d}/last_best.pt'), ('Full', 'Target + Full Source', 'officehome/{target}/TargetPlusFullSource/Full/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'Random', 'officehome/{target}/Random_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'KMeans', 'officehome/{target}/KMeans_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'KCenter', 'officehome/{target}/KCenter_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'FacilityLocation', 'officehome/{target}/FacilityLocation_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'CRAIG', 'officehome/{target}/CRAIG_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'TargetMMD', 'officehome/{target}/TargetMMD_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'TargetGradMatch', 'officehome/{target}/TargetGradMatch_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'GLISTER', 'officehome/{target}/GLISTER_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'ORIENT', 'officehome/{target}/ORIENT_B/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'TAVO_8D', 'officehome_tavo_formal_allbudgets/{target}/TAVO_8D/B15/split00/train_seed{seed:02d}/last_best.pt'), ('B15', 'TAVO_8D_best', 'officehome_tavo_formal_allbudgets/{target}/TAVO_8D_best/B15/split00/train_seed{seed:02d}/last_best.pt')]
DA_METHODS = [('B15', 'DANN', 'officehome_da/{target}/DANN_RandomB/B15/split00/train_seed{seed:02d}/final_results.json'), ('B15', 'MMD', 'officehome_da/{target}/MMD_RandomB/B15/split00/train_seed{seed:02d}/final_results.json'), ('B15', 'CORAL', 'officehome_da/{target}/CORAL_RandomB/B15/split00/train_seed{seed:02d}/final_results.json'), ('B15', 'CDAN', 'officehome_da/{target}/CDAN_RandomB/B15/split00/train_seed{seed:02d}/final_results.json')]

def build_test_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def macro_f1_and_balanced_acc(y_true: list[int], y_pred: list[int], num_classes: int) -> tuple[float, float]:
    f1_values: list[float] = []
    recalls: list[float] = []
    for c in range(num_classes):
        tp = sum((1 for t, p in zip(y_true, y_pred) if t == c and p == c))
        fp = sum((1 for t, p in zip(y_true, y_pred) if t != c and p == c))
        fn = sum((1 for t, p in zip(y_true, y_pred) if t == c and p != c))
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        f1_values.append(f1)
        recalls.append(recall)
    return (float(sum(f1_values) / len(f1_values)), float(sum(recalls) / len(recalls)))

def evaluate_checkpoint(ckpt_path: Path, device: torch.device, batch_size: int, num_workers: int) -> dict[str, float]:
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    num_classes = int(cfg['model']['num_classes'])
    model = ResNet50Classifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    dataset = OfficeDataset(cfg['data']['target_test'], transform=build_test_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    criterion = nn.CrossEntropyLoss(reduction='sum')
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(criterion(logits, labels).item())
            preds = logits.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = sum((int(t == p) for t, p in zip(y_true, y_pred))) / max(len(y_true), 1)
    macro_f1, balanced_acc = macro_f1_and_balanced_acc(y_true, y_pred, num_classes)
    return {'loss': total_loss / max(len(y_true), 1), 'acc': float(acc), 'macro_f1': macro_f1, 'balanced_acc': balanced_acc, 'num_examples': len(y_true)}

def read_da_json(path: Path) -> dict[str, float] | None:
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    stats = data.get('last_best.pt')
    if not stats:
        return None
    return {'loss': float(stats.get('loss', 0.0)), 'acc': float(stats.get('acc', 0.0)), 'macro_f1': float(stats.get('macro_f1', 0.0)), 'balanced_acc': float(stats.get('balanced_acc', 0.0)), 'num_examples': int(stats.get('num_examples', 0) or 0)}

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='experiments_cls_revise')
    parser.add_argument('--out-json', default='analysis_cls_revise/officehome_b15_macro_f1_last_best.json')
    parser.add_argument('--out-csv', default='analysis_cls_revise/officehome_b15_macro_f1_last_best.csv')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    args = parser.parse_args()
    root = Path(args.root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    rows: list[dict[str, object]] = []
    for target in TARGETS:
        for budget, method, template in STANDARD_METHODS:
            for seed in args.seeds:
                ckpt_path = root / template.format(target=target, seed=seed)
                if not ckpt_path.exists():
                    rows.append({'target': target, 'budget': budget, 'method': method, 'seed': seed, 'status': 'missing', 'path': str(ckpt_path)})
                    continue
                print(f'eval {target} {method} seed={seed}')
                stats = evaluate_checkpoint(ckpt_path, device, args.batch_size, args.num_workers)
                rows.append({'target': target, 'budget': budget, 'method': method, 'seed': seed, 'status': 'ok', 'path': str(ckpt_path), **stats})
        for budget, method, template in DA_METHODS:
            for seed in args.seeds:
                json_path = root / template.format(target=target, seed=seed)
                stats = read_da_json(json_path)
                if stats is None:
                    rows.append({'target': target, 'budget': budget, 'method': method, 'seed': seed, 'status': 'missing', 'path': str(json_path)})
                else:
                    rows.append({'target': target, 'budget': budget, 'method': method, 'seed': seed, 'status': 'ok', 'path': str(json_path), **stats})
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)
    fieldnames = ['target', 'budget', 'method', 'seed', 'status', 'loss', 'acc', 'macro_f1', 'balanced_acc', 'num_examples', 'path']
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    print(f'wrote {out_json}')
    print(f'wrote {out_csv}')
if __name__ == '__main__':
    main()
