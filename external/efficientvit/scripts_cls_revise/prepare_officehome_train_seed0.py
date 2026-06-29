#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
import yaml
TARGETS = ['Art', 'Clipart', 'Product', 'RealWorld']
SELECTION_METHODS = ['KMeans-B', 'KCenter-B', 'FacilityLocation-B', 'CRAIG-B', 'TargetMMD-B', 'TargetGradMatch-B', 'GLISTER-B', 'ORIENT-B']

def safe_name(name):
    return name.replace('-', '_').replace('+', 'Plus').replace(' ', '')

def read_items(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, label = line.rsplit(' ', 1)
            items.append({'path': img_path, 'label': label})
    return items

def write_items(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in items:
            f.write(f"{item['path']} {item['label']}\n")

def group_by_label(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[item['label']].append(item)
    return dict(sorted(grouped.items()))

def make_random_subset(source_list, out_path, budget_per_class, seed):
    source_items = read_items(source_list)
    grouped = group_by_label(source_items)
    rng = random.Random(seed)
    selected = []
    for label, pool in grouped.items():
        if len(pool) < budget_per_class:
            raise ValueError(f'{label} has {len(pool)} source images, need {budget_per_class}')
        selected.extend(rng.sample(pool, budget_per_class))
    selected = sorted(selected, key=lambda x: (x['label'], x['path']))
    write_items(out_path, selected)
    return out_path

def build_config(target, method, source_train, target_selected, target_val, target_test, save_dir, train_seed, epochs, batch_size, num_workers, lr, weight_decay):
    return {'experiment': {'name': f'officehome_{target}_{safe_name(method)}_train{train_seed:02d}', 'save_dir': save_dir.as_posix()}, 'data': {'num_classes': 65, 'source_train': source_train.as_posix() if source_train else None, 'source_val': target_val.as_posix(), 'target_selected': target_selected.as_posix() if target_selected else None, 'target_test': target_test.as_posix(), 'batch_size': batch_size, 'num_workers': num_workers}, 'model': {'backbone': 'resnet50', 'pretrained': True, 'num_classes': 65}, 'optimizer': {'lr': lr, 'weight_decay': weight_decay}, 'scheduler': {'T_max': epochs}, 'training': {'seed': train_seed, 'epochs': epochs}}

def write_config(path, cfg):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--subset-root', default='data_cls_revise/source_subsets/officehome')
    parser.add_argument('--selection-config-root', default='configs_cls_revise/officehome')
    parser.add_argument('--baseline-config-root', default='configs_cls_revise/officehome_baselines')
    parser.add_argument('--output-root', default='experiments_cls_revise/officehome')
    parser.add_argument('--manifest', default='configs_cls_revise/officehome_train_seed0_manifest.json')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--target-shots', type=int, default=3)
    parser.add_argument('--val-shots', type=int, default=2)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    args = parser.parse_args()
    split_root = Path(args.split_root)
    subset_root = Path(args.subset_root)
    selection_config_root = Path(args.selection_config_root)
    baseline_config_root = Path(args.baseline_config_root)
    output_root = Path(args.output_root)
    manifest_rows = []
    for target in TARGETS:
        for method in SELECTION_METHODS:
            method_key = safe_name(method)
            for budget in args.budgets:
                cfg_path = selection_config_root / target / method_key / f'B{budget}' / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}.yaml'
                if not cfg_path.exists():
                    raise FileNotFoundError(cfg_path)
                manifest_rows.append({'target': target, 'method': method, 'budget': f'B{budget}', 'train_seed': args.train_seed, 'config': cfg_path.as_posix(), 'category': 'source_selection'})
    for target in TARGETS:
        split_dir = split_root / target / f'seed{args.split_seed:02d}'
        source_train = split_dir / 'source_train.txt'
        target_train = split_dir / f'target_train_{args.target_shots}shot.txt'
        target_val = split_dir / f'target_val_{args.val_shots}shot.txt'
        target_test = split_dir / 'target_test.txt'
        baselines = [('Target-only', 'NoBudget', None, target_train), ('Source-only Full', 'Full', source_train, None), ('Target+Full Source', 'Full', source_train, target_train)]
        for method, budget, source_path, target_path in baselines:
            method_key = safe_name(method)
            save_dir = output_root / target / method_key / budget / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}'
            cfg = build_config(target=target, method=method, source_train=source_path, target_selected=target_path, target_val=target_val, target_test=target_test, save_dir=save_dir, train_seed=args.train_seed, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr, weight_decay=args.weight_decay)
            cfg_path = baseline_config_root / target / method_key / budget / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}.yaml'
            write_config(cfg_path, cfg)
            manifest_rows.append({'target': target, 'method': method, 'budget': budget, 'train_seed': args.train_seed, 'config': cfg_path.as_posix(), 'category': 'baseline'})
        for budget in args.budgets:
            random_subset = subset_root / target / f'seed{args.split_seed:02d}' / f'B{budget}' / f'Random-B_B{budget}_seed{args.split_seed:02d}.txt'
            make_random_subset(source_list=source_train, out_path=random_subset, budget_per_class=budget, seed=10000 + args.split_seed * 100 + budget)
            method = 'Random-B'
            method_key = safe_name(method)
            save_dir = output_root / target / method_key / f'B{budget}' / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}'
            cfg = build_config(target=target, method=method, source_train=random_subset, target_selected=target_train, target_val=target_val, target_test=target_test, save_dir=save_dir, train_seed=args.train_seed, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr, weight_decay=args.weight_decay)
            cfg_path = baseline_config_root / target / method_key / f'B{budget}' / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}.yaml'
            write_config(cfg_path, cfg)
            manifest_rows.append({'target': target, 'method': method, 'budget': f'B{budget}', 'train_seed': args.train_seed, 'config': cfg_path.as_posix(), 'source_subset': random_subset.as_posix(), 'category': 'baseline'})
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_rows, f, indent=2)
    print(f'Manifest rows: {len(manifest_rows)}')
    print(f'Manifest: {manifest_path}')
if __name__ == '__main__':
    main()
