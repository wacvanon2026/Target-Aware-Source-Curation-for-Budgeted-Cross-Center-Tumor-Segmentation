#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import yaml
DOMAINS = ['Art', 'Clipart', 'Product', 'RealWorld']
METHODS = ['KMeans-B', 'KCenter-B', 'FacilityLocation-B', 'CRAIG-B', 'TargetMMD-B', 'TargetGradMatch-B', 'GLISTER-B', 'ORIENT-B']

def safe_name(name):
    return name.replace('-', '_')

def main():
    parser = argparse.ArgumentParser(description='Generate OfficeHome training configs for source-selection baselines.')
    parser.add_argument('--split-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--subset-root', default='data_cls_revise/source_subsets/officehome')
    parser.add_argument('--config-root', default='configs_cls_revise/officehome')
    parser.add_argument('--output-root', default='experiments_cls_revise/officehome')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--target-shots', type=int, default=3)
    parser.add_argument('--val-shots', type=int, default=2)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 5])
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--methods', nargs='+', default=METHODS)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    args = parser.parse_args()
    split_root = Path(args.split_root)
    subset_root = Path(args.subset_root)
    config_root = Path(args.config_root)
    output_root = Path(args.output_root)
    manifest = []
    for target in DOMAINS:
        split_dir = split_root / target / f'seed{args.split_seed:02d}'
        target_train = split_dir / f'target_train_{args.target_shots}shot.txt'
        target_val = split_dir / f'target_val_{args.val_shots}shot.txt'
        target_test = split_dir / 'target_test.txt'
        for budget in args.budgets:
            subset_dir = subset_root / target / f'seed{args.split_seed:02d}' / f'B{budget}'
            for method in args.methods:
                method_key = safe_name(method)
                subset_path = subset_dir / f'{method}_B{budget}_seed{args.split_seed:02d}.txt'
                for train_seed in args.train_seeds:
                    exp_name = f'officehome_{target}_TargetPlus{method_key}_B{budget}_split{args.split_seed:02d}_train{train_seed:02d}'
                    save_dir = output_root / target / method_key / f'B{budget}' / f'split{args.split_seed:02d}' / f'train_seed{train_seed:02d}'
                    cfg = {'experiment': {'name': exp_name, 'save_dir': save_dir.as_posix()}, 'data': {'num_classes': 65, 'source_train': subset_path.as_posix(), 'source_val': target_val.as_posix(), 'target_selected': target_train.as_posix(), 'target_test': target_test.as_posix(), 'batch_size': args.batch_size, 'num_workers': args.num_workers}, 'model': {'backbone': 'resnet50', 'pretrained': True, 'num_classes': 65}, 'optimizer': {'lr': args.lr, 'weight_decay': args.weight_decay}, 'scheduler': {'T_max': args.epochs}, 'training': {'seed': train_seed, 'epochs': args.epochs}}
                    cfg_path = config_root / target / method_key / f'B{budget}' / f'split{args.split_seed:02d}' / f'train_seed{train_seed:02d}.yaml'
                    cfg_path.parent.mkdir(parents=True, exist_ok=True)
                    with cfg_path.open('w', encoding='utf-8') as f:
                        yaml.safe_dump(cfg, f, sort_keys=False)
                    manifest.append({'target': target, 'method': method, 'budget_per_class': budget, 'split_seed': args.split_seed, 'train_seed': train_seed, 'config': cfg_path.as_posix(), 'source_subset': subset_path.as_posix(), 'save_dir': save_dir.as_posix()})
    config_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config_root / f'manifest_split{args.split_seed:02d}.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'Generated configs: {len(manifest)}')
    print(f'Manifest: {manifest_path}')
if __name__ == '__main__':
    main()
