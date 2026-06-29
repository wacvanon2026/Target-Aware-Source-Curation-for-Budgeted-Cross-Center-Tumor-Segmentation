#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import yaml
DOMAINS = ['Art', 'Clipart', 'Product', 'RealWorld']

def main():
    parser = argparse.ArgumentParser(description='Generate OfficeHome Target+Full Source warmup configs for source selection.')
    parser.add_argument('--split-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--config-root', default='configs_cls_revise/officehome_warmup')
    parser.add_argument('--output-root', default='experiments_cls_revise/officehome')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--target-shots', type=int, default=3)
    parser.add_argument('--val-shots', type=int, default=2)
    parser.add_argument('--train-seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    args = parser.parse_args()
    split_root = Path(args.split_root)
    config_root = Path(args.config_root)
    output_root = Path(args.output_root)
    manifest = []
    for target in DOMAINS:
        split_dir = split_root / target / f'seed{args.split_seed:02d}'
        source_train = split_dir / 'source_train.txt'
        target_train = split_dir / f'target_train_{args.target_shots}shot.txt'
        target_val = split_dir / f'target_val_{args.val_shots}shot.txt'
        target_test = split_dir / 'target_test.txt'
        for train_seed in args.train_seeds:
            save_dir = output_root / target / 'warmup_full' / f'split{args.split_seed:02d}' / f'train_seed{train_seed:02d}'
            exp_name = f'officehome_{target}_warmup_full_split{args.split_seed:02d}_train{train_seed:02d}'
            cfg = {'experiment': {'name': exp_name, 'save_dir': save_dir.as_posix()}, 'data': {'num_classes': 65, 'source_train': source_train.as_posix(), 'source_val': target_val.as_posix(), 'target_selected': target_train.as_posix(), 'target_test': target_test.as_posix(), 'batch_size': args.batch_size, 'num_workers': args.num_workers}, 'model': {'backbone': 'resnet50', 'pretrained': True, 'num_classes': 65}, 'optimizer': {'lr': args.lr, 'weight_decay': args.weight_decay}, 'scheduler': {'T_max': args.epochs}, 'training': {'seed': train_seed, 'epochs': args.epochs}}
            cfg_path = config_root / target / f'split{args.split_seed:02d}' / f'train_seed{train_seed:02d}.yaml'
            cfg_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg_path.open('w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            manifest.append({'target': target, 'split_seed': args.split_seed, 'train_seed': train_seed, 'config': cfg_path.as_posix(), 'save_dir': save_dir.as_posix(), 'best_ckpt': (save_dir / 'best.pt').as_posix()})
    config_root.mkdir(parents=True, exist_ok=True)
    manifest_path = config_root / f'manifest_split{args.split_seed:02d}.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'Generated warmup configs: {len(manifest)}')
    print(f'Manifest: {manifest_path}')
if __name__ == '__main__':
    main()
