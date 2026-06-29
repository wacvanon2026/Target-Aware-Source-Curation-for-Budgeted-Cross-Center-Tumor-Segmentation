#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import yaml
TARGETS = ['Art', 'Clipart', 'Product', 'RealWorld']
DA_METHODS = [('DANN-Full', 'dann', 'Full', None), ('MMD-Full', 'mmd', 'Full', None), ('CORAL-Full', 'coral', 'Full', None), ('CDAN-Full', 'cdan_e', 'Full', None)]
RANDOM_DA_METHODS = [('DANN-RandomB', 'dann'), ('MMD-RandomB', 'mmd'), ('CORAL-RandomB', 'coral'), ('CDAN-RandomB', 'cdan_e')]

def safe_name(name: str) -> str:
    return name.replace('-', '_').replace('+', 'Plus').replace(' ', '')

def count_lines(path: Path) -> int:
    return sum((1 for line in path.read_text().splitlines() if line.strip()))

def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path

def build_config(*, target: str, method_name: str, da_method: str, budget: str, source_train: Path, target_train: Path, target_val: Path, target_test: Path, save_dir: Path, train_seed: int, epochs: int, batch_size: int, num_workers: int, lr: float, weight_decay: float, smoke: bool) -> dict:
    steps_per_epoch: int | str = 2 if smoke else 'auto'
    epochs = 1 if smoke else epochs
    return {'experiment': {'name': f'officehome_{target}_{safe_name(method_name)}_seed{train_seed:02d}', 'save_dir': save_dir.as_posix()}, 'data': {'num_classes': 65, 'source_train': source_train.as_posix(), 'source_val': target_val.as_posix(), 'target_selected': target_train.as_posix(), 'target_test': target_test.as_posix(), 'batch_size': batch_size, 'num_workers': num_workers}, 'model': {'backbone': 'resnet50', 'pretrained': True, 'num_classes': 65}, 'optimizer': {'lr': lr, 'weight_decay': weight_decay}, 'scheduler': {'T_max': epochs}, 'training': {'seed': train_seed, 'epochs': epochs}, 'da': {'method': da_method, 'lambda_max': 0.1, 'target_ce_weight': 1.0, 'steps_per_epoch': steps_per_epoch, 'domain_hidden_dim': 1024, 'domain_dropout': 0.5, 'kernel_multipliers': [0.25, 0.5, 1.0, 2.0, 4.0], 'fixed_sigma': 'auto', 'entropy_conditioning': True, 'random_dim': 1024}}

def write_yaml(path: Path, cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def write_json(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def build_rows(args: argparse.Namespace, smoke: bool=False) -> list[dict]:
    rows = []
    config_root = Path(args.smoke_config_root if smoke else args.config_root)
    output_root = Path(args.smoke_output_root if smoke else args.output_root)
    targets = [args.smoke_target] if smoke else TARGETS
    budgets = [args.smoke_budget] if smoke else args.budgets
    method_specs = [(method_name, da_method, 'Random-B', args.smoke_budget) for method_name, da_method in RANDOM_DA_METHODS] if smoke else DA_METHODS
    for target in targets:
        split_dir = Path(args.split_root) / target / f'seed{args.split_seed:02d}'
        source_full = require(split_dir / 'source_train.txt')
        target_train = require(split_dir / f'target_train_{args.target_shots}shot.txt')
        target_val = require(split_dir / f'target_val_{args.val_shots}shot.txt')
        target_test = require(split_dir / 'target_test.txt')
        specs = list(method_specs)
        if not smoke:
            for budget in budgets:
                for method_name, da_method in RANDOM_DA_METHODS:
                    specs.append((method_name, da_method, 'Random-B', budget))
        for method_name, da_method, source_kind, budget in specs:
            if source_kind == 'Full':
                source_train = source_full
                budget_label = 'Full'
                source_provenance = 'full_source'
            else:
                budget_label = f'B{budget}'
                source_train = require(Path(args.subset_root) / target / f'seed{args.split_seed:02d}' / budget_label / f'Random-B_{budget_label}_seed{args.split_seed:02d}.txt')
                source_provenance = 'existing_random_b_subset'
            method_key = safe_name(method_name)
            cfg_path = config_root / target / method_key / budget_label / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}.yaml'
            save_dir = output_root / target / method_key / budget_label / f'split{args.split_seed:02d}' / f'train_seed{args.train_seed:02d}'
            cfg = build_config(target=target, method_name=method_name, da_method=da_method, budget=budget_label, source_train=source_train, target_train=target_train, target_val=target_val, target_test=target_test, save_dir=save_dir, train_seed=args.train_seed, epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr, weight_decay=args.weight_decay, smoke=smoke)
            write_yaml(cfg_path, cfg)
            rows.append({'target': target, 'method': method_name, 'da_method': da_method, 'budget': budget_label, 'train_seed': args.train_seed, 'config': cfg_path.as_posix(), 'output_dir': save_dir.as_posix(), 'source_split': source_train.as_posix(), 'target_split': split_dir.as_posix(), 'target_train': target_train.as_posix(), 'target_val': target_val.as_posix(), 'target_test': target_test.as_posix(), 'source_count': count_lines(source_train), 'target_train_count': count_lines(target_train), 'target_val_count': count_lines(target_val), 'target_test_count': count_lines(target_test), 'source_provenance': source_provenance, 'category': 'officehome_cls_da_smoke' if smoke else 'officehome_cls_da'})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--subset-root', default='data_cls_revise/source_subsets/officehome')
    parser.add_argument('--config-root', default='configs_cls_revise/officehome_da')
    parser.add_argument('--output-root', default='experiments_cls_revise/officehome_da')
    parser.add_argument('--smoke-config-root', default='configs_cls_revise/officehome_da_smoke')
    parser.add_argument('--smoke-output-root', default='experiments_cls_revise/officehome_da_smoke')
    parser.add_argument('--manifest', default='configs_cls_revise/officehome_da/manifest_seed00.json')
    parser.add_argument('--manifest-csv', default='configs_cls_revise/officehome_da/manifest_seed00.csv')
    parser.add_argument('--smoke-manifest', default='configs_cls_revise/officehome_da/smoke_manifest_seed00.json')
    parser.add_argument('--smoke-manifest-csv', default='configs_cls_revise/officehome_da/smoke_manifest_seed00.csv')
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--target-shots', type=int, default=3)
    parser.add_argument('--val-shots', type=int, default=2)
    parser.add_argument('--budgets', type=int, nargs='+', default=[1, 3, 5, 8, 15, 25])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--smoke-target', default='Art', choices=TARGETS)
    parser.add_argument('--smoke-budget', type=int, default=1)
    args = parser.parse_args()
    rows = build_rows(args, smoke=False)
    smoke_rows = build_rows(args, smoke=True)
    write_json(Path(args.manifest), rows)
    write_csv(Path(args.manifest_csv), rows)
    write_json(Path(args.smoke_manifest), smoke_rows)
    write_csv(Path(args.smoke_manifest_csv), smoke_rows)
    print(f'Wrote DA rows: {len(rows)}')
    print(f'Wrote smoke rows: {len(smoke_rows)}')
    print(f'Manifest: {args.manifest}')
    print(f'CSV: {args.manifest_csv}')
if __name__ == '__main__':
    main()
