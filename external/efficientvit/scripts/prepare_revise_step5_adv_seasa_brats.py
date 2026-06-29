#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import yaml
REMOTE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REMOTE_ROOT / 'data' / '002_BraTS21'
TAVO_DATA = REMOTE_ROOT / 'data' / 'TAVO'
TARGETS = ['C4', 'C5', 'TCGA_LGG', 'TCGA_GBM']
TRAIN_SEEDS = [0]
BUDGETS = {'K50': 50, 'K150': 150, 'K250': 250}
TIME_LIMITS = {'Full': '48:00:00', 'K50': '08:00:00', 'K150': '16:00:00', 'K250': '24:00:00'}
TARGET_SPLITS = {'C4': TAVO_DATA / 'splits_C4_holdout' / 'repeat_01', 'C5': TAVO_DATA / 'split_C5_T22', 'TCGA_LGG': TAVO_DATA / 'split_TCGA_LGG_T25', 'TCGA_GBM': TAVO_DATA / 'split_TCGA_GBM_T40'}
SOURCE_FULL_SPLITS = {target: TAVO_DATA / f'splits_{target}_source' for target in TARGETS}

def read_subjects(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def write_subjects(path: Path, subjects: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(subjects) + '\n')

def ensure_split_file(split_dir: Path, split: str='train', expected_count: int | None=None) -> None:
    path = split_dir / f'{split}_subjects.txt'
    if not path.exists():
        raise FileNotFoundError(path)
    if expected_count is not None:
        n = len(read_subjects(path))
        if n != expected_count:
            raise ValueError(f'{path} has {n} subjects, expected {expected_count}')

def ensure_target_split(target: str) -> Path:
    split = TARGET_SPLITS[target]
    for name in ['train', 'val', 'test']:
        ensure_split_file(split, name)
    return split

def ensure_randomk_split(target: str, budget: str) -> Path:
    k = BUDGETS[budget]
    if target == 'C4':
        base = TAVO_DATA / 'splits_C4_random'
        split = base / f'random_{k // 50}T'
        source_250 = base / 'random_5T'
    else:
        base = TAVO_DATA / f'splits_{target}_random' / 'seed00'
        split = base / f'random_{k // 50}T'
        source_250 = base / 'random_5T'
    if budget == 'K150' and (not (split / 'train_subjects.txt').exists()):
        subjects = read_subjects(source_250 / 'train_subjects.txt')
        if len(subjects) < k:
            raise ValueError(f'{source_250} has fewer than {k} subjects')
        write_subjects(split / 'train_subjects.txt', subjects[:k])
    ensure_split_file(split, 'train', expected_count=k)
    return split

def source_split(target: str, source_kind: str, budget: str) -> tuple[Path, int, str]:
    if source_kind == 'Full':
        split = SOURCE_FULL_SPLITS[target]
        ensure_split_file(split, 'train')
        return (split, len(read_subjects(split / 'train_subjects.txt')), 'full_source')
    split = ensure_randomk_split(target, budget)
    return (split, BUDGETS[budget], 'random_seed00_prefix')

def method_da_cfg(method_name: str, smoke: bool) -> dict:
    if method_name.startswith('ADVENT_AdvEnt'):
        return {'method': 'advent_advent', 'official_reference': 'valeoai/advent AdvEnt: adversarial entropy minimization on output entropy maps', 'lambda_max': 0.001, 'lambda_schedule': 'fixed', 'target_seg_weight': 1.0, 'steps_per_epoch': 2 if smoke else 'auto', 'lr_d': 0.0001, 'beta1_d': 0.9, 'beta2_d': 0.99, 'output_discriminator_ndf': 64}
    if method_name.startswith('SE_ASA'):
        return {'method': 'se_asa', 'official_reference': 'fengweie/SE_ASA: selective entropy constraints and adaptive semantic alignment', 'lambda_max': 0.003, 'lambda_schedule': 'fixed', 'target_seg_weight': 1.0, 'steps_per_epoch': 2 if smoke else 'auto', 'lr_d': 0.0001, 'beta1_d': 0.9, 'beta2_d': 0.99, 'output_discriminator_ndf': 64, 'seasa_lambda_class': 0.1, 'seasa_lambda_selective': 0.01, 'seasa_class_center_momentum': 0.01, 'seasa_num_aug': 3, 'seasa_consistency_threshold': 2, 'seasa_fourier_beta': 0.01, 'seasa_noise_std': 0.03}
    raise ValueError(method_name)

def da_config(*, target: str, method_name: str, source_kind: str, budget: str, seed: int, source_dir: Path, output_root: Path, smoke: bool=False) -> dict:
    target_dir = ensure_target_split(target)
    epochs = 1 if smoke else 20
    save_dir = output_root / target / method_name / budget / f'train_seed{seed:02d}'
    return {'model': {'name': 'efficientvit_l1', 'in_channels': 4, 'num_classes': 4, 'pretrained': True}, 'data': {'skip_empty_train': True, 'skip_empty_val': False, 'skip_empty_align': True, 'source': {'name': f'BraTS21_{target}_{source_kind}', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(source_dir)}, 'target': {'name': f'BraTS21_{target}_Ttrain', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(target_dir)}, 'target_align': {'name': f'BraTS21_{target}_Ttrain_align', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(target_dir)}, 'val': {'path': str(DATA_ROOT), 'split': 'val', 'split_txt': str(target_dir)}, 'test': {'path': str(DATA_ROOT), 'split': 'test', 'split_txt': str(target_dir)}, 'img_size': 512, 'batch_size': 4, 'num_workers': 4}, 'optimizer': {'lr': 0.0001, 'weight_decay': 1e-05}, 'scheduler': {'T_max': epochs, 'eta_min': 1e-06}, 'training': {'epochs': epochs, 'seed': seed, 'save_dir': str(save_dir), 'auto_eval': not smoke, 'keep_epoch_checkpoints': False, 'feature_mode': False}, 'da': method_da_cfg(method_name, smoke=smoke)}

def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        yaml.safe_dump(data, f, sort_keys=False)

def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_rows(config_root: Path, output_root: Path, smoke: bool=False) -> list[dict]:
    rows: list[dict] = []
    targets = ['C4'] if smoke else TARGETS
    seeds = [0]
    methods = [('ADVENT_AdvEnt_Full', 'Full', 'Full'), ('ADVENT_AdvEnt_RandomK', 'RandomK', 'K50'), ('ADVENT_AdvEnt_RandomK', 'RandomK', 'K150'), ('ADVENT_AdvEnt_RandomK', 'RandomK', 'K250'), ('SE_ASA_Full', 'Full', 'Full'), ('SE_ASA_RandomK', 'RandomK', 'K50'), ('SE_ASA_RandomK', 'RandomK', 'K150'), ('SE_ASA_RandomK', 'RandomK', 'K250')]
    if smoke:
        methods = [('ADVENT_AdvEnt_RandomK', 'RandomK', 'K50'), ('SE_ASA_RandomK', 'RandomK', 'K50')]
    for target in targets:
        for method_name, source_kind, budget in methods:
            src, source_cases, provenance = source_split(target, source_kind, budget)
            da_method = method_da_cfg(method_name, smoke=smoke)['method']
            for seed in seeds:
                cfg = da_config(target=target, method_name=method_name, source_kind=source_kind, budget=budget, seed=seed, source_dir=src, output_root=output_root, smoke=smoke)
                cfg_path = config_root / target / method_name / budget / f'train_seed{seed:02d}.yaml'
                write_yaml(cfg_path, cfg)
                rows.append({'target': target, 'method': method_name, 'da_method': da_method, 'source_kind': source_kind, 'budget': budget, 'source_cases': source_cases, 'train_seed': seed, 'time_limit': '00:45:00' if smoke else TIME_LIMITS[budget], 'config_path': str(cfg_path), 'output_dir': cfg['training']['save_dir'], 'source_split': str(src), 'target_split': str(TARGET_SPLITS[target]), 'source_split_provenance': provenance})
    return rows

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=REMOTE_ROOT)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_root = project_root / 'configs_revise_step5_adv_seasa' / 'brats21'
    output_root = project_root / 'outputs_revise_step5_adv_seasa' / 'brats21'
    analysis_root = project_root / 'analysis' / 'revise_step5_adv_seasa'
    rows = build_rows(config_root, output_root, smoke=False)
    write_manifest(analysis_root / 'revise_step5_adv_seasa_brats_train_manifest.csv', rows)
    for budget in ['Full', 'K50', 'K150', 'K250']:
        write_manifest(analysis_root / f'revise_step5_adv_seasa_brats_train_manifest_{budget}.csv', [row for row in rows if row['budget'] == budget])
    smoke_rows = build_rows(project_root / 'configs_revise_step5_adv_seasa_smoke' / 'brats21', project_root / 'outputs_revise_step5_adv_seasa_smoke' / 'brats21', smoke=True)
    write_manifest(analysis_root / 'revise_step5_adv_seasa_brats_smoke_manifest.csv', smoke_rows)
    print(f'Wrote ADVENT/SE-ASA jobs: {len(rows)}')
    for budget in ['Full', 'K50', 'K150', 'K250']:
        print(f"{budget}: {sum((row['budget'] == budget for row in rows))}")
    print(f'Wrote smoke jobs: {len(smoke_rows)}')
if __name__ == '__main__':
    main()
