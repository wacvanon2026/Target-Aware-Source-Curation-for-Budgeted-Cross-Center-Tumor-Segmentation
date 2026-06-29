#!/usr/bin/env python3
import csv
import re
from pathlib import Path
LOCAL_ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = 'external/efficientvit'
DATA_ROOT = f'{REMOTE_ROOT}/data/002_BraTS21'
LOCAL_TAVO_DATA = LOCAL_ROOT / 'data' / 'TAVO'
REMOTE_TAVO_DATA = f'{REMOTE_ROOT}/data/TAVO'
TARGETS = ['C4', 'C5', 'TCGA_LGG', 'TCGA_GBM']
BUDGETS = {'1T': 50, '3T': 150, '5T': 250}
TIME_BY_BUDGET = {'1T': '03:00:00', '3T': '06:00:00', '5T': '08:00:00'}
TRAIN_SEEDS = [0, 1, 2]
METHODS = [('rds', 'RDS'), ('gradmatch', 'GradMatch'), ('less', 'LESS'), ('orient', 'ORIENT'), ('diversity', 'Diversity'), ('kmeans', 'KMeans'), ('craig', 'CRAIG'), ('kcenter', 'KCenter'), ('tavo8d', 'TAVO_8D'), ('tavo8d_best', 'TAVO_8D_best')]
TARGET_SPLITS = {'C4': 'splits_C4_holdout/repeat_01', 'C5': 'split_C5_T22', 'TCGA_LGG': 'split_TCGA_LGG_T25', 'TCGA_GBM': 'split_TCGA_GBM_T40'}

def remote_from_local(path: Path) -> str:
    path_text = path.as_posix()
    root_text = LOCAL_ROOT.as_posix().rstrip('/') + '/'
    if not path_text.startswith(root_text):
        raise ValueError(f'{path_text} is not under {root_text}')
    rel = path_text[len(root_text):]
    return f'{REMOTE_ROOT}/{rel}'

def local_from_remote(path: str) -> Path:
    marker = f'{REMOTE_ROOT}/'
    if not path.startswith(marker):
        raise ValueError(path)
    return LOCAL_ROOT / path[len(marker):]

def read_subjects(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]

def write_subjects(path: Path, subjects: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(subjects) + '\n')

def ensure_prefix_split(source_dir: Path, dest_dir: Path, k: int) -> None:
    src = source_dir / 'train_subjects.txt'
    dst = dest_dir / 'train_subjects.txt'
    if not src.exists():
        raise FileNotFoundError(src)
    subjects = read_subjects(src)
    if len(subjects) < k:
        raise ValueError(f'{src} has {len(subjects)} subjects, need {k}')
    selected = subjects[:k]
    if dst.exists() and read_subjects(dst) == selected:
        return
    write_subjects(dst, selected)

def ensure_split_files(split_dir: Path, expected_train_count: int | None=None) -> None:
    train_file = split_dir / 'train_subjects.txt'
    if not train_file.exists():
        raise FileNotFoundError(train_file)
    if expected_train_count is not None:
        n = len(read_subjects(train_file))
        if n != expected_train_count:
            raise ValueError(f'{train_file} has {n} subjects, expected {expected_train_count}')

def method_split_dir(target: str, method: str, budget: str) -> tuple[Path, str]:
    if method == 'tavo8d':
        return (tavo_split_dir(target, budget, variant='8D'), 'from_old_8D_5T_config_prefix')
    if method == 'tavo8d_best':
        return (tavo_split_dir(target, budget, variant='8D_best'), 'from_old_8D_best_5T_config_prefix')
    if method == 'random':
        if target == 'C4':
            base = LOCAL_TAVO_DATA / 'splits_C4_random'
            split = base / f'random_{budget}'
        else:
            base = LOCAL_TAVO_DATA / f'splits_{target}_random' / 'seed00'
            split = base / f'random_{budget}'
        if budget == '3T':
            ensure_prefix_split(base / 'random_5T', split, BUDGETS[budget])
        return (split, 'random_seed00_prefix' if target != 'C4' else 'random_prefix')
    base = LOCAL_TAVO_DATA / f'splits_{target}_{method}'
    if target == 'C4' and (base / 'repeat01').exists():
        base = base / 'repeat01'
    split = base / f'{method}_{budget}'
    if budget == '3T':
        ensure_prefix_split(base / f'{method}_5T', split, BUDGETS[budget])
    return (split, 'criterion_rank_prefix')

def remap_to_tavo(path: str) -> str:
    marker = '/EfficientVit/data/'
    if marker in path and '/EfficientVit/data/TAVO/' not in path:
        return path.replace(marker, '/EfficientVit/data/TAVO/', 1)
    return path

def extract_first_source_split(config_text: str) -> str:
    match = re.search('split_txt:\\s*(.+)', config_text)
    if not match:
        raise ValueError('No split_txt found in TAVO config')
    return remap_to_tavo(match.group(1).strip())

def tavo_old_5t_config(target: str, variant: str) -> Path:
    if variant == '8D_best':
        if target == 'C4':
            raise FileNotFoundError('C4 does not have an old 8D_best config')
        return LOCAL_ROOT / 'configs_TAVO' / f'configs_{target}_cma' / 'train_config_BraTS21_8D_best_5T.yaml'
    if target == 'C4':
        return LOCAL_ROOT / 'configs_TAVO' / 'configs_C4_cma' / 'repeat01' / 'train_config_BraTS21_8D_5T.yaml'
    return LOCAL_ROOT / 'configs_TAVO' / f'configs_{target}_cma' / 'train_config_BraTS21_8D_5T.yaml'

def tavo_split_dir(target: str, budget: str, variant: str) -> Path:
    old_cfg = tavo_old_5t_config(target, variant)
    old_split_remote = extract_first_source_split(old_cfg.read_text())
    old_split = local_from_remote(old_split_remote)
    if budget == '5T':
        return old_split
    parts = list(old_split.parts)
    for i, part in enumerate(parts):
        if part == '5T':
            parts[i] = budget
            break
    new_split = Path(*parts)
    ensure_prefix_split(old_split, new_split, BUDGETS[budget])
    return new_split

def target_split_dir(target: str) -> Path:
    split = LOCAL_TAVO_DATA / TARGET_SPLITS[target]
    for name in ['train_subjects.txt', 'val_subjects.txt', 'test_subjects.txt']:
        if not (split / name).exists():
            raise FileNotFoundError(split / name)
    return split

def yaml_for_run(target: str, method_key: str, method_name: str, budget: str, seed: int, source_split: Path) -> str:
    k = BUDGETS[budget]
    target_split = target_split_dir(target)
    save_dir = f'{REMOTE_ROOT}/outputs_revise_step3/brats21/{target}/{method_name}/K{k}/train_seed{seed:02d}'
    source_split_remote = remote_from_local(source_split)
    target_split_remote = remote_from_local(target_split)
    return f'model:\n  name: efficientvit_l1\n  in_channels: 4\n  num_classes: 4\n  pretrained: true\n\ndata:\n  skip_empty_train: true\n  skip_empty_val: false\n  domains:\n    - name: BraTS21_{target}_{method_name}_K{k}\n      path: {DATA_ROOT}\n      split: train\n      split_txt: {source_split_remote}\n\n    - name: BraTS21_{target}_T_train\n      path: {DATA_ROOT}\n      split: train\n      split_txt: {target_split_remote}\n\n  val:\n    path: {DATA_ROOT}\n    split: val\n    split_txt: {target_split_remote}\n\n  test:\n    path: {DATA_ROOT}\n    split: test\n    split_txt: {target_split_remote}\n\n  img_size: 512\n  batch_size: 4\n  num_workers: 4\n\noptimizer:\n  lr: 0.0001\n  weight_decay: 0.00001\n\nscheduler:\n  T_max: 20\n  eta_min: 0.000001\n\ntraining:\n  epochs: 20\n  seed: {seed}\n  save_dir: {save_dir}\n  feature_mode: false\n'

def main() -> None:
    config_root = LOCAL_ROOT / 'configs_revise_step3' / 'brats21'
    analysis_root = LOCAL_ROOT / 'analysis' / 'revise_step3'
    config_root.mkdir(parents=True, exist_ok=True)
    analysis_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for target in TARGETS:
        for method_key, method_name in METHODS:
            if method_key == 'tavo8d_best' and target == 'C4':
                continue
            for budget, k in BUDGETS.items():
                source_split, provenance = method_split_dir(target, method_key, budget)
                ensure_split_files(source_split, expected_train_count=k)
                for seed in TRAIN_SEEDS:
                    cfg_dir = config_root / target / method_name / f'K{k}'
                    cfg_dir.mkdir(parents=True, exist_ok=True)
                    cfg_path = cfg_dir / f'train_seed{seed:02d}.yaml'
                    cfg_path.write_text(yaml_for_run(target, method_key, method_name, budget, seed, source_split))
                    rows.append({'target': target, 'method': method_name, 'budget_label': budget, 'budget_cases': k, 'train_seed': seed, 'time_limit': TIME_BY_BUDGET[budget], 'config_path': remote_from_local(cfg_path), 'output_dir': f'{REMOTE_ROOT}/outputs_revise_step3/brats21/{target}/{method_name}/K{k}/train_seed{seed:02d}', 'source_split': remote_from_local(source_split), 'source_split_provenance': provenance})
    fieldnames = list(rows[0])
    manifest = analysis_root / 'revise_step3_brats_train_manifest.csv'
    with manifest.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    for budget, k in BUDGETS.items():
        budget_rows = [row for row in rows if row['budget_label'] == budget]
        path = analysis_root / f'revise_step3_brats_train_manifest_K{k}.csv'
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(budget_rows)
    print(f'Wrote {len(rows)} jobs')
    print(manifest)
    for budget, k in BUDGETS.items():
        print(f"K{k}: {sum((row['budget_label'] == budget for row in rows))} jobs")
if __name__ == '__main__':
    main()
