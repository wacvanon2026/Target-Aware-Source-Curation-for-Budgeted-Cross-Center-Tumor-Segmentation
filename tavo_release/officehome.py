from __future__ import annotations
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from .common import download_file, list_images, stable_shuffle, write_json, write_lines
from .matrix import OFFICEHOME_BUDGETS
DOMAINS = ('Art', 'Clipart', 'Product', 'RealWorld')

def download_officehome(url: str, output_dir: str | Path, filename: str='officehome.zip', overwrite: bool=False) -> Path:
    return download_file(url, Path(output_dir) / filename, overwrite=overwrite)

def extract_archive(archive: str | Path, output_dir: str | Path) -> Path:
    archive = Path(archive)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == '.zip':
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    else:
        raise ValueError(f'unsupported archive type: {archive}')
    return out

def image_label_pairs(domain_root: str | Path) -> list[tuple[str, str]]:
    pairs = []
    root = Path(domain_root)
    for image in list_images(root):
        if image.parent == root:
            continue
        pairs.append((str(image), image.parent.name))
    return sorted(pairs)

def group_by_label(pairs: list[tuple[str, str]]) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for path, label in pairs:
        grouped[label].append((path, label))
    return {label: sorted(items) for label, items in sorted(grouped.items())}

def split_target_shots(pairs: list[tuple[str, str]], train_shots: tuple[int, ...], val_shots: int, seed: int):
    rng = random.Random(seed)
    max_train = max(train_shots)
    train_by_shot = {shot: [] for shot in train_shots}
    val: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for label, items in group_by_label(pairs).items():
        pool = list(items)
        rng.shuffle(pool)
        required = max_train + val_shots + 1
        if len(pool) < required:
            raise ValueError(f'OfficeHome class {label} has {len(pool)} images, need at least {required}')
        for shot in train_shots:
            train_by_shot[shot].extend(pool[:shot])
        val.extend(pool[max_train:max_train + val_shots])
        test.extend(pool[max_train + val_shots:])
    return ({shot: sorted(items, key=lambda x: (x[1], x[0])) for shot, items in train_by_shot.items()}, sorted(val, key=lambda x: (x[1], x[0])), sorted(test, key=lambda x: (x[1], x[0])))

def class_balanced_random_source(pairs: list[tuple[str, str]], budget_per_class: int, seed: int):
    rng = random.Random(seed)
    selected: list[tuple[str, str]] = []
    for label, items in group_by_label(pairs).items():
        pool = list(items)
        if len(pool) < budget_per_class:
            raise ValueError(f'OfficeHome source class {label} has {len(pool)} images, need {budget_per_class}')
        selected.extend(rng.sample(pool, budget_per_class))
    return sorted(selected, key=lambda x: (x[1], x[0]))

def build_splits(data_root: str | Path, output_root: str | Path, target_domain: str, seed: int=42, target_train_shots: tuple[int, ...]=(3,), target_val_shots: int=2, source_budgets: tuple[int, ...]=OFFICEHOME_BUDGETS) -> dict[str, int | dict[str, int]]:
    data = Path(data_root)
    out = Path(output_root) / target_domain
    sources = [d for d in DOMAINS if d != target_domain]
    source_pairs: list[tuple[str, str]] = []
    for source in sources:
        source_pairs.extend(image_label_pairs(data / source))
    target_pairs = image_label_pairs(data / target_domain)
    target_train_by_shot, target_val, target_test = split_target_shots(target_pairs, tuple(sorted(target_train_shots)), target_val_shots, seed)
    write_pair_file(out / 'source_train.txt', sorted(source_pairs, key=lambda x: (x[1], x[0])))
    write_pair_file(out / 'source_pool.txt', sorted(source_pairs, key=lambda x: (x[1], x[0])))
    write_pair_file(out / f'target_val_{target_val_shots}shot.txt', target_val)
    write_pair_file(out / 'target_val.txt', target_val)
    write_pair_file(out / 'target_test.txt', target_test)
    for shot, items in target_train_by_shot.items():
        write_pair_file(out / f'target_train_{shot}shot.txt', items)
    default_shot = min(target_train_by_shot)
    write_pair_file(out / 'target_train.txt', target_train_by_shot[default_shot])
    for budget in source_budgets:
        subset = class_balanced_random_source(source_pairs, budget, seed)
        write_pair_file(out / 'random' / f'random_B{budget}.txt', subset)
        write_pair_file(out / 'random' / f'random_{budget}.txt', subset)
    summary = {'source_pool': len(source_pairs), 'target_train': {f'{shot}shot': len(items) for shot, items in target_train_by_shot.items()}, 'target_val': len(target_val), 'target_test': len(target_test), 'source_budgets': {f'B{budget}': budget * len(group_by_label(source_pairs)) for budget in source_budgets}}
    write_json(out / 'summary.json', summary)
    return summary

def write_pair_file(path: str | Path, pairs: list[tuple[str, str]]) -> Path:
    return write_lines(path, [f'{p} {label}' for p, label in pairs])

def build_train_command(config: str | Path, entrypoint: str='python -m tavo_release.officehome_train') -> list[str]:
    return entrypoint.split() + ['--config', str(Path(config))]

def build_config(output: str | Path, split_dir: str | Path, output_dir: str | Path, backbone: str='resnet50', epochs: int=30, batch_size: int=64, target_shots: int=3, val_shots: int=2) -> Path:
    split_dir = Path(split_dir)
    cfg = {'dataset': 'officehome', 'splits': {'source_train': str(split_dir / 'source_train.txt'), 'source_val': str(split_dir / f'target_val_{val_shots}shot.txt'), 'target_train': str(split_dir / f'target_train_{target_shots}shot.txt'), 'target_val': str(split_dir / f'target_val_{val_shots}shot.txt'), 'target_test': str(split_dir / 'target_test.txt')}, 'model': {'backbone': backbone, 'pretrained': True, 'num_classes': 65}, 'training': {'epochs': epochs, 'batch_size': batch_size, 'output_dir': str(Path(output_dir))}}
    return write_json(output, cfg)
