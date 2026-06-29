#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
DOMAINS = ['Art', 'Clipart', 'Product', 'RealWorld']
IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

def list_domain_images(data_root: Path, domain: str):
    items = []
    domain_root = data_root / domain
    for class_dir in sorted(domain_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTS:
                rel_path = path.as_posix()
                items.append({'path': rel_path, 'label': class_dir.name})
    return items

def write_list(items, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(f"{item['path']} {item['label']}\n")

def class_counts(items):
    counts = defaultdict(int)
    for item in items:
        counts[item['label']] += 1
    return dict(sorted(counts.items()))

def split_target(items, train_shots_list, val_shots: int, rng: random.Random):
    by_class = defaultdict(list)
    for item in items:
        by_class[item['label']].append(item)
    max_train_shots = max(train_shots_list)
    target_train_by_shot = {shot: [] for shot in train_shots_list}
    target_val = []
    target_test = []
    for label in sorted(by_class):
        pool = list(by_class[label])
        rng.shuffle(pool)
        required = max_train_shots + val_shots + 1
        if len(pool) < required:
            raise ValueError(f'Class {label} has {len(pool)} images, need at least {required} for train/val/test split.')
        for shot in train_shots_list:
            target_train_by_shot[shot].extend(pool[:shot])
        target_val.extend(pool[max_train_shots:max_train_shots + val_shots])
        target_test.extend(pool[max_train_shots + val_shots:])
    return ({shot: sorted(items, key=lambda x: (x['label'], x['path'])) for shot, items in target_train_by_shot.items()}, sorted(target_val, key=lambda x: (x['label'], x['path'])), sorted(target_test, key=lambda x: (x['label'], x['path'])))

def main():
    parser = argparse.ArgumentParser(description='Generate leave-one-domain-out OfficeHome splits for source selection.')
    parser.add_argument('--data-root', default='data_cls/office_home')
    parser.add_argument('--output-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--target-shots', type=int, nargs='+', default=[3])
    parser.add_argument('--val-shots', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    rng = random.Random(args.seed)
    all_items = {domain: list_domain_images(data_root, domain) for domain in DOMAINS}
    classes = sorted({item['label'] for items in all_items.values() for item in items})
    manifest = {'dataset': 'OfficeHome', 'domains': DOMAINS, 'num_classes': len(classes), 'target_shots': sorted(args.target_shots), 'max_target_shots': max(args.target_shots), 'val_shots': args.val_shots, 'seed': args.seed, 'targets': {}}
    for target in DOMAINS:
        source_domains = [domain for domain in DOMAINS if domain != target]
        source_train = []
        for domain in source_domains:
            source_train.extend(all_items[domain])
        source_train = sorted(source_train, key=lambda x: (x['label'], x['path']))
        target_train_by_shot, target_val, target_test = split_target(all_items[target], train_shots_list=sorted(args.target_shots), val_shots=args.val_shots, rng=rng)
        split_dir = output_root / target / f'seed{args.seed:02d}'
        write_list(source_train, split_dir / 'source_train.txt')
        for shot, target_train in target_train_by_shot.items():
            write_list(target_train, split_dir / f'target_train_{shot}shot.txt')
        write_list(target_val, split_dir / f'target_val_{args.val_shots}shot.txt')
        write_list(target_test, split_dir / 'target_test.txt')
        stats = {'target': target, 'source_domains': source_domains, 'source_train': len(source_train), 'target_train': {f'{shot}shot': len(target_train) for shot, target_train in target_train_by_shot.items()}, 'target_val': len(target_val), 'target_test': len(target_test), 'source_train_by_class': class_counts(source_train), 'target_train_by_class': {f'{shot}shot': class_counts(target_train) for shot, target_train in target_train_by_shot.items()}, 'target_val_by_class': class_counts(target_val), 'target_test_by_class': class_counts(target_test)}
        with (split_dir / 'stats.json').open('w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        manifest['targets'][target] = stats
        print(f"{target}: source={len(source_train)} target_train={stats['target_train']} target_val={len(target_val)} target_test={len(target_test)}")
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / f'manifest_seed{args.seed:02d}.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
if __name__ == '__main__':
    main()
