#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
import numpy as np
import yaml
METHODS_8D = ('rds', 'less', 'orient', 'craig', 'gradmatch', 'kmeans', 'kcenter', 'diversity')

def read_lines(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]

def write_lines(path: str | Path, values) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('\n'.join((str(v) for v in values)) + '\n')
    return p

def read_json(path: str | Path) -> dict[str, float]:
    data = json.loads(Path(path).read_text())
    return {str(k): float(v) for k, v in data.items()}

def write_json(path: str | Path, value) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    return p

def parse_score_args(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        name, path = item.split('=', 1)
        if name not in METHODS_8D:
            raise SystemExit(f"Unknown score method {name!r}; expected one of {', '.join(METHODS_8D)}")
        out[name] = Path(path)
    missing = [name for name in METHODS_8D if name not in out]
    if missing:
        raise SystemExit(f"Missing score files for: {', '.join(missing)}")
    return out

def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    items = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    n = len(items)
    if n < 2:
        return {key: 1.0 for key, _ in items}
    value_to_indices: dict[float, list[int]] = defaultdict(list)
    for idx, (_, value) in enumerate(items):
        value_to_indices[float(value)].append(idx)
    out: dict[str, float] = {}
    for indices in value_to_indices.values():
        avg_rank = sum(indices) / len(indices)
        value = 1.0 - avg_rank / (n - 1)
        for idx in indices:
            out[items[idx][0]] = float(value)
    return out

def load_scores(paths: dict[str, Path], normalize: bool=True) -> dict[str, dict[str, float]]:
    loaded = {name: read_json(path) for name, path in paths.items()}
    if normalize:
        loaded = {name: rank_normalize(scores) for name, scores in loaded.items()}
    keys = set.intersection(*(set(scores) for scores in loaded.values()))
    if not keys:
        raise SystemExit('Score dictionaries have no shared source IDs')
    return {name: {key: loaded[name][key] for key in sorted(keys)} for name in METHODS_8D}

def project_simplex(values) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    z = np.clip(z, 0.0, None)
    total = float(z.sum())
    if total <= 0:
        return np.ones(len(z), dtype=np.float64) / len(z)
    return z / total

def fused_scores(scores: dict[str, dict[str, float]], weights) -> dict[str, float]:
    w = project_simplex(weights)
    return {case_id: float(sum((w[idx] * scores[name][case_id] for idx, name in enumerate(METHODS_8D)))) for case_id in next(iter(scores.values()))}

def topk(scores: dict[str, float], k: int) -> list[str]:
    return [case_id for case_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]]

def make_proxy_config(args, candidate_id: int, selected: list[str]) -> tuple[Path, Path]:
    candidate_root = Path(args.output_dir) / 'candidates' / f'candidate_{candidate_id:04d}'
    split_dir = candidate_root / 'split'
    target_train = read_lines(args.target_train)
    train_subjects = list(dict.fromkeys([*selected, *target_train]))
    write_lines(split_dir / 'train_subjects.txt', train_subjects)
    val_split_dir = candidate_root / 'val_split'
    write_lines(val_split_dir / 'val_subjects.txt', read_lines(args.target_val))
    save_dir = candidate_root / 'proxy'
    cfg = {'model': {'name': args.model_name, 'in_channels': args.in_channels, 'num_classes': args.num_classes, 'pretrained': not args.no_pretrained}, 'data': {'domains': [{'name': f'brats_tavo_candidate_{candidate_id:04d}', 'path': args.data_root, 'split': 'train', 'split_txt': str(split_dir)}], 'val': {'path': args.data_root, 'split': 'val', 'split_txt': str(val_split_dir)}, 'img_size': args.img_size, 'batch_size': args.batch_size, 'num_workers': args.num_workers, 'skip_empty_train': True, 'skip_empty_val': False}, 'trainer': {'max_iters': args.max_iters}, 'optimizer': {'lr': args.lr, 'weight_decay': args.weight_decay}, 'training': {'save_dir': str(save_dir)}}
    cfg_path = candidate_root / 'proxy_config.yaml'
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return (cfg_path, save_dir)

def proxy_fitness(args, candidate_id: int, selected: list[str]) -> float:
    from train_seg_short import main as run_short_training
    cfg_path, save_dir = make_proxy_config(args, candidate_id, selected)
    seeds = [int(x) for x in str(args.seeds_eval).replace(',', ' ').split() if x.strip()]
    run_short_training(str(cfg_path), seeds)
    val_mean = np.load(save_dir / 'val_losses_mean.npy')
    return -float(np.nanmean(val_mean))

def rank_weights(mu: int) -> np.ndarray:
    raw = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
    return raw / raw.sum()

def cma_search(args, scores: dict[str, dict[str, float]]):
    rng = np.random.default_rng(args.seed)
    dim = len(METHODS_8D)
    mean = np.ones(dim, dtype=np.float64) / dim
    cov = np.eye(dim, dtype=np.float64)
    sigma = args.sigma0
    mu = min(args.mu, args.popsize)
    recomb = rank_weights(mu)
    best = {'fitness': -float('inf'), 'weights': mean.tolist(), 'selected': [], 'candidate_id': None}
    history = []
    candidate_id = 0
    initial = [np.eye(dim)[idx] for idx in range(dim)] + [mean.copy()]
    for z in initial:
        weights = project_simplex(z)
        selected = topk(fused_scores(scores, weights), args.budget)
        fitness = float(np.mean([scores[name][case_id] for name in METHODS_8D for case_id in selected]))
        if not args.score_only and (not args.dry_run):
            fitness = proxy_fitness(args, candidate_id, selected)
        record = {'candidate_id': candidate_id, 'generation': -1, 'fitness': fitness, 'weights': weights.tolist()}
        history.append(record)
        if fitness > best['fitness']:
            best = {**record, 'selected': selected}
        candidate_id += 1
    if args.dry_run:
        return {'best': best, 'history': history}
    for generation in range(args.generations):
        eigvals, basis = np.linalg.eigh(0.5 * (cov + cov.T))
        eigvals = np.maximum(eigvals, 1e-09)
        transform = basis @ np.diag(np.sqrt(eigvals))
        candidates = []
        for _ in range(args.popsize):
            z = mean + sigma * (transform @ rng.standard_normal(dim))
            weights = project_simplex(z)
            selected = topk(fused_scores(scores, weights), args.budget)
            if args.score_only:
                fitness = float(np.mean([scores[name][case_id] for name in METHODS_8D for case_id in selected]))
            else:
                fitness = proxy_fitness(args, candidate_id, selected)
            record = {'candidate_id': candidate_id, 'generation': generation, 'fitness': float(fitness), 'weights': weights.tolist()}
            history.append(record)
            candidates.append((float(fitness), z, weights, selected, record))
            if fitness > best['fitness']:
                best = {**record, 'selected': selected}
            print(f'[gen {generation:02d}] candidate={candidate_id:04d} fitness={fitness:.6f} weights={weights}')
            candidate_id += 1
        candidates.sort(key=lambda item: item[0], reverse=True)
        old_mean = mean.copy()
        mean = sum((recomb[idx] * candidates[idx][1] for idx in range(mu)))
        centered = np.stack([candidates[idx][1] - old_mean for idx in range(mu)])
        cov = sum((recomb[idx] * np.outer(centered[idx], centered[idx]) for idx in range(mu))) + 1e-06 * np.eye(dim)
    return {'best': best, 'history': history}

def main() -> None:
    parser = argparse.ArgumentParser(description='BraTS 8D TAVO CMA-ES proxy search.')
    parser.add_argument('--score', action='append', required=True, help='method=score.json for each 8D criterion')
    parser.add_argument('--budget', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--data-root', default='data/brats')
    parser.add_argument('--target-train', default='splits/brats/target_train.txt')
    parser.add_argument('--target-val', default='splits/brats/target_val.txt')
    parser.add_argument('--generations', type=int, default=12)
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--mu', type=int, default=10)
    parser.add_argument('--sigma0', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds-eval', default='0')
    parser.add_argument('--max-iters', type=int, default=1000)
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight-decay', type=float, default=1e-05)
    parser.add_argument('--model-name', default='efficientvit_l1')
    parser.add_argument('--in-channels', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=4)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--score-only', action='store_true', help='Do not run proxy training; optimize aggregate normalized scores.')
    parser.add_argument('--dry-run', action='store_true', help='Validate score loading and write the corner/uniform search record only.')
    args = parser.parse_args()
    random.seed(args.seed)
    paths = parse_score_args(args.score)
    scores = load_scores(paths, normalize=True)
    result = cma_search(args, scores)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'search.json', {'methods': METHODS_8D, **result})
    write_lines(out / f'selection_{args.budget}.txt', result['best']['selected'])
    print(f"best_fitness={result['best']['fitness']:.6f}")
    print(f"selection={out / f'selection_{args.budget}.txt'}")
if __name__ == '__main__':
    main()
