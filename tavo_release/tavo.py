from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from .common import read_json, write_json, write_lines
METHODS_8D = ('rds', 'less', 'orient', 'craig', 'gradmatch', 'kmeans', 'kcenter', 'diversity')

def rank_normalize(scores: dict[str, float]) -> dict[str, float]:
    items = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    n = len(items)
    if n < 2:
        return {key: 1.0 for key, _ in items}
    value_to_indices: dict[float, list[int]] = defaultdict(list)
    for idx, (_, value) in enumerate(items):
        value_to_indices[float(value)].append(idx)
    out = {}
    for indices in value_to_indices.values():
        avg_rank = sum(indices) / len(indices)
        value = 1.0 - avg_rank / (n - 1)
        for idx in indices:
            out[items[idx][0]] = float(value)
    return out

def align_score_dicts(score_dicts: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    key_sets = [set(v) for v in score_dicts.values()]
    if not key_sets:
        raise ValueError('no score dictionaries')
    keys = set.intersection(*key_sets)
    if not keys:
        raise ValueError('score dictionaries have no shared keys')
    return {name: {key: float(scores[key]) for key in sorted(keys)} for name, scores in score_dicts.items()}

def project_simplex(values) -> np.ndarray:
    z = np.asarray(values, dtype=np.float64)
    z = np.clip(z, 0.0, None)
    total = float(z.sum())
    if total <= 0:
        return np.ones(len(z), dtype=np.float64) / len(z)
    return z / total

def fuse_scores(score_dicts: dict[str, dict[str, float]], weights) -> dict[str, float]:
    names = list(score_dicts)
    w = project_simplex(weights)
    if len(w) != len(names):
        raise ValueError('weight dimension does not match score dictionaries')
    keys = list(next(iter(score_dicts.values())))
    fused = {}
    for key in keys:
        fused[key] = float(sum((float(w[idx]) * float(score_dicts[name][key]) for idx, name in enumerate(names))))
    return fused

def select_topk(scores: dict[str, float], k: int) -> list[str]:
    ranked = sorted(scores.items(), key=lambda item: (-float(item[1]), item[0]))
    return [key for key, _ in ranked[:k]]

def load_score_jsons(score_paths: dict[str, str | Path], normalize: bool=True) -> dict[str, dict[str, float]]:
    loaded = {name: {str(k): float(v) for k, v in read_json(path).items()} for name, path in score_paths.items()}
    if normalize:
        loaded = {name: rank_normalize(scores) for name, scores in loaded.items()}
    return align_score_dicts(loaded)

def write_selection(score_paths: dict[str, str | Path], weights, budget: int, output: str | Path, normalize: bool=True) -> list[str]:
    scores = load_score_jsons(score_paths, normalize=normalize)
    selected = select_topk(fuse_scores(scores, weights), budget)
    write_lines(output, selected)
    return selected

def cma_es(objective, dim: int, seed: int=0, popsize: int=20, generations: int=12, sigma0: float=0.3):
    rng = np.random.default_rng(seed)
    mean = np.ones(dim, dtype=np.float64) / dim
    cov = np.eye(dim, dtype=np.float64)
    sigma = float(sigma0)
    mu = max(1, popsize // 2)
    raw_weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
    recomb = raw_weights / raw_weights.sum()
    best = {'score': -float('inf'), 'weights': mean.tolist(), 'generation': -1}
    history = []
    for gen in range(generations):
        eigvals, basis = np.linalg.eigh(0.5 * (cov + cov.T))
        eigvals = np.maximum(eigvals, 1e-09)
        transform = basis @ np.diag(np.sqrt(eigvals))
        candidates = []
        for _ in range(popsize):
            z = mean + sigma * (transform @ rng.standard_normal(dim))
            w = project_simplex(z)
            score = float(objective(w))
            candidates.append((score, z, w))
            if score > best['score']:
                best = {'score': score, 'weights': w.tolist(), 'generation': gen}
        candidates.sort(key=lambda item: item[0], reverse=True)
        old_mean = mean.copy()
        mean = sum((recomb[i] * candidates[i][1] for i in range(mu)))
        centered = np.stack([candidates[i][1] - old_mean for i in range(mu)])
        cov = sum((recomb[i] * np.outer(centered[i], centered[i]) for i in range(mu))) + 1e-06 * np.eye(dim)
        history.append({'generation': gen, 'best': candidates[0][0], 'weights': candidates[0][2].tolist()})
    return {'best': best, 'history': history}

def run_score_file_search(score_paths: dict[str, str | Path], budget: int, output_dir: str | Path, seed: int=0, generations: int=12, popsize: int=20):
    scores = load_score_jsons(score_paths, normalize=True)
    names = list(scores)

    def objective(weights):
        selected = select_topk(fuse_scores(scores, weights), budget)
        return float(np.mean([scores[n][case] for n in names for case in selected]))
    result = cma_es(objective, len(names), seed=seed, generations=generations, popsize=popsize)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = select_topk(fuse_scores(scores, result['best']['weights']), budget)
    write_json(out / 'search.json', result)
    write_lines(out / f'selection_{budget}.txt', selected)
    return result
