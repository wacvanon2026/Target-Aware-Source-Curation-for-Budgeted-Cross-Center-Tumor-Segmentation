#!/usr/bin/env python3
"""Paper-style 8D TAVO CMA search for MAMAMIA nnUNet proxy fitness.

This runner mirrors the modern BraTS TAVO mechanics:
  * 8 criteria: RDS, LESS, ORIENT, CRAIG, GradMatch, KMeans, KCenter, Diversity
  * true gradient criteria for LESS/GradMatch/CRAIG, not embedding proxies
  * tie-aware rank normalization before score fusion
  * clip-and-renormalize simplex mapping
  * target-train cases included in every candidate training set
  * target-val cases used for proxy fitness
  * full covariance CMA logic implemented locally, plus corner and refinement evals

The default search hyperparameters intentionally match the BraTS 8D scripts.
For smoke tests, use --score-only.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - score-only mode can run without torch loaded.
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


METHODS = [
    "rds",
    "less",
    "orient",
    "craig",
    "gradmatch",
    "kmeans",
    "kcenter",
    "diversity",
]

TARGET_ROOTS = {
    "NACT": "mamamia_clean",
    "ISPY1": "mamamia_ispy1",
    "DUKE": "mamamia_duke",
    "ISPY2": "mamamia_ispy2",
}

TARGET_FULL_SOURCE_DATASETS = {
    "NACT": "Dataset1303_MAMAMIA_NACT_LODO_SEED42_TAVO_TARGET_FULL_SOURCE_2d_3ch",
    "ISPY1": "Dataset1313_MAMAMIA_ISPY1_LODO_SEED42_TAVO_TARGET_FULL_SOURCE_2d_3ch",
    "DUKE": "Dataset1323_MAMAMIA_DUKE_LODO_SEED42_TAVO_TARGET_FULL_SOURCE_2d_3ch",
    "ISPY2": "Dataset1333_MAMAMIA_ISPY2_LODO_SEED42_TAVO_TARGET_FULL_SOURCE_2d_3ch",
}

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_embeddings_jsonl(path: Path) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    vectors: list[list[float]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec["image"])
            vectors.append(rec["embedding"])
    if not vectors:
        raise RuntimeError(f"No embeddings found in {path}")
    return ids, np.asarray(vectors, dtype=np.float32)


def load_gradients(path: Path) -> tuple[list[str], np.ndarray]:
    ids = json.loads((path / "pool_ids.json").read_text())
    gradients = np.load(path / "pool_gradients.npy")
    if len(ids) != len(gradients):
        raise RuntimeError(f"Pool gradient ID/data mismatch in {path}: {len(ids)} vs {len(gradients)}")
    return ids, np.asarray(gradients, dtype=np.float32)


def load_query_gradients(path: Path) -> tuple[list[str], np.ndarray]:
    ids = json.loads((path / "query_ids.json").read_text())
    gradients = np.load(path / "query_gradients.npy")
    if len(ids) != len(gradients):
        raise RuntimeError(f"Query gradient ID/data mismatch in {path}: {len(ids)} vs {len(gradients)}")
    return ids, np.asarray(gradients, dtype=np.float32)


def rank_normalize(score_dict: dict[str, float]) -> dict[str, float]:
    items = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
    n = len(items)
    if n < 2:
        raise ValueError("Need at least two scores for rank normalization")

    value_to_indices: dict[float, list[int]] = defaultdict(list)
    for idx, (_, value) in enumerate(items):
        value_to_indices[float(value)].append(idx)

    ranks: dict[str, float] = {}
    for indices in value_to_indices.values():
        avg_rank = sum(indices) / len(indices)
        value = 1.0 - avg_rank / (n - 1)
        for idx in indices:
            ranks[items[idx][0]] = float(value)
    return ranks


def project_simplex(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    w = np.clip(z, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.ones_like(w) / len(w)
    return w / total


def rank_weights(mu: int) -> np.ndarray:
    weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
    return weights / weights.sum()


def update_eigensystem(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cov_sym = 0.5 * (cov + cov.T)
    eigvals, basis = np.linalg.eigh(cov_sym)
    eigvals = np.maximum(eigvals, 1e-12)
    scale = np.sqrt(eigvals)
    invsqrt = basis @ np.diag(1.0 / scale) @ basis.T
    return basis, scale, invsqrt


def aggregate(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    raise ValueError(f"Unknown aggregation mode: {mode}")


def seed_training(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def combine_scores(
    score_dicts: dict[str, dict[str, float]],
    weights_vec: np.ndarray,
) -> dict[str, float]:
    weights_vec = project_simplex(weights_vec)
    weights = {name: float(weights_vec[i]) for i, name in enumerate(METHODS)}
    keys = list(next(iter(score_dicts.values())).keys())
    combined = {}
    for key in keys:
        combined[key] = float(sum(weights[name] * score_dicts[name][key] for name in METHODS))
    return combined


def select_top_b(
    score_dicts: dict[str, dict[str, float]],
    weights_vec: np.ndarray,
    budget: int,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    weights_vec = project_simplex(weights_vec)
    weights = {name: float(weights_vec[i]) for i, name in enumerate(METHODS)}
    combined = combine_scores(score_dicts, weights_vec)
    ranked = sorted(combined.items(), key=lambda kv: (-kv[1], kv[0]))
    return [case for case, _ in ranked[:budget]], weights, combined


def import_target_code(target_root: Path):
    clean_root = target_root if target_root.name == "mamamia_clean" else target_root.parent / "mamamia_clean"
    sys.path.insert(0, str(clean_root))
    from selection import SELECTION_METHODS, get_selection_method  # type: ignore
    from meta.fitness.nnunet_proxy import NNUNetProxyFitnessEvaluator  # type: ignore

    return SELECTION_METHODS, get_selection_method, NNUNetProxyFitnessEvaluator


class CachedMamamiaSliceDataset(Dataset):  # type: ignore[misc]
    """MamamiaSliceDataset-compatible dataset with cached per-case slice indexes."""

    _index_cache: dict[tuple[tuple[str, ...], str, int, bool], tuple[Path, list[int], list[bool]]] = {}

    def __init__(
        self,
        data_dirs: list[Path],
        case_ids: list[str],
        augment: bool = False,
        min_tumor_pixels: int = 100,
        crop_size: int = 512,
        include_all: bool = False,
    ):
        self.data_dirs = [Path(d) for d in data_dirs]
        self.augment = augment
        self.crop_size = crop_size
        self.slices: list[tuple[str, int]] = []
        self.is_foreground: list[bool] = []
        self.case_to_dir: dict[str, Path] = {}
        data_key = tuple(str(d) for d in self.data_dirs)

        for case_id in case_ids:
            cache_key = (data_key, case_id, int(min_tumor_pixels), bool(include_all))
            cached = self._index_cache.get(cache_key)
            if cached is None:
                found = self._index_case(case_id, min_tumor_pixels, include_all)
                if found is None:
                    print(f"Warning: Missing files for {case_id}")
                    continue
                self._index_cache[cache_key] = found
                cached = found

            case_dir, slice_indices, foreground_flags = cached
            self.case_to_dir[case_id] = case_dir
            self.slices.extend((case_id, slice_idx) for slice_idx in slice_indices)
            self.is_foreground.extend(foreground_flags)

        n_fg = sum(self.is_foreground)
        n_bg = len(self.slices) - n_fg
        print(
            f"Loaded {len(self.slices)} slices from {len(case_ids)} cases "
            f"({n_fg} tumor, {n_bg} non-tumor; cached index cases={len(self._index_cache)})"
        )

    def _find_case_dir(self, case_id: str) -> Path | None:
        for data_dir in self.data_dirs:
            if (data_dir / f"{case_id}.npy").exists() and (data_dir / f"{case_id}_seg.npy").exists():
                return data_dir
            if (data_dir / f"{case_id}.npz").exists():
                return data_dir
        return None

    def _index_case(
        self,
        case_id: str,
        min_tumor_pixels: int,
        include_all: bool,
    ) -> tuple[Path, list[int], list[bool]] | None:
        case_dirs: list[Path] = []
        for data_dir in self.data_dirs:
            if (data_dir / f"{case_id}.npy").exists() and (data_dir / f"{case_id}_seg.npy").exists():
                case_dirs.append(data_dir)
            elif (data_dir / f"{case_id}.npz").exists():
                case_dirs.append(data_dir)
        if not case_dirs:
            return None
        last_error: Exception | None = None
        for case_dir in case_dirs:
            try:
                _, seg = self._load_case(case_dir, case_id)
                break
            except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                last_error = exc
                print(f"Warning: Could not load {case_id} from {case_dir}: {type(exc).__name__}: {exc}")
        else:
            if last_error is not None:
                print(f"Warning: Skipping {case_id}; all proxy copies failed to load.")
            return None
        slice_indices: list[int] = []
        foreground_flags: list[bool] = []
        for slice_idx in range(seg.shape[0]):
            tumor_pixels = (seg[slice_idx] > 0).sum()
            has_tumor = bool(tumor_pixels >= min_tumor_pixels)
            if include_all or has_tumor:
                slice_indices.append(slice_idx)
                foreground_flags.append(has_tumor)
        return case_dir, slice_indices, foreground_flags

    def _load_case(self, data_dir: Path, case_id: str) -> tuple[np.ndarray, np.ndarray]:
        data_file = data_dir / f"{case_id}.npy"
        seg_file = data_dir / f"{case_id}_seg.npy"
        if data_file.exists() and seg_file.exists():
            data = np.load(data_file)
            seg = np.load(seg_file)
        else:
            with np.load(data_dir / f"{case_id}.npz") as npz:
                data = npz["data"]
                seg = npz["seg"]
        if seg.ndim == 4:
            seg = seg[0]
        return data, seg

    def __len__(self) -> int:
        return len(self.slices)

    def _normalize_and_resize(self, img: np.ndarray, mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if F is None:
            raise RuntimeError("torch is required for proxy fitness evaluation")
        img_t = torch.from_numpy(img).float().unsqueeze(0)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
        img_t = F.interpolate(img_t, size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False)
        mask_t = F.interpolate(mask_t, size=(self.crop_size, self.crop_size), mode="nearest")
        img_t = img_t.squeeze(0)
        mask_t = (mask_t.squeeze(0).squeeze(0) > 0).long()
        for channel_idx in range(img_t.shape[0]):
            channel = img_t[channel_idx]
            channel_min = channel.min()
            channel_max = channel.max()
            if channel_max > channel_min:
                channel = (channel - channel_min) / (channel_max - channel_min)
            img_t[channel_idx] = (channel - MEAN[channel_idx]) / STD[channel_idx]
        return img_t, mask_t

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return img, mask
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[1])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[0])
        return img, mask

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        case_id, slice_idx = self.slices[idx]
        data_dir = self.case_to_dir[case_id]
        data, seg = self._load_case(data_dir, case_id)
        img_slice = data[:, slice_idx, :, :]
        mask_slice = seg[slice_idx, :, :]
        img_t, mask_t = self._normalize_and_resize(img_slice, mask_slice)
        img_t, mask_t = self._augment(img_t, mask_t)
        return {
            "image": img_t,
            "mask": mask_t,
            "case_id": case_id,
            "slice_idx": slice_idx,
        }


def install_cached_proxy_dataset(evaluator_cls: Any) -> None:
    module = sys.modules.get(evaluator_cls.__module__)
    if module is None:
        raise RuntimeError(f"Cannot find evaluator module {evaluator_cls.__module__}")
    module.MamamiaSliceDataset = CachedMamamiaSliceDataset


def load_target_arrays(target_root: Path) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    emb_dir = target_root / "data" / "embeddings"
    grad_dir = target_root / "data" / "gradients"

    emb_ids, pool_emb = load_embeddings_jsonl(emb_dir / "pool_embeddings.jsonl")
    query_emb_ids, query_emb = load_embeddings_jsonl(emb_dir / "query_embeddings.jsonl")
    grad_ids, pool_grad = load_gradients(grad_dir)
    query_grad_ids, query_grad = load_query_gradients(grad_dir)

    emb_map = {case_id: pool_emb[i] for i, case_id in enumerate(emb_ids)}
    grad_map = {case_id: pool_grad[i] for i, case_id in enumerate(grad_ids)}
    common_ids = [case_id for case_id in emb_ids if case_id in grad_map]
    if len(common_ids) < 2:
        raise RuntimeError("Embedding/gradient pool intersection is empty")

    if len(common_ids) != len(emb_ids) or len(common_ids) != len(grad_ids):
        print(
            "WARNING: aligning pool to embedding/gradient intersection: "
            f"embeddings={len(emb_ids)}, gradients={len(grad_ids)}, common={len(common_ids)}"
        )
    if set(query_emb_ids) != set(query_grad_ids):
        print(
            "WARNING: query embedding/gradient ID sets differ; "
            f"embedding_query={len(query_emb_ids)}, gradient_query={len(query_grad_ids)}"
        )

    aligned_emb = np.asarray([emb_map[case_id] for case_id in common_ids], dtype=np.float32)
    aligned_grad = np.asarray([grad_map[case_id] for case_id in common_ids], dtype=np.float32)
    return common_ids, aligned_emb, query_emb, aligned_grad, query_grad


def compute_score_dicts(target_root: Path, budget: int, seed: int) -> dict[str, dict[str, float]]:
    selection_methods, get_selection_method, _ = import_target_code(target_root)
    missing = [method for method in METHODS if method not in selection_methods]
    if missing:
        raise RuntimeError(f"Target root is missing required methods: {missing}")

    pool_ids, pool_emb, query_emb, pool_grad, query_grad = load_target_arrays(target_root)
    raw_scores: dict[str, dict[str, float]] = {}

    for method_name in METHODS:
        print(f"Computing raw scores for {method_name}...")
        method_cls = get_selection_method(method_name)
        if method_name == "less":
            method = method_cls(seed=seed, rank=50)
        elif method_name == "gradmatch":
            method = method_cls(seed=seed, omp=True)
        else:
            method = method_cls(seed=seed)

        result = method.select(
            pool_ids=pool_ids,
            budget=budget,
            embeddings=pool_emb,
            query_embeddings=query_emb,
            gradients=pool_grad,
            query_gradients=query_grad,
        )
        raw_scores[method_name] = result.scores

    key_sets = [set(scores) for scores in raw_scores.values()]
    base = key_sets[0]
    for method_name, keys in zip(METHODS, key_sets):
        if keys != base:
            raise RuntimeError(f"Score key mismatch for {method_name}")

    return {method: rank_normalize(scores) for method, scores in raw_scores.items()}


def build_fitness_evaluator(args: argparse.Namespace, target_root: Path):
    _, _, evaluator_cls = import_target_code(target_root)
    install_cached_proxy_dataset(evaluator_cls)
    nnunet_preprocessed = (
        args.project_root
        / "externals"
        / "MAMA-MIA"
        / "nnUNet"
        / "nnunetv2"
        / "nnUNet_preprocessed"
    )
    data_dirs = []
    if args.dataset_name:
        dataset_names = args.dataset_name
    else:
        dataset_names = [TARGET_FULL_SOURCE_DATASETS[args.target]]
        dataset_names.extend(
            name
            for target, name in TARGET_FULL_SOURCE_DATASETS.items()
            if target != args.target
        )
    for dataset_name in dataset_names:
        path = nnunet_preprocessed / dataset_name / "nnUNetPlans_2d"
        if path.exists():
            data_dirs.append(path)
            print(f"Found proxy data: {dataset_name}")
    if not data_dirs:
        raise RuntimeError(f"No proxy data dirs found under {nnunet_preprocessed}")

    target_val = read_lines(args.split_root / args.target / "target_val.txt")
    target_train = read_lines(args.split_root / args.target / "target_train.txt")
    print(f"Target train cases: {len(target_train)}")
    print(f"Target val cases: {len(target_val)}")

    evaluator = evaluator_cls(
        data_dirs=data_dirs,
        val_cases=target_val,
        batch_size=args.batch_size,
        lr=args.lr,
        crop_size=args.crop_size,
        best_k=args.best_k,
    )
    return evaluator, target_train


def evaluate_candidate(
    score_dicts: dict[str, dict[str, float]],
    evaluator: Any,
    target_train: list[str],
    budget: int,
    weights_vec: np.ndarray,
    n_steps: int,
    seeds: list[int],
) -> dict[str, Any]:
    selected, weights, combined = select_top_b(score_dicts, weights_vec, budget)
    train_cases = list(dict.fromkeys(selected + target_train))
    per_seed = {}
    elapsed_total = 0.0
    for seed in seeds:
        seed_training(seed)
        t0 = time.time()
        result = evaluator.evaluate(train_cases, n_steps)
        elapsed = time.time() - t0
        elapsed_total += elapsed
        per_seed[str(seed)] = {
            "fitness": float(result["fitness"]),
            "final_val_dice": result.get("final_val_dice"),
            "val_dices": result.get("val_dices", []),
            "n_train_slices": result.get("n_train_slices"),
            "elapsed_seconds": elapsed,
        }
    fitness = aggregate([v["fitness"] for v in per_seed.values()], "median")
    return {
        "weights": weights,
        "n_steps": int(n_steps),
        "fitness": float(fitness),
        "per_seed": per_seed,
        "selected_cases": selected,
        "selected_scores": {case: combined[case] for case in selected},
        "elapsed_seconds": elapsed_total,
    }


def save_json(path: Path, data: Any) -> None:
    def default(obj: Any) -> str:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, default=default)
    tmp.replace(path)


def run_cma(args: argparse.Namespace) -> dict[str, Any]:
    target_root = args.project_root / TARGET_ROOTS[args.target]
    score_dicts = compute_score_dicts(target_root, args.budget, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    score_summary = {
        "target": args.target,
        "budget": args.budget,
        "methods": METHODS,
        "n_sources": len(next(iter(score_dicts.values()))),
    }
    save_json(args.output_dir / "score_summary.json", score_summary)
    if args.score_only:
        print(json.dumps(score_summary, indent=2))
        return {"score_summary": score_summary}

    evaluator, target_train = build_fitness_evaluator(args, target_root)

    dim = len(METHODS)
    lam = args.popsize
    mu = args.mu
    n_gen = args.generations
    w_rank = rank_weights(mu)
    mueff = 1.0 / np.sum(w_rank**2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))

    mean = np.ones(dim) / dim
    sigma = args.sigma0
    cov = np.eye(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    basis, scale, invsqrt = update_eigensystem(cov)
    rng = np.random.default_rng(args.seed)

    seeds_eval = [int(s) for s in args.seeds_eval.replace(",", " ").split()]
    seeds_refine = [int(s) for s in args.seeds_refine.replace(",", " ").split()]
    all_evals: list[dict[str, Any]] = []
    generations: list[dict[str, Any]] = []
    best_so_far: dict[str, Any] | None = None
    partial_path = args.output_dir / "stageA2_cma_partial.json"

    def record_eval(record: dict[str, Any], gen: int, z: np.ndarray) -> None:
        nonlocal best_so_far
        record["gen"] = gen
        record["z"] = np.asarray(z, dtype=np.float64).tolist()
        all_evals.append(record)
        if best_so_far is None or record["fitness"] > best_so_far["fitness"]:
            best_so_far = record

    print("Evaluating 8D corners plus uniform...")
    corners = []
    for i in range(dim):
        corner = np.zeros(dim)
        corner[i] = 1.0
        corners.append(corner)
    corners.append(np.ones(dim) / dim)

    for corner in corners:
        result = evaluate_candidate(score_dicts, evaluator, target_train, args.budget, corner, args.n_steps, seeds_eval)
        print(f"[Corner] fitness={result['fitness']:.4f} weights={result['weights']}")
        record_eval(result, -1, corner)
    save_json(partial_path, {"config": vars(args), "methods": METHODS, "best_so_far": best_so_far, "generations": generations})

    for gen in range(n_gen):
        print(f"Generation {gen}: sigma={sigma:.4f}, mean_proj={project_simplex(mean).tolist()}")
        arz = rng.normal(size=(lam, dim))
        ary = (arz * scale) @ basis.T
        arx = mean[None, :] + sigma * ary

        evals = []
        for idx in range(lam):
            z = arx[idx]
            result = evaluate_candidate(score_dicts, evaluator, target_train, args.budget, z, args.n_steps, seeds_eval)
            print(f"[Gen {gen} | Cand {idx}] fitness={result['fitness']:.4f} weights={result['weights']}")
            record_eval(result, gen, z)
            evals.append(result)

        evals.sort(key=lambda r: r["fitness"], reverse=True)
        parents = evals[:mu]
        z_parents = np.array([p["z"] for p in parents], dtype=np.float64)
        mean_old = mean.copy()
        mean = np.sum(z_parents * w_rank[:, None], axis=0)

        y_w = (mean - mean_old) / max(sigma, 1e-12)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrt @ y_w)
        ps_norm = np.linalg.norm(ps)
        hsig = 1.0 if ps_norm / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w

        y = (z_parents - mean_old[None, :]) / max(sigma, 1e-12)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            yi = y[i][:, None]
            rank_mu += w_rank[i] * (yi @ yi.T)
        cov = (1 - c1 - cmu) * cov + c1 * (pc[:, None] @ pc[None, :] + (1 - hsig) * cc * (2 - cc) * cov) + cmu * rank_mu
        sigma = max(args.sigma_min, sigma * math.exp((cs / damps) * (ps_norm / chi_n - 1.0)))
        basis, scale, invsqrt = update_eigensystem(cov)
        eigvals = np.linalg.eigvalsh(cov)

        gen_summary = {
            "gen": gen,
            "sigma": float(sigma),
            "mean_z": mean.tolist(),
            "mean_proj": project_simplex(mean).tolist(),
            "eigvals": eigvals.tolist(),
            "best_fitness_gen": float(parents[0]["fitness"]),
            "best_fitness_global": float(best_so_far["fitness"] if best_so_far else float("-inf")),
        }
        generations.append(gen_summary)
        save_json(
            partial_path,
            {"config": vars(args), "methods": METHODS, "best_so_far": best_so_far, "generations": generations, "n_evals": len(all_evals)},
        )

    print("Refining top candidates...")
    refine_records = []
    for record in sorted(all_evals, key=lambda r: r["fitness"], reverse=True)[: args.refine_topk]:
        z = np.array([record["weights"][m] for m in METHODS], dtype=np.float64)
        refined = evaluate_candidate(score_dicts, evaluator, target_train, args.budget, z, args.refine_steps, seeds_refine)
        print(f"[Refine] fitness={refined['fitness']:.4f} weights={refined['weights']}")
        refine_records.append(refined)
    refine_records.sort(key=lambda r: r["fitness"], reverse=True)
    best = refine_records[0] if refine_records else best_so_far

    result = {
        "config": vars(args),
        "methods": METHODS,
        "score_summary": score_summary,
        "generations": generations,
        "best_so_far": best_so_far,
        "top3": refine_records[:3],
        "refine_records": refine_records,
        "best": best,
    }
    save_json(args.output_dir / "stageA2_cma.json", result)
    if best is not None:
        save_json(
            args.output_dir / f"selection_{args.budget}_median_val_{args.seed}.json",
            {
                "selected": best["selected_cases"],
                "selected_scores": best["selected_scores"],
                "method": "tavo_8d_cmaes_median_val",
                "budget": args.budget,
                "weights": best["weights"],
                "methods": METHODS,
                "fitness": best["fitness"],
            },
        )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=sorted(TARGET_ROOTS))
    parser.add_argument("--budget", type=int, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--split-root", type=Path, default=Path(__file__).resolve().parents[2] / "splits" / "mamamia_lodo_seed42")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset-name",
        nargs="+",
        default=None,
        help="Optional override for proxy dataset(s). Defaults to the target-specific TARGET_FULL_SOURCE dataset.",
    )
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--popsize", type=int, default=20)
    parser.add_argument("--mu", type=int, default=10)
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--refine-steps", type=int, default=2000)
    parser.add_argument("--refine-topk", type=int, default=8)
    parser.add_argument("--sigma0", type=float, default=0.30)
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--seeds-eval", default="0")
    parser.add_argument("--seeds-refine", default="0,1,2")
    parser.add_argument("--best-k", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    args.target = args.target.upper().replace("-", "")
    args.project_root = args.project_root.resolve()
    args.split_root = args.split_root.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.mu > args.popsize:
        raise SystemExit("--mu cannot exceed --popsize")
    return args


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and args.score_only:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("MAMAMIA TAVO 8D CMA-ES")
    print("=" * 72)
    print(f"Target: {args.target}")
    print(f"Budget: {args.budget}")
    print(f"Methods: {METHODS}")
    print(f"Output: {args.output_dir}")
    print(f"Score only: {args.score_only}")
    run_cma(args)


if __name__ == "__main__":
    main()
