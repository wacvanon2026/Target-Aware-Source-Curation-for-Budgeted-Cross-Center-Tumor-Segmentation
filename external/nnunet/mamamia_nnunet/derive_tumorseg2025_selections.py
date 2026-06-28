#!/usr/bin/env python3
"""Derive MAMAMIA selections with the Tumor Seg 2025 ranking rules.

This script intentionally mirrors the standalone baseline method semantics in
project2/EfficentVitCopy/scripts/run_*.py:

- RDS ranks by maximum cosine similarity to any target/query embedding.
- LESS ranks by beta-weighted gradient cosine similarity to target gradients.
- ORIENT greedily maximizes target-query facility-location MI on ORIENT-style
  case gradient vectors.
- GradMatch uses OMP-style positive matching of the mean target case gradient
  vector.
- CRAIG greedily maximizes source case-gradient facility-location coverage.
- KMeans ranks cluster representatives by cluster size.
- KCenter uses seeded feature-space farthest-first ranking.
- Diversity uses seeded cumulative feature-space distance ranking. The original
  script can reselect a previous case; MAMAMIA keeps the same cumulative
  objective but masks already selected cases so nnUNet receives unique volumes.

When a target is missing raw gradients, gradient-only methods are skipped unless
explicitly allowed to fall back. The materializer may still fall back to
pre-existing legacy/proxy artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_similarity

from core import REPO_ROOT, project_root

try:
    from submodlib.functions.facilityLocation import FacilityLocationFunction
    from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
except ImportError:  # pragma: no cover - fallback is for lightweight validation envs.
    FacilityLocationFunction = None
    FacilityLocationMutualInformationFunction = None


PROJECT_ROOT = project_root()
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "tumorseg2025_selections"

TARGET_INPUTS = {
    "NACT": {
        "embedding_root": PROJECT_ROOT / "dataset_mamamia" / "embeddings",
        "gradient_root": PROJECT_ROOT / "mamamia_clean" / "data" / "gradients",
    },
    "ISPY1": {
        "embedding_root": PROJECT_ROOT / "mamamia_ispy1" / "data" / "embeddings",
        "gradient_root": PROJECT_ROOT / "mamamia_ispy1" / "data" / "gradients",
    },
    "DUKE": {
        "embedding_root": PROJECT_ROOT / "mamamia_duke" / "data" / "embeddings",
        "gradient_root": PROJECT_ROOT / "mamamia_duke" / "data" / "gradients",
    },
    "ISPY2": {
        "embedding_root": PROJECT_ROOT / "mamamia_ispy2" / "data" / "embeddings",
        "gradient_root": PROJECT_ROOT / "mamamia_ispy2" / "data" / "gradients",
    },
}

METHODS = ("rds", "less", "orient", "gradmatch", "craig", "kmeans", "kcenter", "diversity")
GRADIENT_METHODS = {"less", "orient", "gradmatch", "craig"}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def load_embeddings(root: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    pool_path = root / "pool_embeddings.jsonl"
    query_path = root / "query_embeddings.jsonl"
    if not pool_path.exists() or not query_path.exists():
        raise FileNotFoundError(f"Missing embedding files under {root}")

    def read_jsonl(path: Path) -> tuple[list[str], np.ndarray]:
        ids: list[str] = []
        vecs: list[list[float]] = []
        with path.open() as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                ids.append(str(record["image"]))
                vecs.append(record["embedding"])
        return ids, np.asarray(vecs, dtype=np.float64)

    pool_ids, pool_vecs = read_jsonl(pool_path)
    query_ids, query_vecs = read_jsonl(query_path)
    return pool_ids, pool_vecs, query_ids, query_vecs


def load_gradients(root: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray] | None:
    required = [root / "pool_ids.json", root / "pool_gradients.npy", root / "query_ids.json", root / "query_gradients.npy"]
    if not all(path.exists() for path in required):
        return None
    pool_ids = [str(x) for x in json.loads((root / "pool_ids.json").read_text())]
    query_ids = [str(x) for x in json.loads((root / "query_ids.json").read_text())]
    pool_grads = np.load(root / "pool_gradients.npy").astype(np.float64)
    query_grads = np.load(root / "query_gradients.npy").astype(np.float64)
    if len(pool_ids) != pool_grads.shape[0] or len(query_ids) != query_grads.shape[0]:
        raise ValueError(f"Gradient id/vector count mismatch under {root}")
    return pool_ids, pool_grads, query_ids, query_grads


def order_to_rank_scores(ids: list[str], order: list[int]) -> dict[str, float]:
    scores = {case_id: 0.0 for case_id in ids}
    denom = max(len(order) - 1, 1)
    for rank, idx in enumerate(order):
        scores[ids[idx]] = float(1.0 - rank / denom)
    return scores


def order_to_gain_scores(ids: list[str], order: list[int], gains: list[float]) -> dict[str, float]:
    scores = {case_id: 0.0 for case_id in ids}
    for idx, gain in zip(order, gains):
        scores[ids[idx]] = float(gain)
    return scores


def write_selection(
    output_root: Path,
    target: str,
    method: str,
    ids: list[str],
    order: list[int],
    scores: dict[str, float],
    budget: int,
    metadata: dict[str, Any],
) -> Path:
    selected = [ids[idx] for idx in order[:budget]]
    out_dir = output_root / target
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method}_{budget}.json"
    payload = {
        "method": method,
        "lineage": "tumorseg2025",
        "budget": budget,
        "selected": selected,
        "selected_cases": selected,
        "scores": scores,
        "metadata": metadata,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    return out_path


def rds_order(pool_ids: list[str], pool_vecs: np.ndarray, query_vecs: np.ndarray) -> tuple[list[int], dict[str, float]]:
    src = l2_normalize(pool_vecs)
    tgt = l2_normalize(query_vecs)
    scores_arr = (src @ tgt.T).max(axis=1)
    order = np.argsort(scores_arr)[::-1].tolist()
    scores = {pool_ids[i]: float(scores_arr[i]) for i in range(len(pool_ids))}
    return order, scores


def less_order(pool_ids: list[str], pool_grads: np.ndarray, query_grads: np.ndarray, beta: float = 20.0) -> tuple[list[int], dict[str, float]]:
    src = l2_normalize(pool_grads)
    tgt = l2_normalize(query_grads)
    scores_arr = np.zeros(src.shape[0], dtype=np.float64)
    batch = 256
    for start in range(0, src.shape[0], batch):
        end = min(start + batch, src.shape[0])
        sim = src[start:end] @ tgt.T
        weights = np.exp(beta * sim)
        weights /= weights.sum(axis=1, keepdims=True) + 1e-12
        scores_arr[start:end] = (weights * sim).sum(axis=1)
    order = np.argsort(scores_arr)[::-1].tolist()
    scores = {pool_ids[i]: float(scores_arr[i]) for i in range(len(pool_ids))}
    return order, scores


def orient_order(pool_ids: list[str], pool_vecs: np.ndarray, query_vecs: np.ndarray, budget: int, eta: float = 1.0) -> tuple[list[int], dict[str, float], str]:
    src = l2_normalize(pool_vecs)
    tgt = l2_normalize(query_vecs)

    if FacilityLocationMutualInformationFunction is not None:
        data_sijs = np.maximum(cosine_similarity(src, src), 0.0).astype(np.float64)
        query_sijs = np.maximum(cosine_similarity(src, tgt), 0.0).astype(np.float64)
        obj = FacilityLocationMutualInformationFunction(
            n=src.shape[0],
            num_queries=tgt.shape[0],
            data_sijs=data_sijs,
            query_sijs=query_sijs,
            magnificationEta=eta,
        )
        # Mirrors Tumor Seg 2025: run LazyGreedy on the FLMI objective and keep
        # the greedy order. The requested max rank is enough for the K50/K150
        # and K250 table rows while avoiding unnecessary full-pool work.
        result = obj.maximize(
            budget=min(budget, src.shape[0]),
            optimizer="LazyGreedy",
            stopIfNegativeGain=False,
            show_progress=True,
        )
        selected = [int(elem[0] if isinstance(elem, tuple) else elem) for elem in result]
        return selected, order_to_rank_scores(pool_ids, selected), "submodlib_facility_location_mi_lazy_greedy"

    q = np.maximum(src @ tgt.T, 0.0)
    current = np.zeros(q.shape[1], dtype=np.float64)
    selected: list[int] = []
    unused = np.ones(q.shape[0], dtype=bool)
    for _ in range(min(budget, q.shape[0])):
        gains = np.maximum(q, current).sum(axis=1) - current.sum()
        gains[~unused] = -np.inf
        idx = int(np.argmax(gains))
        if not np.isfinite(gains[idx]):
            break
        selected.append(idx)
        unused[idx] = False
        current = np.maximum(current, q[idx])
    return selected, order_to_rank_scores(pool_ids, selected), "fallback_query_facility_location_greedy"


def gradmatch_order(pool_ids: list[str], pool_vecs: np.ndarray, query_vecs: np.ndarray, budget: int) -> tuple[list[int], dict[str, float]]:
    src = l2_normalize(pool_vecs)
    target = query_vecs.mean(axis=0).astype(np.float64)
    target = target / (np.linalg.norm(target) + 1e-12)
    selected: list[int] = []
    gains: list[float] = []
    residual = target.copy()
    prev_norm = float(np.dot(residual, residual))
    for step in range(min(budget, src.shape[0])):
        corr = src @ residual
        if selected:
            corr[np.asarray(selected)] = -np.inf
        idx = int(np.argmax(corr))
        selected.append(idx)
        a = src[selected]
        reg = LinearRegression(fit_intercept=False, positive=True)
        reg.fit(a.T, target)
        approx = a.T @ reg.coef_
        residual = target - approx
        new_norm = float(np.dot(residual, residual))
        gains.append(prev_norm - new_norm)
        prev_norm = new_norm
        if step % 50 == 0:
            pass
    return selected, order_to_gain_scores(pool_ids, selected, gains)


def craig_order(pool_ids: list[str], pool_vecs: np.ndarray, budget: int) -> tuple[list[int], dict[str, float], str]:
    src = l2_normalize(pool_vecs)
    sim = np.maximum(cosine_similarity(src), 0.0)

    if FacilityLocationFunction is not None:
        obj = FacilityLocationFunction(
            n=src.shape[0],
            mode="dense",
            sijs=sim.astype(np.float64),
            separate_rep=False,
        )
        result = obj.maximize(
            budget=min(budget, src.shape[0]),
            optimizer="LazyGreedy",
            stopIfNegativeGain=False,
            show_progress=True,
        )
        selected: list[int] = []
        gains: list[float] = []
        for elem in result:
            if isinstance(elem, tuple):
                idx, gain = elem
            else:
                idx, gain = elem, 0.0
            selected.append(int(idx))
            gains.append(float(gain))
        return selected, order_to_gain_scores(pool_ids, selected, gains), "submodlib_facility_location_lazy_greedy"

    current = np.zeros(sim.shape[0], dtype=np.float64)
    selected: list[int] = []
    unused = np.ones(sim.shape[0], dtype=bool)
    for _ in range(min(budget, sim.shape[0])):
        gains = np.maximum(sim, current[:, None]).sum(axis=0) - current.sum()
        gains[~unused] = -np.inf
        idx = int(np.argmax(gains))
        selected.append(idx)
        unused[idx] = False
        current = np.maximum(current, sim[:, idx])
    return selected, order_to_rank_scores(pool_ids, selected), "fallback_facility_location_greedy"


def kmeans_order(pool_ids: list[str], pool_vecs: np.ndarray, max_rank: int, seed: int) -> tuple[list[int], dict[str, float]]:
    x = l2_normalize(pool_vecs)
    k = min(max_rank, x.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(x)
    centers = kmeans.cluster_centers_
    reps: list[int] = []
    sizes: list[int] = []
    for cluster in range(k):
        idxs = np.flatnonzero(labels == cluster)
        if idxs.size == 0:
            continue
        dists = np.linalg.norm(x[idxs] - centers[cluster], axis=1)
        reps.append(int(idxs[int(np.argmin(dists))]))
        sizes.append(int(idxs.size))
    reps_sorted = [idx for _, idx in sorted(zip(sizes, reps), reverse=True)]
    selected = set(reps_sorted)
    remaining = [idx for idx in range(x.shape[0]) if idx not in selected]
    order = reps_sorted + remaining
    return order, order_to_rank_scores(pool_ids, order)


def kcenter_order(pool_ids: list[str], pool_vecs: np.ndarray, max_rank: int, seed: int) -> tuple[list[int], dict[str, float]]:
    x = l2_normalize(pool_vecs)
    k = min(max_rank, x.shape[0])
    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(x.shape[0]))]
    min_dist = np.linalg.norm(x - x[selected[0]], axis=1)
    gains = [float(np.max(min_dist))]
    for _ in range(1, k):
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        gains.append(float(min_dist[idx]))
        new_dist = np.linalg.norm(x - x[idx], axis=1)
        min_dist = np.minimum(min_dist, new_dist)
    return selected, order_to_gain_scores(pool_ids, selected, gains)


def diversity_order(pool_ids: list[str], pool_vecs: np.ndarray, max_rank: int, seed: int) -> tuple[list[int], dict[str, float]]:
    x = l2_normalize(pool_vecs)
    k = min(max_rank, x.shape[0])
    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(x.shape[0]))]
    dist_sum = np.linalg.norm(x - x[selected[0]], axis=1)
    gains = [float(np.max(dist_sum))]
    for _ in range(1, k):
        dist_sum[np.asarray(selected)] = -np.inf
        idx = int(np.argmax(dist_sum))
        if not np.isfinite(dist_sum[idx]):
            break
        selected.append(idx)
        gains.append(float(dist_sum[idx]))
        dist_sum += np.linalg.norm(x - x[idx], axis=1)
    return selected, order_to_gain_scores(pool_ids, selected, gains)


def derive_target(args: argparse.Namespace, target: str) -> list[Path]:
    target = target.upper()
    inputs = TARGET_INPUTS[target]
    output_paths: list[Path] = []

    pool_ids, pool_vecs, query_ids, query_vecs = load_embeddings(inputs["embedding_root"])
    gradients = load_gradients(inputs["gradient_root"])

    print(f"{target}: embeddings pool={len(pool_ids)} query={len(query_ids)}")
    if gradients is None:
        missing_gradient_methods = sorted(methods & GRADIENT_METHODS)
        if missing_gradient_methods and not args.allow_missing_gradients:
            raise SystemExit(
                f"{target}: missing gradient arrays under {inputs['gradient_root']} for "
                f"{', '.join(missing_gradient_methods)}. Generate gradients or rerun with "
                "--allow-missing-gradients to skip these artifacts."
            )
        print(f"{target}: no gradient arrays found; skipping gradient-only methods")
    else:
        grad_pool_ids, pool_grads, grad_query_ids, query_grads = gradients
        print(f"{target}: gradients pool={len(grad_pool_ids)} query={len(grad_query_ids)}")

    methods = set(args.methods)
    if "rds" in methods:
        order, scores = rds_order(pool_ids, pool_vecs, query_vecs)
        output_paths.append(write_selection(args.output_root, target, "rds", pool_ids, order, scores, args.max_rank, {"rule": "max_source_to_query_cosine"}))

    if gradients is not None:
        grad_pool_ids, pool_grads, _, query_grads = gradients
        if "orient" in methods:
            order, scores, backend = orient_order(grad_pool_ids, pool_grads, query_grads, args.max_rank, eta=args.orient_eta)
            output_paths.append(
                write_selection(
                    args.output_root,
                    target,
                    "orient",
                    grad_pool_ids,
                    order,
                    scores,
                    args.max_rank,
                    {
                        "rule": "facility_location_mi_greedy",
                        "backend": backend,
                        "eta": args.orient_eta,
                        "source": "gradients",
                        "gradient_root": str(inputs["gradient_root"]),
                    },
                )
            )
        if "less" in methods:
            order, scores = less_order(grad_pool_ids, pool_grads, query_grads, beta=args.less_beta)
            output_paths.append(
                write_selection(
                    args.output_root,
                    target,
                    "less",
                    grad_pool_ids,
                    order,
                    scores,
                    args.max_rank,
                    {"rule": "beta_weighted_gradient_cosine", "beta": args.less_beta, "source": "gradients"},
                )
            )
        if "gradmatch" in methods:
            order, scores = gradmatch_order(grad_pool_ids, pool_grads, query_grads, args.max_rank)
            output_paths.append(
                write_selection(
                    args.output_root,
                    target,
                    "gradmatch",
                    grad_pool_ids,
                    order,
                    scores,
                    args.max_rank,
                    {"rule": "positive_omp_mean_target_gradient", "source": "gradients", "normalize": True},
                )
            )
        if "craig" in methods:
            order, scores, backend = craig_order(grad_pool_ids, pool_grads, args.max_rank)
            output_paths.append(
                write_selection(
                    args.output_root,
                    target,
                    "craig",
                    grad_pool_ids,
                    order,
                    scores,
                    args.max_rank,
                    {"rule": "source_gradient_facility_location", "backend": backend, "source": "gradients", "normalize": True},
                )
            )

    if gradients is None and args.allow_missing_gradients and "orient" in methods:
        order, scores, backend = orient_order(pool_ids, pool_vecs, query_vecs, args.max_rank, eta=args.orient_eta)
        output_paths.append(
            write_selection(
                args.output_root,
                target,
                "orient",
                pool_ids,
                order,
                scores,
                args.max_rank,
                {"rule": "facility_location_mi_greedy", "backend": backend, "eta": args.orient_eta, "source": "embeddings_fallback"},
            )
        )
    if gradients is None and args.allow_missing_gradients and "gradmatch" in methods:
        order, scores = gradmatch_order(pool_ids, pool_vecs, query_vecs, args.max_rank)
        output_paths.append(
            write_selection(
                args.output_root,
                target,
                "gradmatch",
                pool_ids,
                order,
                scores,
                args.max_rank,
                {"rule": "positive_omp_mean_target_embedding", "source": "embeddings_fallback", "normalize": True},
            )
        )
    if gradients is None and args.allow_missing_gradients and "craig" in methods:
        order, scores, backend = craig_order(pool_ids, pool_vecs, args.max_rank)
        output_paths.append(
            write_selection(
                args.output_root,
                target,
                "craig",
                pool_ids,
                order,
                scores,
                args.max_rank,
                {"rule": "source_embedding_facility_location", "backend": backend, "source": "embeddings_fallback", "normalize": True},
            )
        )

    if "kmeans" in methods:
        order, scores = kmeans_order(pool_ids, pool_vecs, args.max_rank, seed=args.seed)
        output_paths.append(write_selection(args.output_root, target, "kmeans", pool_ids, order, scores, args.max_rank, {"rule": "kmeans_representatives_sorted_by_cluster_size", "seed": args.seed}))
    if "kcenter" in methods:
        order, scores = kcenter_order(pool_ids, pool_vecs, args.max_rank, seed=args.seed)
        output_paths.append(write_selection(args.output_root, target, "kcenter", pool_ids, order, scores, args.max_rank, {"rule": "feature_farthest_first", "seed": args.seed, "score": "farthest_first_gain"}))
    if "diversity" in methods:
        order, scores = diversity_order(pool_ids, pool_vecs, args.max_rank, seed=args.seed)
        output_paths.append(write_selection(args.output_root, target, "diversity", pool_ids, order, scores, args.max_rank, {"rule": "cumulative_euclidean_distance", "seed": args.seed, "score": "cumulative_distance_gain"}))

    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=["NACT", "ISPY1", "DUKE", "ISPY2"])
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-rank", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--less-beta", type=float, default=20.0)
    parser.add_argument("--orient-eta", type=float, default=1.0)
    parser.add_argument(
        "--allow-missing-gradients",
        action="store_true",
        help="Skip or fallback for gradient methods when a target lacks gradient arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    all_paths: list[Path] = []
    for target in args.targets:
        all_paths.extend(derive_target(args, target))
    print(json.dumps([str(path) for path in all_paths], indent=2))


if __name__ == "__main__":
    main()
