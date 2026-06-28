#!/usr/bin/env python3
"""Compute foreground HD95 for completed MAMAMIA nnUNet predictions."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from core import expand_experiments, project_root


TARGET_ORDER = ("NACT", "ISPY1", "DUKE", "ISPY2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", default=["all"])
    parser.add_argument("--targets", nargs="+", default=list(TARGET_ORDER))
    parser.add_argument("--project-root", type=Path, default=project_root())
    parser.add_argument("--output-name", default="summary_hd95.json")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def normalize_target(target: str) -> str:
    normalized = target.upper().replace("-", "")
    if normalized not in TARGET_ORDER:
        raise ValueError(f"Unknown target: {target}")
    return normalized


def output_dir(project_root_: Path, target: str, experiment: str) -> Path:
    return (
        project_root_
        / "outputs"
        / f"tavo_mamamia_{target.lower()}_nnunet_{experiment}"
        / "repeat_01"
    )


def read_mask(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image) > 0
    spacing_xyz = image.GetSpacing()
    spacing_zyx = tuple(float(v) for v in reversed(spacing_xyz))
    return array, spacing_zyx


def surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    structure = generate_binary_structure(mask.ndim, 1)
    eroded = binary_erosion(mask, structure=structure, border_value=0)
    return mask ^ eroded


def image_diagonal(mask: np.ndarray, spacing: tuple[float, ...]) -> float:
    extents = [(size - 1) * sp for size, sp in zip(mask.shape, spacing)]
    return float(math.sqrt(sum(v * v for v in extents)))


def hd95(pred: np.ndarray, ref: np.ndarray, spacing: tuple[float, ...]) -> float:
    pred = pred.astype(bool, copy=False)
    ref = ref.astype(bool, copy=False)
    pred_has = bool(np.any(pred))
    ref_has = bool(np.any(ref))
    if not pred_has and not ref_has:
        return 0.0
    if pred_has != ref_has:
        return image_diagonal(pred if pred_has else ref, spacing)

    pred_surface = surface(pred)
    ref_surface = surface(ref)
    if not np.any(pred_surface) and not np.any(ref_surface):
        return 0.0
    if not np.any(pred_surface) or not np.any(ref_surface):
        return image_diagonal(pred, spacing)

    dist_to_ref = distance_transform_edt(~ref_surface, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate([dist_to_ref[pred_surface], dist_to_pred[ref_surface]])
    if distances.size == 0:
        return 0.0
    return float(np.percentile(distances, 95))


def compute_case(pred_path: Path, gt_path: Path) -> tuple[str, float]:
    pred, pred_spacing = read_mask(pred_path)
    ref, ref_spacing = read_mask(gt_path)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch for {pred_path.name}: pred={pred.shape} ref={ref.shape}")
    value = hd95(pred, ref, ref_spacing or pred_spacing)
    return pred_path.name.removesuffix(".nii.gz"), value


def compute_one(pred_dir: Path, gt_dir: Path, output_path: Path, workers: int) -> dict:
    pred_files = sorted(p for p in pred_dir.glob("*.nii.gz") if p.is_file())
    if not pred_files:
        raise FileNotFoundError(f"No prediction NIfTI files in {pred_dir}")

    per_case = {}
    values = []
    missing_gt = []
    work_items = []
    for pred_path in pred_files:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            missing_gt.append(pred_path.name)
            continue
        work_items.append((pred_path, gt_path))

    if workers > 1 and len(work_items) > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(compute_case, (item[0] for item in work_items), (item[1] for item in work_items))
            for case_id, value in results:
                per_case[case_id] = {"1": {"HD95": value}}
                values.append(value)
    else:
        for pred_path, gt_path in work_items:
            case_id, value = compute_case(pred_path, gt_path)
            per_case[case_id] = {"1": {"HD95": value}}
            values.append(value)

    if missing_gt:
        raise FileNotFoundError(f"Missing {len(missing_gt)} GT files under {gt_dir}: {missing_gt[:5]}")
    if not values:
        raise RuntimeError(f"No HD95 values computed for {pred_dir}")

    result = {
        "foreground_mean": {"HD95": float(np.mean(values))},
        "mean": {"1": {"HD95": float(np.mean(values))}},
        "metric_per_case": per_case,
        "num_cases": len(values),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    args = parse_args()
    project_root_ = args.project_root.expanduser().resolve()
    targets = [normalize_target(target) for target in args.targets]
    experiments = expand_experiments(args.experiments)
    workers = max(1, args.workers)

    failures = []
    completed = 0
    skipped = 0
    for target in targets:
        for experiment in experiments:
            out_dir = output_dir(project_root_, target, experiment)
            pred_dir = out_dir / "test_preds"
            gt_dir = out_dir / "test_gt"
            output_path = pred_dir / args.output_name
            if output_path.exists() and not args.force:
                skipped += 1
                continue
            if not (pred_dir / "summary.json").exists():
                continue
            try:
                result = compute_one(pred_dir, gt_dir, output_path, workers)
                completed += 1
                print(f"{target}/{experiment}: HD95={result['foreground_mean']['HD95']:.4f} n={result['num_cases']}", flush=True)
            except Exception as exc:  # noqa: BLE001
                message = f"{target}/{experiment}: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"FAILED {message}", flush=True)
                if args.strict:
                    raise

    print(f"HD95 complete: computed={completed} skipped={skipped} failures={len(failures)}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
