#!/usr/bin/env python3
"""Materialize MAMAMIA method selections into the current LODO split tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core import METHODS, METHOD_BUDGETS, REPO_ROOT, TARGET_BASES, TAVO_BUDGET_OFFSETS, normalize_target, project_root, split_root

PROJECT_ROOT = project_root()
DEFAULT_SPLIT_ROOT = split_root()
SELECTION_ROOT = PROJECT_ROOT
DERIVED_TUMORSEG_SELECTION_ROOT = REPO_ROOT / "artifacts" / "tumorseg2025_selections"

TARGET_SELECTION_DIRS = {
    "NACT": SELECTION_ROOT / "mamamia_clean" / "outputs" / "selections",
    "ISPY1": SELECTION_ROOT / "mamamia_ispy1" / "outputs" / "selections",
    "DUKE": SELECTION_ROOT / "mamamia_duke" / "outputs" / "selections",
    "ISPY2": SELECTION_ROOT / "mamamia_ispy2" / "outputs" / "selections",
}
GLOBAL_SELECTION_DIR = SELECTION_ROOT / "dataset_mamamia" / "selections"


def real_tavo_selection_paths(target: str, budget: int) -> list[Path]:
    """Prefer fresh paper-style 8D CMA searches over legacy artifacts."""
    target_roots = {
        "NACT": SELECTION_ROOT / "mamamia_clean",
        "ISPY1": SELECTION_ROOT / "mamamia_ispy1",
        "DUKE": SELECTION_ROOT / "mamamia_duke",
        "ISPY2": SELECTION_ROOT / "mamamia_ispy2",
    }
    global_meta_dir = SELECTION_ROOT / "outputs" / "meta"
    fresh_8d_candidates: list[Path] = []
    if global_meta_dir.exists():
        fresh_8d_candidates = list(
            global_meta_dir.glob(f"tavo_8d_cmaes_*_{target}_b{budget}/selection_{budget}_median_val_42.json")
        )
    fresh_8d_candidates = sorted(fresh_8d_candidates, key=lambda path: path.stat().st_mtime, reverse=True)

    meta_dir = target_roots[target] / "outputs" / "meta"
    if not meta_dir.exists():
        return fresh_8d_candidates
    legacy_candidates = list(meta_dir.glob(f"cmaes_nnunet500_real_*_b{budget}/selection_{budget}_median_val_42.json"))
    legacy_candidates = sorted(legacy_candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    return fresh_8d_candidates + legacy_candidates


TAVO_SELECTION_PATHS = {
    "NACT": [
        SELECTION_ROOT / "mamamia_clean" / "outputs" / "meta" / "cmaes_nnunet500" / "selection_250_median_val_42.json",
        SELECTION_ROOT / "mamamia_clean" / "outputs" / "meta" / "cmaes_20gen" / "selection_250_median_val_42.json",
        SELECTION_ROOT / "mamamia_clean" / "outputs" / "meta" / "cmaes" / "selection_250_median_val_42.json",
    ],
    "ISPY1": [
        SELECTION_ROOT / "mamamia_ispy1" / "outputs" / "meta" / "cmaes_nnunet500" / "selection_250_median_val_42.json",
        SELECTION_ROOT
        / "mamamia_ispy1"
        / "outputs"
        / "meta"
        / "bo_sh_eta3_nnunet500"
        / "selection_250_median_val_eta3_42.json",
    ],
    "DUKE": [
        SELECTION_ROOT / "mamamia_duke" / "outputs" / "selections" / "meta_cmaes_250.json",
        SELECTION_ROOT / "mamamia_duke" / "outputs" / "selections" / "meta_bo_sh_eta3_250.json",
    ],
    "ISPY2": [
        SELECTION_ROOT / "mamamia_ispy2" / "outputs" / "selections" / "meta_cmaes_gen6_250.json",
        SELECTION_ROOT / "mamamia_ispy2" / "outputs" / "selections" / "meta_bo_gen7_250.json",
    ],
}
MATERIALIZABLE_METHODS = METHODS + ("tavo",)
MATERIALIZABLE_BUDGETS = tuple(sorted(set(METHOD_BUDGETS) | set(TAVO_BUDGET_OFFSETS)))


def read_list(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing list file: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n")


def selected_cases(data: dict[str, Any]) -> list[str]:
    for key in ("selected", "selected_cases"):
        values = data.get(key)
        if isinstance(values, list):
            return [str(value) for value in values]
    scores = data.get("scores")
    if isinstance(scores, dict):
        return [case_id for case_id, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    raise ValueError("selection JSON has no selected/selected_cases list or scores dictionary")


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def candidate_paths(target: str, method: str, budget: int) -> list[Path]:
    target_dir = TARGET_SELECTION_DIRS[target]
    candidates: list[Path] = []

    if method == "tavo":
        candidates.extend(real_tavo_selection_paths(target, budget))
        candidates.extend(TAVO_SELECTION_PATHS[target])
    else:
        candidates.extend(
            [
                DERIVED_TUMORSEG_SELECTION_ROOT / target / f"{method}_{budget}.json",
                DERIVED_TUMORSEG_SELECTION_ROOT / target / f"{method}_250.json",
            ]
        )

    if method == "tavo":
        pass
    elif target == "NACT" and method in {"rds", "less", "gradmatch", "craig"}:
        candidates.extend([target_dir / f"{method}_{budget}.json", target_dir / f"{method}_260.json"])
    elif target in {"ISPY1", "DUKE", "ISPY2"} and method in {"less", "gradmatch", "craig"}:
        candidates.extend(
            [
                target_dir / f"{method}_proxy_{budget}.json",
                target_dir / f"{method}_proxy_250.json",
                target_dir / f"{method}_{budget}.json",
                target_dir / f"{method}_250.json",
            ]
        )
    else:
        candidates.extend([target_dir / f"{method}_{budget}.json", target_dir / f"{method}_250.json"])

    candidates.extend([GLOBAL_SELECTION_DIR / f"{method}_{budget}.json", GLOBAL_SELECTION_DIR / f"{method}_250.json"])
    if method == "rds":
        candidates.extend([GLOBAL_SELECTION_DIR / f"rds_plus_{budget}.json", GLOBAL_SELECTION_DIR / "rds_plus_250.json"])

    out: list[Path] = []
    seen = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        out.append(path)
    return existing(out)


def materialize_one(args: argparse.Namespace, target: str, method: str, budget: int) -> dict[str, Any]:
    target_dir = args.split_root / target
    source_pool = set(read_list(target_dir / "source_pool.txt"))
    method_dir = target_dir / "methods"
    output_path = method_dir / f"{method}_{budget}.txt"
    provenance_path = method_dir / f"{method}_{budget}.provenance.json"

    attempts: list[dict[str, Any]] = []
    for path in candidate_paths(target, method, budget):
        data = json.loads(path.read_text())
        ordered = selected_cases(data)
        valid: list[str] = []
        seen = set()
        skipped_not_source: list[str] = []
        duplicates = 0
        for case_id in ordered:
            if case_id in seen:
                duplicates += 1
                continue
            seen.add(case_id)
            if case_id not in source_pool:
                skipped_not_source.append(case_id)
                continue
            valid.append(case_id)
        attempt = {
            "path": str(path),
            "raw_count": len(ordered),
            "valid_source_count": len(valid),
            "duplicate_count": duplicates,
            "skipped_not_source_count": len(skipped_not_source),
        }
        attempts.append(attempt)
        if len(valid) < budget:
            continue

        selected = valid[:budget]
        if args.dry_run:
            status = "would_write"
        else:
            write_list(output_path, selected)
            provenance = {
                "target": target,
                "method": method,
                "budget": budget,
                "output": str(output_path),
                "source_artifact": str(path),
                "artifact_method": data.get("method"),
                "selected_count": len(selected),
                "valid_source_count": len(valid),
                "raw_count": len(ordered),
                "duplicate_count": duplicates,
                "skipped_not_source_count": len(skipped_not_source),
                "skipped_not_source_preview": skipped_not_source[:25],
                "all_attempts": attempts,
            }
            provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
            status = "written"
        return {
            "target": target,
            "method": method,
            "budget": budget,
            "status": status,
            "output": str(output_path),
            "source_artifact": str(path),
            "selected_count": budget,
        }

    result = {
        "target": target,
        "method": method,
        "budget": budget,
        "status": "missing_or_insufficient",
        "attempts": attempts,
    }
    if args.strict:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument("--methods", nargs="+", default=list(MATERIALIZABLE_METHODS))
    parser.add_argument("--budgets", nargs="+", type=int, default=list(METHOD_BUDGETS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail if any target/method/budget cannot be materialized.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = list(TARGET_BASES) if "all" in args.targets else [normalize_target(target) for target in args.targets]
    methods = [method.lower() for method in args.methods]
    unknown_methods = [method for method in methods if method not in MATERIALIZABLE_METHODS]
    if unknown_methods:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown_methods)}")
    unknown_budgets = [budget for budget in args.budgets if budget not in MATERIALIZABLE_BUDGETS]
    if unknown_budgets:
        raise SystemExit(f"Unsupported budget(s): {', '.join(map(str, unknown_budgets))}")

    results = [
        materialize_one(args, target, method, budget)
        for target in targets
        for budget in args.budgets
        for method in methods
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
