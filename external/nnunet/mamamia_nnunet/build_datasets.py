#!/usr/bin/env python3
"""Create nnUNet raw datasets for the corrected MAMAMIA LODO table subset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from core import (
    EXPERIMENTS,
    TARGET_BASES,
    Experiment,
    dataset_id,
    dataset_name,
    dataset_root,
    expand_experiments,
    nnunet_raw_root,
    normalize_target,
    split_root,
)

DEFAULT_DATASET_ROOT = dataset_root()
DEFAULT_NNUNET_RAW = nnunet_raw_root()
DEFAULT_SPLIT_ROOT = split_root()


def read_list(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing list file: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def unique_in_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def channel_names(num_phases: int) -> dict[str, str]:
    return {str(idx): f"dce_{idx:04d}" for idx in range(num_phases)}


def source_image_paths(images_root: Path, case_id: str, num_phases: int) -> list[Path]:
    case_dir = images_root / case_id
    if not case_dir.exists():
        raise SystemExit(f"Missing image case directory: {case_dir}")
    paths = [case_dir / f"{case_id}_{phase:04d}.nii.gz" for phase in range(num_phases)]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing image phase(s) for {case_id}: {missing}")
    return paths


def link_or_copy(src: Path, dst: Path, link_mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if link_mode == "symlink":
        dst.symlink_to(src)
        return "symlink"
    if link_mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"
    shutil.copy2(src, dst)
    return "copy"


def train_cases_for(exp: Experiment, target_dir: Path) -> list[str]:
    target_train = read_list(target_dir / "target_train.txt")
    if exp.key == "target_only":
        return target_train
    if exp.key == "source_only":
        return read_list(target_dir / "source_pool.txt")
    if exp.key == "target_full_source":
        return unique_in_order(read_list(target_dir / "source_pool.txt") + target_train)
    if exp.key.startswith("random"):
        budget = exp.key.removeprefix("random")
        return unique_in_order(read_list(target_dir / "random" / f"random_{budget}.txt") + target_train)
    if exp.method and exp.budget:
        method_cases = read_list(target_dir / "methods" / f"{exp.method}_{exp.budget}.txt")
        return unique_in_order(method_cases + target_train)
    raise SystemExit(f"Unsupported experiment: {exp.key}")


def validate_cases(
    images_root: Path,
    labels_root: Path,
    train_cases: list[str],
    val_cases: list[str],
    test_cases: list[str],
    num_phases: int,
) -> None:
    for case_id in train_cases + val_cases:
        source_image_paths(images_root, case_id, num_phases)
        label = labels_root / f"{case_id}.nii.gz"
        if not label.exists():
            raise SystemExit(f"Missing expert label: {label}")
    for case_id in test_cases:
        source_image_paths(images_root, case_id, num_phases)


def build_one(args: argparse.Namespace, target: str, exp: Experiment) -> dict[str, object]:
    images_root = args.dataset_root / "images"
    labels_root = args.dataset_root / "segmentations" / "expert"
    target_dir = args.split_root / target
    if not target_dir.exists():
        raise SystemExit(f"Missing split directory: {target_dir}. Run build_splits.py first.")

    train_cases = train_cases_for(exp, target_dir)
    val_cases = read_list(target_dir / "target_val.txt")
    test_cases = read_list(target_dir / "target_test.txt")

    for required in (images_root, labels_root, args.nnunet_raw):
        if not required.exists():
            raise SystemExit(f"Missing required path: {required}")

    validate_cases(images_root, labels_root, train_cases, val_cases, test_cases, args.num_phases)

    ds_id = dataset_id(target, exp)
    ds_name = dataset_name(target, exp)
    dataset_dir = args.nnunet_raw / f"Dataset{ds_id}_{ds_name}"
    metadata = {
        "target": target,
        "experiment": exp.key,
        "dataset_id": ds_id,
        "dataset_name": ds_name,
        "dataset_dir": str(dataset_dir),
        "description": exp.description,
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "test_cases": len(test_cases),
        "train_source": exp.train_source,
        "split_root": str(args.split_root),
    }

    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return metadata

    if dataset_dir.exists():
        if args.skip_existing:
            print(f"Skipping existing dataset: {dataset_dir}")
            return metadata | {"skipped": True}
        if not args.overwrite:
            raise SystemExit(f"Dataset already exists; pass --overwrite or --skip-existing: {dataset_dir}")
        shutil.rmtree(dataset_dir)

    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)
    images_ts.mkdir(parents=True)

    op_counts = {"symlink": 0, "hardlink": 0, "copy": 0}
    for case_id in train_cases + val_cases:
        for image in source_image_paths(images_root, case_id, args.num_phases):
            op = link_or_copy(image, images_tr / image.name, args.link_mode)
            op_counts[op] += 1
        label = labels_root / f"{case_id}.nii.gz"
        op = link_or_copy(label, labels_tr / label.name, args.link_mode)
        op_counts[op] += 1

    for case_id in test_cases:
        for image in source_image_paths(images_root, case_id, args.num_phases):
            op = link_or_copy(image, images_ts / image.name, args.link_mode)
            op_counts[op] += 1

    dataset_json = {
        "channel_names": channel_names(args.num_phases),
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(train_cases) + len(val_cases),
        "file_ending": ".nii.gz",
        "dataset_name": ds_name.removesuffix("_2d_3ch"),
    }
    splits_final = [{"train": train_cases, "val": val_cases}]
    dataset_dir.joinpath("dataset.json").write_text(json.dumps(dataset_json, indent=2) + "\n")
    dataset_dir.joinpath("splits_final.json").write_text(json.dumps(splits_final, indent=2) + "\n")
    dataset_dir.joinpath("tavo_mamamia_manifest.json").write_text(
        json.dumps(metadata | {"link_mode": args.link_mode, "file_operations": op_counts}, indent=2) + "\n"
    )

    print(f"Prepared {target}/{exp.key}: {dataset_dir}")
    print(f"  train={len(train_cases)} val={len(val_cases)} test={len(test_cases)} ops={op_counts}")
    return metadata | {"file_operations": op_counts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiments", nargs="*", default=["all"], help="Experiment keys or 'all'.")
    parser.add_argument("--targets", nargs="+", default=["all"], help="Target domains or 'all'.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--nnunet-raw", type=Path, default=DEFAULT_NNUNET_RAW)
    parser.add_argument("--num-phases", type=int, default=3)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_exps = expand_experiments(args.experiments)
    requested_targets = list(TARGET_BASES) if "all" in args.targets else [normalize_target(target) for target in args.targets]
    unknown_exps = [key for key in requested_exps if key not in EXPERIMENTS]
    unknown_targets = [target for target in requested_targets if target not in TARGET_BASES]
    if unknown_exps:
        raise SystemExit(f"Unknown experiment(s): {', '.join(unknown_exps)}")
    if unknown_targets:
        raise SystemExit(f"Unknown target(s): {', '.join(unknown_targets)}")

    summaries = [
        build_one(args, target, EXPERIMENTS[exp_key])
        for target in requested_targets
        for exp_key in requested_exps
    ]
    print("Summary:")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
