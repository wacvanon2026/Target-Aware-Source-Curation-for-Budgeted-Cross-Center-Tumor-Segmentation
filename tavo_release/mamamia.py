from __future__ import annotations

import json
from pathlib import Path

from .common import list_cases, ratio_split, read_lines, stable_shuffle, symlink_or_copy, write_json, write_lines


DOMAINS = {"NACT": "NACT_", "ISPY1": "ISPY1_", "DUKE": "DUKE_", "ISPY2": "ISPY2_"}
BUDGETS = (50, 150, 250)
METHODS = ("rds", "gradmatch", "less", "orient", "diversity", "kmeans", "craig", "kcenter", "tavo")


def domain_for_case(case_id: str) -> str:
    for domain, prefix in DOMAINS.items():
        if case_id.startswith(prefix):
            return domain
    raise ValueError(f"unrecognized MAMA-MIA case id: {case_id}")


def build_lodo_splits(dataset_root: str | Path, output_root: str | Path, seed: int = 42, ratios=(2, 1, 7)) -> dict[str, dict[str, int]]:
    cases = list_cases(Path(dataset_root) / "images")
    by_domain = {domain: [] for domain in DOMAINS}
    for case in cases:
        by_domain[domain_for_case(case)].append(case)
    summary = {}
    out = Path(output_root)
    for target, target_cases in by_domain.items():
        target_train, target_val, target_test = ratio_split(target_cases, ratios, seed)
        source_pool = sorted(case for domain, values in by_domain.items() if domain != target for case in values)
        root = out / target
        write_lines(root / "target_train.txt", target_train)
        write_lines(root / "target_val.txt", target_val)
        write_lines(root / "target_test.txt", target_test)
        write_lines(root / "source_pool.txt", source_pool)
        random_dir = root / "random"
        shuffled = stable_shuffle(source_pool, seed)
        for budget in BUDGETS:
            write_lines(random_dir / f"random_{budget}.txt", shuffled[:budget])
        summary[target] = {
            "target_train": len(target_train),
            "target_val": len(target_val),
            "target_test": len(target_test),
            "source_pool": len(source_pool),
        }
    write_json(out / "summary.json", summary)
    return summary


def selection_path(split_root: str | Path, target: str, method: str, budget: int) -> Path:
    if method == "random":
        return Path(split_root) / target / "random" / f"random_{budget}.txt"
    return Path(split_root) / target / "methods" / f"{method}_{budget}.txt"


def experiment_cases(split_root: str | Path, target: str, experiment: str, budget: int | None = None) -> list[str]:
    root = Path(split_root) / target
    if experiment == "target_only":
        return read_lines(root / "target_train.txt")
    if experiment == "source_only":
        return read_lines(root / "source_pool.txt")
    if experiment == "target_full_source":
        return read_lines(root / "source_pool.txt") + read_lines(root / "target_train.txt")
    if budget is None:
        raise ValueError("budget is required for selection experiments")
    return read_lines(selection_path(split_root, target, experiment, budget)) + read_lines(root / "target_train.txt")


def dataset_id(target: str, experiment: str, budget: int | None = None, base_id: int = 1300) -> int:
    domain_offset = list(DOMAINS).index(target) * 100
    if experiment == "target_only":
        return base_id + domain_offset + 1
    if experiment == "source_only":
        return base_id + domain_offset + 2
    if experiment == "target_full_source":
        return base_id + domain_offset + 3
    if budget is None:
        raise ValueError("budget is required")
    method_offset = {"random": 4, "rds": 10, "gradmatch": 13, "less": 16, "orient": 19, "diversity": 22, "kmeans": 25, "craig": 28, "kcenter": 31, "tavo": 34}
    budget_offset = {50: 0, 150: 1, 250: 2}
    return base_id + domain_offset + method_offset[experiment] + budget_offset[budget]


def parse_experiment_ref(ref: str) -> tuple[str, str, int | None]:
    if ":" not in ref:
        raise ValueError("expected TARGET:EXPERIMENT")
    target, experiment = ref.split(":", 1)
    if target not in DOMAINS:
        raise ValueError(f"unknown target: {target}")
    for budget in sorted(BUDGETS, reverse=True):
        suffix = str(budget)
        if experiment.endswith(suffix):
            method = experiment[: -len(suffix)]
            return target, method, budget
    return target, experiment, None


def resolve_dataset(value: str | int) -> int:
    if isinstance(value, int):
        return value
    text = str(value)
    if text.isdigit():
        return int(text)
    target, experiment, budget = parse_experiment_ref(text)
    return dataset_id(target, experiment, budget)


def dataset_name(target: str, experiment: str, budget: int | None = None) -> str:
    did = dataset_id(target, experiment, budget)
    suffix = experiment.upper() if budget is None else f"{experiment.upper()}{budget}"
    return f"Dataset{did}_MAMAMIA_{target}_LODO_TAVO_{suffix}_2d_3ch"


def materialize_nnunet_raw(dataset_root: str | Path, split_root: str | Path, output_raw: str | Path, target: str, experiment: str, budget: int | None = None, copy: bool = False) -> Path:
    cases = experiment_cases(split_root, target, experiment, budget)
    ds_name = dataset_name(target, experiment, budget)
    out = Path(output_raw) / ds_name
    images_tr = out / "imagesTr"
    labels_tr = out / "labelsTr"
    images_ts = out / "imagesTs"
    for case in cases:
        case_dir = Path(dataset_root) / "images" / case
        for channel in ("0000", "0001", "0002"):
            src = case_dir / f"{case}_{channel}.nii.gz"
            if src.exists():
                symlink_or_copy(src, images_tr / f"{case}_{channel}.nii.gz", copy=copy)
        label = Path(dataset_root) / "segmentations" / "expert" / f"{case}.nii.gz"
        if label.exists():
            symlink_or_copy(label, labels_tr / f"{case}.nii.gz", copy=copy)
    for case in read_lines(Path(split_root) / target / "target_test.txt"):
        case_dir = Path(dataset_root) / "images" / case
        for channel in ("0000", "0001", "0002"):
            src = case_dir / f"{case}_{channel}.nii.gz"
            if src.exists():
                symlink_or_copy(src, images_ts / f"{case}_{channel}.nii.gz", copy=copy)
    dataset_json = {
        "channel_names": {"0": "dce_0", "1": "dce_1", "2": "dce_2"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    write_json(out / "dataset.json", dataset_json)
    return out


def nnunet_commands(dataset: str | int, trainer: str = "nnUNetTrainer", fold: int = 0, configuration: str = "2d") -> dict[str, list[str]]:
    ds = str(dataset)
    train_command = ["nnUNetv2_train", ds, configuration, str(fold), "-tr", trainer]
    if trainer.startswith("nnUNetTrainerTAVO"):
        train_command = ["python", "-m", "tavo_release.mamamia_nnunet_train", ds, configuration, str(fold), "-tr", trainer]
    return {
        "plan": ["nnUNetv2_plan_and_preprocess", "-d", ds, "-c", configuration, "--verify_dataset_integrity"],
        "train": train_command,
        "predict": ["nnUNetv2_predict", "-d", ds, "-c", configuration, "-f", str(fold), "-tr", trainer],
    }


def collect_metric_jsons(results_root: str | Path) -> list[dict]:
    rows = []
    for path in Path(results_root).rglob("summary.json"):
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        row = {"path": str(path)}
        if isinstance(value, dict):
            for key in ("dice", "mean_dice", "hd95", "mean_hd95"):
                if key in value:
                    row[key] = value[key]
            if "foreground_mean" in value and isinstance(value["foreground_mean"], dict):
                row.update(value["foreground_mean"])
        rows.append(row)
    return rows
