from __future__ import annotations

from pathlib import Path

import yaml

from .common import list_cases, ratio_split, read_lines, stable_shuffle, write_json, write_lines


METHODS = ("rds", "less", "orient", "craig", "gradmatch", "kmeans", "kcenter", "diversity", "tavo")


def infer_domain(subject: str) -> str:
    parts = subject.replace("-", "_").split("_")
    if len(parts) >= 2 and not parts[1].isdigit():
        return parts[1]
    return "default"


def build_domain_splits(data_root: str | Path, output_root: str | Path, target_domain: str, seed: int = 42, ratios=(2, 1, 7)) -> dict[str, int]:
    subjects = list_cases(data_root)
    by_domain: dict[str, list[str]] = {}
    for subject in subjects:
        by_domain.setdefault(infer_domain(subject), []).append(subject)
    target = sorted(by_domain.get(target_domain, []))
    source = sorted(s for d, values in by_domain.items() if d != target_domain for s in values)
    train, val, test = ratio_split(target, ratios, seed)
    out = Path(output_root) / target_domain
    write_lines(out / "target_train.txt", train)
    write_lines(out / "target_val.txt", val)
    write_lines(out / "target_test.txt", test)
    write_lines(out / "source_pool.txt", source)
    shuffled = stable_shuffle(source, seed)
    for budget in (5, 10, 50, 150, 250):
        write_lines(out / "random" / f"random_{budget}.txt", shuffled[:budget])
    summary = {"target_train": len(train), "target_val": len(val), "target_test": len(test), "source_pool": len(source)}
    write_json(out / "summary.json", summary)
    return summary


def build_training_config(template: str | Path, output: str | Path, train_list: str | Path, val_list: str | Path, output_dir: str | Path, max_iters: int, warmup: str | None = None) -> Path:
    cfg = yaml.safe_load(Path(template).read_text())
    cfg.setdefault("data", {})
    cfg.setdefault("trainer", {})
    cfg.setdefault("training", {})
    cfg["data"]["train_subjects"] = str(Path(train_list))
    cfg["data"]["val_subjects"] = str(Path(val_list))
    cfg["trainer"]["max_iters"] = int(max_iters)
    cfg["training"]["save_dir"] = str(Path(output_dir))
    if warmup:
        cfg["warmup"] = {"checkpoint": str(Path(warmup))}
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out


def build_train_command(config: str | Path, seeds: str = "0", entrypoint: str = "python -m tavo_release.segmentation_train") -> list[str]:
    return entrypoint.split() + ["--config", str(Path(config)), "--seeds", str(seeds)]


def selection_cases(split_root: str | Path, target_domain: str, method: str, budget: int) -> list[str]:
    root = Path(split_root) / target_domain
    if method == "random":
        return read_lines(root / "random" / f"random_{budget}.txt")
    return read_lines(root / "methods" / f"{method}_{budget}.txt")
