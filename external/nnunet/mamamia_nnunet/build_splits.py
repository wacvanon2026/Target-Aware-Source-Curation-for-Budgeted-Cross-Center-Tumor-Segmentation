#!/usr/bin/env python3
"""Generate MAMAMIA leave-one-domain-out splits with a 2:1:7 target ratio."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from core import BUDGETS, DOMAINS, RATIO, dataset_root, split_root

DEFAULT_DATASET_ROOT = dataset_root()
DEFAULT_SPLIT_ROOT = split_root()


def write_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + "\n")


def split_counts(n: int) -> dict[str, int]:
    denom = sum(RATIO.values())
    floors = {name: (n * weight) // denom for name, weight in RATIO.items()}
    remainders = {
        name: (n * weight) / denom - floors[name]
        for name, weight in RATIO.items()
    }
    remaining = n - sum(floors.values())
    order = sorted(RATIO, key=lambda name: (-remainders[name], list(RATIO).index(name)))
    counts = dict(floors)
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def collect_domains(images_root: Path) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = {}
    all_case_dirs = [path.name for path in images_root.iterdir() if path.is_dir()]
    for domain, prefix in DOMAINS.items():
        domains[domain] = sorted(case_id for case_id in all_case_dirs if case_id.startswith(prefix))
    unknown = sorted(
        case_id
        for case_id in all_case_dirs
        if not any(case_id.startswith(prefix) for prefix in DOMAINS.values())
    )
    if unknown:
        raise SystemExit(f"Found case directories outside known domains: {unknown[:20]}")
    return domains


def build_target_split(target: str, domains: dict[str, list[str]], seed: int) -> dict[str, object]:
    target_cases = list(domains[target])
    rng = random.Random(f"{seed}:{target}:target")
    rng.shuffle(target_cases)

    counts = split_counts(len(target_cases))
    train_end = counts["train"]
    val_end = train_end + counts["val"]
    target_train = target_cases[:train_end]
    target_val = target_cases[train_end:val_end]
    target_test = target_cases[val_end:]

    source_pool = sorted(
        case_id
        for domain, cases in domains.items()
        if domain != target
        for case_id in cases
    )
    source_rng = random.Random(f"{seed}:{target}:source")
    random_source = list(source_pool)
    source_rng.shuffle(random_source)

    random_by_budget = {}
    for budget in BUDGETS:
        if budget > len(random_source):
            raise SystemExit(f"Budget {budget} exceeds source pool size {len(random_source)} for {target}")
        random_by_budget[str(budget)] = random_source[:budget]

    return {
        "target": target,
        "seed": seed,
        "ratio": RATIO,
        "target_total": len(target_cases),
        "source_total": len(source_pool),
        "counts": {
            "target_train": len(target_train),
            "target_val": len(target_val),
            "target_test": len(target_test),
        },
        "target_train": target_train,
        "target_val": target_val,
        "target_test": target_test,
        "source_pool": source_pool,
        "random": random_by_budget,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--targets", nargs="+", default=["all"], choices=["all", *DOMAINS])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = args.dataset_root / "images"
    if not images_root.exists():
        raise SystemExit(f"Missing images root: {images_root}")

    domains = collect_domains(images_root)
    targets = list(DOMAINS) if "all" in args.targets else args.targets
    summary = {
        "seed": args.seed,
        "ratio": RATIO,
        "domain_counts": {domain: len(cases) for domain, cases in domains.items()},
        "targets": {},
    }

    for target in targets:
        split = build_target_split(target, domains, args.seed)
        target_dir = args.split_root / target
        write_list(target_dir / "target_train.txt", split["target_train"])
        write_list(target_dir / "target_val.txt", split["target_val"])
        write_list(target_dir / "target_test.txt", split["target_test"])
        write_list(target_dir / "source_pool.txt", split["source_pool"])
        random_dir = target_dir / "random"
        for budget, cases in split["random"].items():
            write_list(random_dir / f"random_{budget}.txt", cases)

        serializable = {
            key: value
            for key, value in split.items()
            if key not in {"target_train", "target_val", "target_test", "source_pool", "random"}
        }
        serializable["random_counts"] = {budget: len(cases) for budget, cases in split["random"].items()}
        (target_dir / "summary.json").write_text(json.dumps(serializable, indent=2) + "\n")
        summary["targets"][target] = serializable
        print(
            f"{target}: target={split['target_total']} train={split['counts']['target_train']} "
            f"val={split['counts']['target_val']} test={split['counts']['target_test']} "
            f"source={split['source_total']}"
        )

    args.split_root.mkdir(parents=True, exist_ok=True)
    (args.split_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote splits to {args.split_root}")


if __name__ == "__main__":
    main()
