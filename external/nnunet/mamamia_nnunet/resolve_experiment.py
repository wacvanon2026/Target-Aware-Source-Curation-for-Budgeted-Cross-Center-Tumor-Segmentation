#!/usr/bin/env python3
"""Resolve shared MAMAMIA experiment metadata for shell scripts."""

from __future__ import annotations

import argparse
import shlex

from experiments import EXPERIMENTS, GROUPS, dataset_basename, dataset_id, dataset_name, expand_experiments, get_experiment, normalize_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", nargs="?")
    parser.add_argument("experiment", nargs="?")
    parser.add_argument("--shell", action="store_true", help="Print shell assignments for run_one.sh.")
    parser.add_argument("--list", choices=("all", *GROUPS), help="Print experiment keys for a group.")
    return parser.parse_args()


def shell_pair(key: str, value: object) -> str:
    return f"{key}={shlex.quote(str(value))}"


def main() -> None:
    args = parse_args()
    if args.list:
        keys = list(EXPERIMENTS) if args.list == "all" else list(GROUPS[args.list])
        print(" ".join(keys))
        return
    if not args.target or not args.experiment:
        raise SystemExit("target and experiment are required unless --list is used")

    target = normalize_target(args.target)
    exp = get_experiment(args.experiment)
    values = {
        "TARGET": target,
        "EXPERIMENT": exp.key,
        "OFFSET": exp.offset,
        "SUFFIX": exp.suffix,
        "LABEL": exp.label,
        "DATASET_ID": dataset_id(target, exp),
        "DATASET_NAME": dataset_name(target, exp),
        "DATASET_BASENAME": dataset_basename(target, exp),
        "EXPERIMENT_BLOCK": exp.block,
        "EXPERIMENT_METHOD": exp.method or "",
        "EXPERIMENT_BUDGET": exp.budget or "",
    }
    if args.shell:
        print("\n".join(shell_pair(key, value) for key, value in values.items()))
    else:
        for key, value in values.items():
            print(f"{key}\t{value}")


if __name__ == "__main__":
    main()
