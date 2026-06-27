from __future__ import annotations

from pathlib import Path

from .matrix import SCORE_METHODS_8D


def score_args(dataset: str, target: str, score_root: str | Path = "scores") -> list[str]:
    args = []
    for method in SCORE_METHODS_8D:
        args.extend(["--score", f"{method}={Path(score_root) / dataset / target / (method + '.json')}"])
    return args


def search_command(dataset: str, target: str, budget: int, output_dir: str | Path | None = None, score_root: str | Path = "scores") -> list[str]:
    out = Path(output_dir) if output_dir else Path("outputs") / dataset / target / f"tavo{budget}"
    return ["python", "-m", "tavo_release.cli", "search", *score_args(dataset, target, score_root), "--budget", str(int(budget)), "--output-dir", str(out)]


def selection_command(dataset: str, target: str, method: str, budget: int, output: str | Path | None = None, score_root: str | Path = "scores") -> list[str]:
    if method not in SCORE_METHODS_8D:
        raise ValueError(method)
    weights = ["1" if name == method else "0" for name in SCORE_METHODS_8D]
    out = Path(output) if output else Path("splits") / dataset / target / "methods" / f"{method}_{budget}.txt"
    return ["python", "-m", "tavo_release.cli", "select", *score_args(dataset, target, score_root), "--weight", *weights, "--budget", str(int(budget)), "--output", str(out)]
