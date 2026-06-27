from __future__ import annotations

import shlex
from pathlib import Path

from .common import write_json
from .matrix import SCORE_METHODS_8D, dataset_methods, method_matrix
from .mamamia import BUDGETS as MAMAMIA_BUDGETS
from .mamamia import DOMAINS as MAMAMIA_DOMAINS
from .mamamia import METHODS as MAMAMIA_METHODS
from .officehome import DOMAINS as OFFICEHOME_DOMAINS


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def score_args(dataset: str, target: str) -> list[str]:
    args = []
    for method in SCORE_METHODS_8D:
        args.extend(["--score", f"{method}=scores/{dataset}/{target}/{method}.json"])
    return args


def one_hot_weight(method: str) -> list[str]:
    return ["1" if name == method else "0" for name in SCORE_METHODS_8D]


def mamamia_da_dataset_id(target: str, method: str, budget: int) -> str:
    target_offset = list(MAMAMIA_DOMAINS).index(target) * 100
    method_offset = list(dataset_methods("mamamia", "domain_adaptation")).index(method) * 10
    budget_offset = list(MAMAMIA_BUDGETS).index(budget)
    return str(9000 + target_offset + method_offset + budget_offset)


def mamamia_plan(data_root: str = "data/mamamia", split_root: str = "splits/mamamia_lodo_seed42", results_root: str = "outputs/nnunet_results") -> list[dict]:
    steps = []
    steps.append({"name": "mamamia_split", "cmd": ["python", "-m", "tavo_release.cli", "split", "--dataset", "mamamia", "--data-root", data_root, "--output-root", split_root]})
    for target in MAMAMIA_DOMAINS:
        for experiment in ("target_only", "source_only", "target_full_source"):
            steps.append({"name": f"mamamia_{target}_{experiment}", "cmd": ["python", "-m", "tavo_release.cli", "command", "--dataset", "mamamia", "--dataset-id", f"{target}:{experiment}"]})
        for budget in MAMAMIA_BUDGETS:
            for method in ("random", *MAMAMIA_METHODS):
                steps.append({"name": f"mamamia_{target}_{method}{budget}", "cmd": ["python", "-m", "tavo_release.cli", "command", "--dataset", "mamamia", "--dataset-id", f"{target}:{method}{budget}"]})
            for method in dataset_methods("mamamia", "domain_adaptation"):
                cfg = f"configs/generated/mamamia_{target}_{method}_{budget}.json"
                steps.append({"name": f"mamamia_{target}_{method}{budget}_da_config", "cmd": ["python", "-m", "tavo_release.cli", "da-config", "--dataset", "mamamia", "--method", method, "--split-dir", f"{split_root}/{target}", "--output-dir", f"outputs/mamamia/{target}/{method}{budget}", "--budget", str(budget), "--output", cfg, "--nnunet-dataset-id", mamamia_da_dataset_id(target, method, budget)]})
                steps.append({"name": f"mamamia_{target}_{method}{budget}_da_command", "cmd": ["python", "-m", "tavo_release.cli", "da-command", "--config", cfg]})
    steps.append({"name": "mamamia_collect", "cmd": ["python", "-m", "tavo_release.cli", "collect", "--dataset", "mamamia", "--results-root", results_root, "--output", "outputs/mamamia_results.json"]})
    return steps


def brats_plan(data_root: str = "data/brats", split_root: str = "splits/brats", target: str = "target") -> list[dict]:
    steps = [
        {"name": "brats_split", "cmd": ["python", "-m", "tavo_release.cli", "split", "--dataset", "brats", "--data-root", data_root, "--output-root", split_root, "--target", target]},
    ]
    for actual_target in method_matrix()["brats"]["targets"]:
        for budget in MAMAMIA_BUDGETS:
            for method in dataset_methods("brats", "selection"):
                if method == "random":
                    continue
                out = f"{split_root}/{actual_target}/methods/{method}_{budget}.txt"
                steps.append({"name": f"brats_{actual_target}_{method}{budget}_selection", "cmd": ["python", "-m", "tavo_release.cli", "select", *score_args("brats", actual_target), "--weight", *one_hot_weight(method), "--budget", str(budget), "--output", out]})
            steps.append({"name": f"brats_{actual_target}_tavo{budget}_search", "cmd": ["python", "-m", "tavo_release.cli", "search", *score_args("brats", actual_target), "--budget", str(budget), "--output-dir", f"outputs/brats/{actual_target}/tavo{budget}"]})
            for method in dataset_methods("brats", "domain_adaptation"):
                cfg = f"configs/generated/brats_{actual_target}_{method}_{budget}.json"
                steps.append({"name": f"brats_{actual_target}_{method}{budget}_da_config", "cmd": ["python", "-m", "tavo_release.cli", "da-config", "--dataset", "brats", "--method", method, "--split-dir", f"{split_root}/{actual_target}", "--output-dir", f"outputs/brats/{actual_target}/{method}{budget}", "--budget", str(budget), "--output", cfg]})
                steps.append({"name": f"brats_{actual_target}_{method}{budget}_da_command", "cmd": ["python", "-m", "tavo_release.cli", "da-command", "--config", cfg]})
        steps.append({"name": f"brats_{actual_target}_train_command", "cmd": ["python", "-m", "tavo_release.cli", "command", "--dataset", "brats", "--config", "configs/brats_efficientvit.yaml", "--seeds", "0"]})
    steps.append({"name": "brats_collect", "cmd": ["python", "-m", "tavo_release.cli", "collect", "--dataset", "brats", "--results-root", "outputs/brats", "--output", "outputs/brats_results.json"]})
    return steps


def officehome_plan(data_root: str = "data/officehome", split_root: str = "splits/officehome") -> list[dict]:
    steps = []
    for target in OFFICEHOME_DOMAINS:
        steps.append({"name": f"officehome_{target}_split", "cmd": ["python", "-m", "tavo_release.cli", "split", "--dataset", "officehome", "--data-root", data_root, "--output-root", split_root, "--target", target]})
        steps.append({"name": f"officehome_{target}_config", "cmd": ["python", "-m", "tavo_release.cli", "officehome-config", "--split-dir", f"{split_root}/{target}", "--output-dir", f"outputs/officehome/{target}", "--output", f"configs/officehome_{target}.json"]})
        steps.append({"name": f"officehome_{target}_train_command", "cmd": ["python", "-m", "tavo_release.cli", "command", "--dataset", "officehome", "--config", f"configs/officehome_{target}.json"]})
        for budget in MAMAMIA_BUDGETS:
            for method in dataset_methods("officehome", "selection"):
                if method == "random":
                    continue
                if method in SCORE_METHODS_8D:
                    steps.append({"name": f"officehome_{target}_{method}{budget}_selection", "cmd": ["python", "-m", "tavo_release.cli", "select", *score_args("officehome", target), "--weight", *one_hot_weight(method), "--budget", str(budget), "--output", f"{split_root}/{target}/methods/{method}_{budget}.txt"]})
                else:
                    steps.append({"name": f"officehome_{target}_{method}{budget}_selection_route", "cmd": ["python", "-m", "tavo_release.cli", "selection-route", "--dataset", "officehome", "--target", target, "--method", method, "--budget", str(budget)]})
            steps.append({"name": f"officehome_{target}_tavo{budget}_search", "cmd": ["python", "-m", "tavo_release.cli", "search", *score_args("officehome", target), "--budget", str(budget), "--output-dir", f"outputs/officehome/{target}/tavo{budget}"]})
            for method in dataset_methods("officehome", "domain_adaptation"):
                cfg = f"configs/generated/officehome_{target}_{method}_{budget}.json"
                steps.append({"name": f"officehome_{target}_{method}{budget}_da_config", "cmd": ["python", "-m", "tavo_release.cli", "da-config", "--dataset", "officehome", "--method", method, "--split-dir", f"{split_root}/{target}", "--output-dir", f"outputs/officehome/{target}/{method}{budget}", "--budget", str(budget), "--output", cfg]})
                steps.append({"name": f"officehome_{target}_{method}{budget}_da_command", "cmd": ["python", "-m", "tavo_release.cli", "da-command", "--config", cfg]})
    steps.append({"name": "officehome_collect", "cmd": ["python", "-m", "tavo_release.cli", "collect", "--dataset", "officehome", "--results-root", "outputs/officehome", "--output", "outputs/officehome_results.json"]})
    return steps


def combined_plan() -> list[dict]:
    return mamamia_plan() + brats_plan() + officehome_plan()


def write_plan(dataset: str, output_dir: str | Path) -> dict:
    if dataset == "mamamia":
        steps = mamamia_plan()
    elif dataset == "brats":
        steps = brats_plan()
    elif dataset == "officehome":
        steps = officehome_plan()
    elif dataset == "all":
        steps = combined_plan()
    else:
        raise ValueError(dataset)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = write_json(out / f"{dataset}_plan.json", steps)
    shell_path = out / f"{dataset}_plan.sh"
    shell_path.write_text("set -euo pipefail\nROOT=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")/..\" && pwd)\"\ncd \"$ROOT\"\nexport PYTHONPATH=\"$ROOT\"\n" + "\n".join(shell_join(step["cmd"]) for step in steps) + "\n")
    shell_path.chmod(0o755)
    return {"steps": len(steps), "json": str(json_path), "shell": str(shell_path)}
