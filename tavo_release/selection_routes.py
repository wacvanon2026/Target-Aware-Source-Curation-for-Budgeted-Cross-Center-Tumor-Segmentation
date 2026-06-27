from __future__ import annotations

from pathlib import Path

from .matrix import BUDGETS, DATASET_METHODS, SCORE_METHODS_8D
from .pathways import REQUIRED_DATASETS, load_pathways
from .tavo_routes import selection_command


KEY_TO_PUBLIC = {value: key for key, value in REQUIRED_DATASETS.items()}


def dataset_spec(dataset: str, pathways_path: str | Path = "configs/pathways.json") -> dict:
    public_name = KEY_TO_PUBLIC.get(dataset)
    if public_name is None:
        raise ValueError(dataset)
    for spec in load_pathways(pathways_path):
        if spec.get("dataset") == public_name:
            return spec
    raise ValueError(dataset)


def render_pattern(pattern: str, target: str, source: str, budget: int) -> str:
    return pattern.replace("TARGET", target).replace("SOURCE", source).replace("BUDGET", str(budget))


def selection_route(dataset: str, target: str, method: str, budget: int, source: str = "source", pathways_path: str | Path = "configs/pathways.json") -> dict:
    spec = dataset_spec(dataset, pathways_path)
    expected = DATASET_METHODS[dataset]
    if target not in expected["targets"]:
        raise ValueError(target)
    if budget not in BUDGETS:
        raise ValueError(str(budget))
    if method not in expected["selection"]:
        raise ValueError(method)
    route = {"dataset": dataset, "target": target, "method": method, "budget": budget}
    if method == "random":
        route.update({"route_type": "split_file", "path": f"splits/{dataset}/{target}/random/random_{budget}.txt"})
        return route
    if method in SCORE_METHODS_8D:
        route.update({"route_type": "score_file", "command": selection_command(dataset, target, method, budget)})
        return route
    entrypoints = spec.get("selection_entrypoints", {})
    patterns = spec.get("selection_config_patterns", {})
    if method in entrypoints:
        route.update({"route_type": "entrypoint", "entrypoint": entrypoints[method]})
        if method in patterns:
            route["config"] = render_pattern(patterns[method], target, source, budget)
        return route
    if method in patterns:
        route.update({"route_type": "config_pattern", "config": render_pattern(patterns[method], target, source, budget)})
        return route
    raise ValueError(method)


def route_inventory(dataset: str = "all", pathways_path: str | Path = "configs/pathways.json") -> list[dict]:
    datasets = DATASET_METHODS if dataset == "all" else {dataset: DATASET_METHODS[dataset]}
    routes = []
    for dataset_key, spec in datasets.items():
        for target in spec["targets"]:
            for budget in BUDGETS:
                for method in spec["selection"]:
                    routes.append(selection_route(dataset_key, target, method, budget, pathways_path=pathways_path))
    return routes


def route_audit(pathways_path: str | Path = "configs/pathways.json") -> dict:
    errors = []
    routes = route_inventory("all", pathways_path=pathways_path)
    expected = 0
    seen = set()
    for dataset, spec in DATASET_METHODS.items():
        expected += len(spec["targets"]) * len(BUDGETS) * len(spec["selection"])
        for target in spec["targets"]:
            for budget in BUDGETS:
                for method in spec["selection"]:
                    seen.add((dataset, target, method, budget))
    observed = {(route["dataset"], route["target"], route["method"], route["budget"]) for route in routes}
    missing = sorted(seen - observed)
    extra = sorted(observed - seen)
    if missing:
        errors.append({"missing": missing})
    if extra:
        errors.append({"extra": extra})
    if len(routes) != expected:
        errors.append({"count": len(routes), "expected": expected})
    return {"ok": not errors, "count": len(routes), "expected": expected, "errors": errors}
