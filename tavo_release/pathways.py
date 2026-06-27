from __future__ import annotations

import json
from pathlib import Path

from .matrix import BUDGETS, DATASET_METHODS


REQUIRED_DATASETS = {"MAMA-MIA": "mamamia", "BraTS": "brats", "OfficeHome": "officehome"}


def load_pathways(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return list(data["pathways"])


def route_present(spec: dict, method: str) -> bool:
    if method == "random":
        return True
    for key in ("selection_entrypoints", "selection_config_patterns", "domain_adaptation_entrypoints", "domain_adaptation_trainers", "tavo_entrypoints"):
        value = spec.get(key, {})
        if isinstance(value, dict) and method in value:
            return True
        if isinstance(value, list) and value:
            return True
    return method in spec.get("score_file_methods", [])


def audit_pathways(path: str | Path = "configs/pathways.json") -> dict:
    specs = load_pathways(path)
    errors = []
    seen = {spec.get("dataset"): spec for spec in specs}
    for public_name, dataset_key in REQUIRED_DATASETS.items():
        spec = seen.get(public_name)
        if spec is None:
            errors.append(f"missing dataset pathway: {public_name}")
            continue
        expected = DATASET_METHODS[dataset_key]
        if tuple(spec.get("budgets", [])) != BUDGETS:
            errors.append(f"{public_name} budgets mismatch")
        for field, family in (("selection_methods", "selection"), ("tavo_methods", "tavo"), ("domain_adaptation_methods", "domain_adaptation")):
            values = tuple(spec.get(field, []))
            missing = [method for method in expected[family] if method not in values]
            if missing:
                errors.append(f"{public_name} missing {field}: {missing}")
            if not values:
                errors.append(f"{public_name} has empty {field}")
        missing_targets = [target for target in expected["targets"] if target not in spec.get("targets", [])]
        if missing_targets:
            errors.append(f"{public_name} missing targets: {missing_targets}")
        for method in spec.get("selection_methods", []):
            if not route_present(spec, method):
                errors.append(f"{public_name} selection route missing: {method}")
        for method in spec.get("domain_adaptation_methods", []):
            if not route_present(spec, method):
                errors.append(f"{public_name} domain adaptation route missing: {method}")
        if not spec.get("tavo_methods"):
            errors.append(f"{public_name} TAVO route missing")
    return {"ok": not errors, "errors": errors, "datasets": sorted(seen)}
