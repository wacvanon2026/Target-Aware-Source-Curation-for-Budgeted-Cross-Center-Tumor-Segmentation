from __future__ import annotations
import json
from pathlib import Path
from .domain_adaptation import BRATS_ENTRYPOINTS, MAMAMIA_TRAINERS, OFFICEHOME_ENTRYPOINTS
from .matrix import BUDGETS, DATASET_METHODS, SCORE_METHODS_8D
REQUIRED_DATASETS = {'MAMA-MIA': 'mamamia', 'BraTS': 'brats', 'OfficeHome': 'officehome'}

def load_pathways(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    return list(data['pathways'])

def selection_route_present(spec: dict, method: str) -> bool:
    if method == 'random':
        return True
    for key in ('selection_entrypoints', 'selection_config_patterns'):
        if method in spec.get(key, {}):
            return True
    return method in spec.get('score_file_methods', [])

def domain_adaptation_route_present(spec: dict, method: str) -> bool:
    for key in ('domain_adaptation_entrypoints', 'domain_adaptation_trainers'):
        if method in spec.get(key, {}):
            return True
    return False

def tavo_route_present(spec: dict) -> bool:
    if not spec.get('tavo_methods'):
        return False
    if not spec.get('tavo_entrypoints'):
        return False
    score_methods = set(spec.get('score_file_methods', []))
    return set(SCORE_METHODS_8D).issubset(score_methods)

def expected_da_manifest(dataset_key: str) -> tuple[str, dict[str, str]]:
    if dataset_key == 'mamamia':
        return ('domain_adaptation_trainers', MAMAMIA_TRAINERS)
    if dataset_key == 'brats':
        return ('domain_adaptation_entrypoints', BRATS_ENTRYPOINTS)
    if dataset_key == 'officehome':
        return ('domain_adaptation_entrypoints', OFFICEHOME_ENTRYPOINTS)
    raise ValueError(dataset_key)

def compare_sequence(public_name: str, field: str, actual: list | tuple, expected: tuple) -> list[str]:
    errors = []
    actual_values = tuple(actual)
    missing = [value for value in expected if value not in actual_values]
    extra = [value for value in actual_values if value not in expected]
    if missing:
        errors.append(f'{public_name} missing {field}: {missing}')
    if extra:
        errors.append(f'{public_name} extra {field}: {extra}')
    if actual_values != expected:
        errors.append(f'{public_name} {field} order mismatch')
    return errors

def compare_route_keys(public_name: str, field: str, actual: dict, allowed: tuple[str, ...]) -> list[str]:
    extra = [value for value in actual if value not in allowed]
    if extra:
        return [f'{public_name} extra {field}: {extra}']
    return []

def audit_pathways(path: str | Path='configs/pathways.json') -> dict:
    path = Path(path)
    root = path.parent.parent if path.parent.name == 'configs' else Path('.')
    specs = load_pathways(path)
    errors = []
    seen = {spec.get('dataset'): spec for spec in specs}
    extra_datasets = [dataset for dataset in seen if dataset not in REQUIRED_DATASETS]
    if extra_datasets:
        errors.append(f'extra dataset pathways: {extra_datasets}')
    for public_name, dataset_key in REQUIRED_DATASETS.items():
        spec = seen.get(public_name)
        if spec is None:
            errors.append(f'missing dataset pathway: {public_name}')
            continue
        expected = DATASET_METHODS[dataset_key]
        config = spec.get('config')
        if not config or not (root / config).exists():
            errors.append(f'{public_name} config missing: {config}')
        if tuple(spec.get('budgets', [])) != BUDGETS:
            errors.append(f'{public_name} budgets mismatch')
        errors.extend(compare_sequence(public_name, 'targets', spec.get('targets', []), expected['targets']))
        for field, family in (('selection_methods', 'selection'), ('tavo_methods', 'tavo'), ('domain_adaptation_methods', 'domain_adaptation')):
            values = tuple(spec.get(field, []))
            errors.extend(compare_sequence(public_name, field, values, expected[family]))
            if not values:
                errors.append(f'{public_name} has empty {field}')
        if tuple(spec.get('score_file_methods', [])) != SCORE_METHODS_8D:
            errors.append(f'{public_name} score_file_methods mismatch')
        allowed_selection_routes = tuple((method for method in expected['selection'] if method != 'random'))
        errors.extend(compare_route_keys(public_name, 'selection_entrypoints', spec.get('selection_entrypoints', {}), allowed_selection_routes))
        errors.extend(compare_route_keys(public_name, 'selection_config_patterns', spec.get('selection_config_patterns', {}), allowed_selection_routes))
        for method in spec.get('selection_methods', []):
            if not selection_route_present(spec, method):
                errors.append(f'{public_name} selection route missing: {method}')
        missing_score_selections = [method for method in SCORE_METHODS_8D if method not in spec.get('selection_methods', [])]
        if missing_score_selections:
            errors.append(f'{public_name} missing 8D selection methods: {missing_score_selections}')
        for method in spec.get('domain_adaptation_methods', []):
            if not domain_adaptation_route_present(spec, method):
                errors.append(f'{public_name} domain adaptation route missing: {method}')
        route_key, route_values = expected_da_manifest(dataset_key)
        if spec.get(route_key, {}) != route_values:
            errors.append(f'{public_name} {route_key} mismatch')
        if not tavo_route_present(spec):
            errors.append(f'{public_name} TAVO route missing')
    return {'ok': not errors, 'errors': errors, 'datasets': sorted(seen)}
