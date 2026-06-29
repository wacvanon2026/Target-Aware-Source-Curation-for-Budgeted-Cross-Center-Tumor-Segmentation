from __future__ import annotations
from pathlib import Path
from .domain_adaptation import implementation
from .matrix import DATASET_METHODS, dataset_budgets, dataset_score_methods
from .pathways import REQUIRED_DATASETS, load_pathways
from .tavo_routes import search_command, selection_command
KEY_TO_PUBLIC = {value: key for key, value in REQUIRED_DATASETS.items()}

def dataset_spec(dataset: str, pathways_path: str | Path='configs/pathways.json') -> dict:
    public_name = KEY_TO_PUBLIC.get(dataset)
    if public_name is None:
        raise ValueError(dataset)
    for spec in load_pathways(pathways_path):
        if spec.get('dataset') == public_name:
            return spec
    raise ValueError(dataset)

def render_pattern(pattern: str, target: str, source: str, budget: int) -> str:
    return pattern.replace('TARGET', target).replace('SOURCE', source).replace('BUDGET', str(budget))

def selection_route(dataset: str, target: str, method: str, budget: int, source: str='source', pathways_path: str | Path='configs/pathways.json') -> dict:
    spec = dataset_spec(dataset, pathways_path)
    expected = DATASET_METHODS[dataset]
    if target not in expected['targets']:
        raise ValueError(target)
    if budget not in dataset_budgets(dataset):
        raise ValueError(str(budget))
    if method not in expected['selection']:
        raise ValueError(method)
    route = {'dataset': dataset, 'target': target, 'method': method, 'budget': budget}
    if method == 'random':
        route.update({'route_type': 'split_file', 'path': f'{selection_split_dir(dataset, target)}/random/random_{budget}.txt'})
        return route
    if method in dataset_score_methods(dataset):
        out = f'{selection_split_dir(dataset, target)}/methods/{method}_{budget}.txt'
        route.update({'route_type': 'score_file', 'command': selection_command(dataset, target, method, budget, output=out)})
        return route
    entrypoints = spec.get('selection_entrypoints', {})
    patterns = spec.get('selection_config_patterns', {})
    if method in entrypoints:
        route.update({'route_type': 'entrypoint', 'entrypoint': entrypoints[method]})
        if method in patterns:
            route['config'] = render_pattern(patterns[method], target, source, budget)
        return route
    if method in patterns:
        route.update({'route_type': 'config_pattern', 'config': render_pattern(patterns[method], target, source, budget)})
        return route
    raise ValueError(method)

def selection_inventory(dataset: str='all', pathways_path: str | Path='configs/pathways.json') -> list[dict]:
    datasets = DATASET_METHODS if dataset == 'all' else {dataset: DATASET_METHODS[dataset]}
    routes = []
    for dataset_key, spec in datasets.items():
        for target in spec['targets']:
            for budget in spec['budgets']:
                for method in spec['selection']:
                    routes.append(selection_route(dataset_key, target, method, budget, pathways_path=pathways_path))
    return routes

def route_inventory(dataset: str='all', pathways_path: str | Path='configs/pathways.json', family: str='selection') -> list[dict]:
    return family_inventory(family, dataset=dataset, pathways_path=pathways_path)

def tavo_route(dataset: str, target: str, budget: int) -> dict:
    expected = DATASET_METHODS[dataset]
    if target not in expected['targets']:
        raise ValueError(target)
    if budget not in dataset_budgets(dataset):
        raise ValueError(str(budget))
    return {'dataset': dataset, 'target': target, 'method': 'tavo_8d_cmaes', 'budget': budget, 'route_type': 'score_fusion', 'command': search_command(dataset, target, budget)}

def mamamia_da_dataset_id(target: str, method: str, budget: int) -> str:
    target_offset = DATASET_METHODS['mamamia']['targets'].index(target) * 100
    method_offset = DATASET_METHODS['mamamia']['domain_adaptation'].index(method) * 10
    budget_offset = dataset_budgets('mamamia').index(budget)
    return str(9000 + target_offset + method_offset + budget_offset)

def split_dir(dataset: str, target: str) -> str:
    if dataset == 'mamamia':
        return f'splits/mamamia_lodo_seed42/{target}'
    return f'splits/{dataset}/{target}'

def selection_split_dir(dataset: str, target: str) -> str:
    if dataset == 'mamamia':
        return f'splits/mamamia_lodo_seed42/{target}'
    return f'splits/{dataset}/{target}'

def domain_adaptation_route(dataset: str, target: str, method: str, budget: int) -> dict:
    expected = DATASET_METHODS[dataset]
    if target not in expected['targets']:
        raise ValueError(target)
    if budget not in dataset_budgets(dataset):
        raise ValueError(str(budget))
    if method not in expected['domain_adaptation']:
        raise ValueError(method)
    route = {'dataset': dataset, 'target': target, 'method': method, 'budget': budget, 'route_type': 'domain_adaptation', 'implementation': implementation(dataset, method)}
    cfg = f'configs/generated/{dataset}_{target}_{method}_{budget}.json'
    config_command = ['python', '-m', 'tavo_release.cli', 'da-config', '--dataset', dataset, '--method', method, '--split-dir', split_dir(dataset, target), '--output-dir', f'outputs/{dataset}/{target}/{method}{budget}', '--budget', str(budget), '--output', cfg, '--target', target]
    if dataset == 'mamamia':
        config_command.extend(['--nnunet-dataset-id', mamamia_da_dataset_id(target, method, budget)])
    route['config_command'] = config_command
    route['train_command'] = ['python', '-m', 'tavo_release.cli', 'da-command', '--config', cfg]
    return route

def family_inventory(family: str, dataset: str='all', pathways_path: str | Path='configs/pathways.json') -> list[dict]:
    if family == 'selection':
        return selection_inventory(dataset, pathways_path=pathways_path)
    datasets = DATASET_METHODS if dataset == 'all' else {dataset: DATASET_METHODS[dataset]}
    routes = []
    for dataset_key, spec in datasets.items():
        for target in spec['targets']:
            for budget in spec['budgets']:
                if family == 'tavo':
                    routes.append(tavo_route(dataset_key, target, budget))
                elif family == 'domain_adaptation':
                    for method in spec['domain_adaptation']:
                        routes.append(domain_adaptation_route(dataset_key, target, method, budget))
                else:
                    raise ValueError(family)
    return routes

def expected_keys(family: str) -> set[tuple]:
    keys = set()
    for dataset, spec in DATASET_METHODS.items():
        for target in spec['targets']:
            for budget in spec['budgets']:
                if family == 'selection':
                    for method in spec['selection']:
                        keys.add((dataset, target, method, budget))
                elif family == 'tavo':
                    for method in spec['tavo']:
                        keys.add((dataset, target, method, budget))
                elif family == 'domain_adaptation':
                    for method in spec['domain_adaptation']:
                        keys.add((dataset, target, method, budget))
                else:
                    raise ValueError(family)
    return keys

def observed_keys(routes: list[dict]) -> set[tuple]:
    return {(route['dataset'], route['target'], route['method'], route['budget']) for route in routes}

def command_value(command: list[str], flag: str) -> str | None:
    if flag not in command:
        return None
    index = command.index(flag) + 1
    if index >= len(command):
        return None
    return command[index]

def route_command_errors(family: str, route: dict) -> list[dict]:
    errors = []
    prefix = {'dataset': route['dataset'], 'target': route['target'], 'method': route['method'], 'budget': route['budget']}
    if family == 'selection' and route['method'] == 'random':
        expected_path = f"{selection_split_dir(route['dataset'], route['target'])}/random/random_{route['budget']}.txt"
        if route.get('path') != expected_path:
            errors.append({**prefix, 'error': 'random_split_path'})
    if family == 'selection' and route['method'] in dataset_score_methods(route['dataset']):
        command = route.get('command', [])
        if command.count('--score') != len(dataset_score_methods(route['dataset'])):
            errors.append({**prefix, 'error': 'selection_score_count'})
        if '--weight' not in command:
            errors.append({**prefix, 'error': 'selection_missing_weight'})
        else:
            start = command.index('--weight') + 1
            weights = command[start:start + len(dataset_score_methods(route['dataset']))]
            if len(weights) != len(dataset_score_methods(route['dataset'])) or weights.count('1') != 1:
                errors.append({**prefix, 'error': 'selection_weight_shape'})
        if command_value(command, '--budget') != str(route['budget']):
            errors.append({**prefix, 'error': 'selection_budget'})
        expected_output = f"{selection_split_dir(route['dataset'], route['target'])}/methods/{route['method']}_{route['budget']}.txt"
        if command_value(command, '--output') != expected_output:
            errors.append({**prefix, 'error': 'selection_output'})
    if family == 'tavo':
        command = route.get('command', [])
        if command.count('--score') != len(dataset_score_methods(route['dataset'])):
            errors.append({**prefix, 'error': 'tavo_score_count'})
        if command_value(command, '--budget') != str(route['budget']):
            errors.append({**prefix, 'error': 'tavo_budget'})
    if family == 'domain_adaptation':
        command = route.get('config_command', [])
        if command_value(command, '--target') != route['target']:
            errors.append({**prefix, 'error': 'da_target'})
        if command_value(command, '--budget') != str(route['budget']):
            errors.append({**prefix, 'error': 'da_budget'})
        if route['dataset'] == 'mamamia' and '--nnunet-dataset-id' not in command:
            errors.append({**prefix, 'error': 'da_nnunet_dataset_id'})
    return errors

def route_audit(pathways_path: str | Path='configs/pathways.json') -> dict:
    families = {}
    ok = True
    for family in ('selection', 'tavo', 'domain_adaptation'):
        routes = family_inventory(family, pathways_path=pathways_path)
        expected = expected_keys(family)
        observed = observed_keys(routes)
        errors = []
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        if missing:
            errors.append({'missing': missing})
        if extra:
            errors.append({'extra': extra})
        if len(routes) != len(expected):
            errors.append({'count': len(routes), 'expected': len(expected)})
        command_errors = []
        for route in routes:
            command_errors.extend(route_command_errors(family, route))
        if command_errors:
            errors.append({'command_errors': command_errors})
        family_ok = not errors
        ok = ok and family_ok
        families[family] = {'ok': family_ok, 'count': len(routes), 'expected': len(expected), 'errors': errors}
    return {'ok': ok, 'families': families}
