#!/usr/bin/env python3
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.parent.name == 'nnunet' and _SCRIPT_DIR.parent.parent.name == 'external':
    REPO_ROOT = _SCRIPT_DIR.parents[2]
else:
    REPO_ROOT = _SCRIPT_DIR.parents[1]

def project_root() -> Path:
    return Path(os.environ.get('MAMAMIA_PROJECT_ROOT', os.environ.get('PROJECT_ROOT', REPO_ROOT))).expanduser().resolve()

def dataset_root() -> Path:
    return Path(os.environ.get('MAMAMIA_DATASET_ROOT', os.environ.get('DATASET_ROOT', project_root() / 'data' / 'mamamia'))).expanduser().resolve()

def nnunet_root() -> Path:
    return Path(os.environ.get('NNUNET_ROOT', project_root() / 'outputs' / 'nnunet')).expanduser().resolve()

def nnunet_raw_root() -> Path:
    return Path(os.environ.get('nnUNet_raw', os.environ.get('NNUNET_RAW', nnunet_root() / 'nnUNet_raw'))).expanduser().resolve()

def nnunet_preprocessed_root() -> Path:
    return Path(os.environ.get('nnUNet_preprocessed', os.environ.get('NNUNET_PREPROCESSED', nnunet_root() / 'nnUNet_preprocessed'))).expanduser().resolve()

def nnunet_results_root() -> Path:
    return Path(os.environ.get('nnUNet_results', os.environ.get('NNUNET_RESULTS', nnunet_root() / 'nnUNet_results_scratch'))).expanduser().resolve()

def split_root() -> Path:
    return Path(os.environ.get('SPLIT_ROOT', REPO_ROOT / 'splits' / 'mamamia_lodo_seed42')).expanduser().resolve()
DOMAINS = {'NACT': 'NACT_', 'ISPY1': 'ISPY1_', 'DUKE': 'DUKE_', 'ISPY2': 'ISPY2_'}
BUDGETS = (50, 150, 250)
RATIO = {'train': 2, 'val': 1, 'test': 7}
TARGET_BASES = {'NACT': {'baseline': 1300, 'method': 1400}, 'ISPY1': {'baseline': 1310, 'method': 1500}, 'DUKE': {'baseline': 1320, 'method': 1600}, 'ISPY2': {'baseline': 1330, 'method': 1700}}
METHODS = ('rds', 'gradmatch', 'less', 'orient', 'diversity', 'kmeans', 'craig', 'kcenter')
METHOD_BUDGETS = (50, 150)
TAVO_BUDGET_OFFSETS = {50: 17, 150: 18, 250: 19}
EXTRA_METHOD_BUDGET_OFFSETS = {250: 20}

@dataclass(frozen=True)
class Experiment:
    key: str
    block: str
    offset: int
    suffix: str
    train_source: str
    description: str
    label: str
    method: str | None = None
    budget: int | None = None
EXPERIMENTS: dict[str, Experiment] = {'target_only': Experiment('target_only', 'baseline', 1, 'TARGET_ONLY', 'target_train.txt', 'target train only', 'Target-only'), 'source_only': Experiment('source_only', 'baseline', 2, 'SOURCE_ONLY', 'source_pool.txt', 'source pool only', 'Source-only'), 'target_full_source': Experiment('target_full_source', 'baseline', 3, 'TARGET_FULL_SOURCE', 'source_pool.txt+target_train.txt', 'target train plus full source pool', 'Target + full source'), 'random50': Experiment('random50', 'baseline', 4, 'RANDOM50_PLUS_TRAIN', 'random/random_50.txt+target_train.txt', 'random 50 plus target train', 'Random K50 + target train'), 'random150': Experiment('random150', 'baseline', 5, 'RANDOM150_PLUS_TRAIN', 'random/random_150.txt+target_train.txt', 'random 150 plus target train', 'Random K150 + target train'), 'random250': Experiment('random250', 'baseline', 6, 'RANDOM250_PLUS_TRAIN', 'random/random_250.txt+target_train.txt', 'random 250 plus target train', 'Random K250 + target train')}
_offset = 1
for _budget in METHOD_BUDGETS:
    for _method in METHODS:
        _key = f'{_method}{_budget}'
        EXPERIMENTS[_key] = Experiment(key=_key, block='method', offset=_offset, suffix=f'{_method.upper()}{_budget}_PLUS_TRAIN', train_source=f'methods/{_method}_{_budget}.txt+target_train.txt', description=f'{_method} {_budget} selected source cases plus target train', label=f'{_method.upper()} K{_budget} + target train', method=_method, budget=_budget)
        _offset += 1
for _budget, _start_offset in EXTRA_METHOD_BUDGET_OFFSETS.items():
    for _method_index, _method in enumerate(METHODS):
        _key = f'{_method}{_budget}'
        EXPERIMENTS[_key] = Experiment(key=_key, block='method', offset=_start_offset + _method_index, suffix=f'{_method.upper()}{_budget}_PLUS_TRAIN', train_source=f'methods/{_method}_{_budget}.txt+target_train.txt', description=f'{_method} {_budget} selected source cases plus target train', label=f'{_method.upper()} K{_budget} + target train', method=_method, budget=_budget)
for _budget, _offset in TAVO_BUDGET_OFFSETS.items():
    _key = f'tavo{_budget}'
    EXPERIMENTS[_key] = Experiment(key=_key, block='method', offset=_offset, suffix=f'TAVO{_budget}_PLUS_TRAIN', train_source=f'methods/tavo_{_budget}.txt+target_train.txt', description=f'TAVO {_budget} selected source cases plus target train', label=f'TAVO K{_budget} + target train', method='tavo', budget=_budget)
ALIASES = {'target': 'target_only', 'source': 'source_only', 'ext_only': 'source_only', 'target_plus_source': 'target_full_source', 'target_source': 'target_full_source', 'ext_plus_train': 'target_full_source', 'rand50': 'random50', 'k50': 'random50', 'rand150': 'random150', 'k150': 'random150', 'rand250': 'random250', 'k250': 'random250'}
for _method in METHODS:
    for _budget in (*METHOD_BUDGETS, *EXTRA_METHOD_BUDGET_OFFSETS):
        _canonical = f'{_method}{_budget}'
        ALIASES[f'{_method}_{_budget}'] = _canonical
        ALIASES[f'{_method}k{_budget}'] = _canonical
        ALIASES[f'{_method}_k{_budget}'] = _canonical
for _budget in TAVO_BUDGET_OFFSETS:
    _canonical = f'tavo{_budget}'
    ALIASES[f'tavo_{_budget}'] = _canonical
    ALIASES[f'tavok{_budget}'] = _canonical
    ALIASES[f'tavo_k{_budget}'] = _canonical
GROUPS = {'baselines': ('target_only', 'source_only', 'target_full_source', 'random50', 'random150', 'random250'), 'basic_methods': tuple((f'{method}{budget}' for budget in METHOD_BUDGETS for method in METHODS)), 'methods': tuple((f'{method}{budget}' for budget in METHOD_BUDGETS for method in METHODS)), 'k50_methods': tuple((f'{method}50' for method in METHODS)), 'k150_methods': tuple((f'{method}150' for method in METHODS)), 'k250_methods': tuple((f'{method}250' for method in METHODS)), 'tavo': tuple((f'tavo{budget}' for budget in TAVO_BUDGET_OFFSETS)), 'tavo_methods': tuple((f'tavo{budget}' for budget in TAVO_BUDGET_OFFSETS)), 'all_methods': tuple((f'{method}{budget}' for budget in METHOD_BUDGETS for method in METHODS)) + tuple((f'{method}{budget}' for budget in EXTRA_METHOD_BUDGET_OFFSETS for method in METHODS)) + tuple((f'tavo{budget}' for budget in TAVO_BUDGET_OFFSETS))}

def normalize_target(target: str) -> str:
    normalized = target.upper().replace('-', '')
    if normalized == 'ISPY1':
        return 'ISPY1'
    if normalized == 'ISPY2':
        return 'ISPY2'
    if normalized in TARGET_BASES:
        return normalized
    raise KeyError(target)

def normalize_experiment(key: str) -> str:
    lowered = key.lower()
    return ALIASES.get(lowered, lowered)

def expand_experiments(keys: list[str]) -> list[str]:
    if 'all' in keys:
        return list(EXPERIMENTS)
    expanded: list[str] = []
    for key in keys:
        normalized = key.lower()
        if normalized in GROUPS:
            expanded.extend(GROUPS[normalized])
            continue
        expanded.append(normalize_experiment(key))
    return expanded

def get_experiment(key: str) -> Experiment:
    normalized = normalize_experiment(key)
    return EXPERIMENTS[normalized]

def dataset_id(target: str, exp: Experiment) -> str:
    normalized_target = normalize_target(target)
    return str(TARGET_BASES[normalized_target][exp.block] + exp.offset)

def dataset_name(target: str, exp: Experiment) -> str:
    normalized_target = normalize_target(target)
    return f'MAMAMIA_{normalized_target}_LODO_SEED42_TAVO_{exp.suffix}_2d_3ch'

def dataset_basename(target: str, exp: Experiment) -> str:
    return f'Dataset{dataset_id(target, exp)}_{dataset_name(target, exp)}'
