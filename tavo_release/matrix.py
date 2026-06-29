from __future__ import annotations
SEG_SCORE_METHODS_8D = ('rds', 'less', 'orient', 'craig', 'gradmatch', 'kmeans', 'kcenter', 'diversity')
OFFICEHOME_SCORE_METHODS_8D = ('kmeans', 'kcenter', 'facilitylocation', 'craig', 'targetmmd', 'targetgradmatch', 'glister', 'orient')
SCORE_METHODS_8D = SEG_SCORE_METHODS_8D
SEG_BUDGETS = (50, 150, 250, 500)
OFFICEHOME_BUDGETS = (1, 3, 5, 8, 15, 25, 40)
BUDGETS = SEG_BUDGETS
DATASET_METHODS = {'mamamia': {'name': 'MAMA-MIA', 'task': 'tumor_segmentation', 'training': 'nnunet', 'selection': ('random', *SEG_SCORE_METHODS_8D), 'score_methods': SEG_SCORE_METHODS_8D, 'tavo': ('tavo_8d_cmaes',), 'domain_adaptation': ('dann', 'mmd', 'advent', 'seasa'), 'targets': ('NACT', 'ISPY1', 'DUKE', 'ISPY2'), 'budgets': SEG_BUDGETS}, 'brats': {'name': 'BraTS 2021', 'task': 'tumor_segmentation', 'training': 'efficientvit', 'selection': ('random', *SEG_SCORE_METHODS_8D), 'score_methods': SEG_SCORE_METHODS_8D, 'tavo': ('tavo_8d_cmaes',), 'domain_adaptation': ('dann', 'mmd', 'advent', 'seasa'), 'targets': ('C4', 'C5', 'TCGA_LGG', 'TCGA_GBM'), 'budgets': SEG_BUDGETS}, 'officehome': {'name': 'OfficeHome', 'task': 'image_classification', 'training': 'classification', 'selection': ('random', *OFFICEHOME_SCORE_METHODS_8D), 'score_methods': OFFICEHOME_SCORE_METHODS_8D, 'tavo': ('tavo_8d_cmaes',), 'domain_adaptation': ('dann', 'mmd', 'coral', 'cdan'), 'targets': ('Art', 'Clipart', 'Product', 'RealWorld'), 'budgets': OFFICEHOME_BUDGETS}}

def dataset_budgets(dataset: str) -> tuple[int, ...]:
    if dataset not in DATASET_METHODS:
        raise ValueError(dataset)
    return DATASET_METHODS[dataset]['budgets']

def dataset_score_methods(dataset: str) -> tuple[str, ...]:
    if dataset not in DATASET_METHODS:
        raise ValueError(dataset)
    return DATASET_METHODS[dataset]['score_methods']

def method_matrix() -> dict[str, dict[str, tuple | str]]:
    return {dataset: {'name': spec['name'], 'task': spec['task'], 'training': spec['training'], 'selection': spec['selection'], 'tavo': spec['tavo'], 'domain_adaptation': spec['domain_adaptation'], 'budgets': spec['budgets'], 'targets': spec['targets']} for dataset, spec in DATASET_METHODS.items()}

def dataset_methods(dataset: str, family: str) -> tuple[str, ...]:
    if dataset not in DATASET_METHODS:
        raise ValueError(dataset)
    values = DATASET_METHODS[dataset][family]
    if not isinstance(values, tuple):
        raise ValueError(f'{family} is not a method family')
    return values

def all_experiments() -> list[dict]:
    rows = []
    for dataset, spec in DATASET_METHODS.items():
        for family in ('selection', 'tavo', 'domain_adaptation'):
            for method in spec[family]:
                for budget in spec['budgets']:
                    rows.append({'dataset': dataset, 'family': family, 'method': method, 'budget': budget})
    return rows
