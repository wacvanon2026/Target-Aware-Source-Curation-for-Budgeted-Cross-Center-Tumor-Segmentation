from __future__ import annotations
SCORE_METHODS_8D = ('rds', 'less', 'orient', 'craig', 'gradmatch', 'kmeans', 'kcenter', 'diversity')
BUDGETS = (50, 150, 250)
DATASET_METHODS = {'mamamia': {'name': 'MAMA-MIA', 'task': 'tumor_segmentation', 'training': 'nnunet', 'selection': ('random', *SCORE_METHODS_8D), 'tavo': ('tavo_8d_cmaes',), 'domain_adaptation': ('dann', 'mmd', 'advent', 'seasa'), 'targets': ('NACT', 'ISPY1', 'DUKE', 'ISPY2')}, 'brats': {'name': 'BraTS', 'task': 'tumor_segmentation', 'training': 'efficientvit', 'selection': ('random', *SCORE_METHODS_8D), 'tavo': ('tavo_8d_cmaes',), 'domain_adaptation': ('dann', 'mmd', 'advent', 'seasa'), 'targets': ('C5', 'TCGA_LGG', 'TCGA_GBM', 'UPENN', 'IVYGAP')}, 'officehome': {'name': 'OfficeHome', 'task': 'image_classification', 'training': 'classification', 'selection': ('random', *SCORE_METHODS_8D), 'tavo': ('tavo_8d_cmaes',), 'domain_adaptation': ('dann', 'mmd', 'coral', 'cdan'), 'targets': ('Art', 'Clipart', 'Product', 'RealWorld')}}

def method_matrix() -> dict[str, dict[str, tuple | str]]:
    return {dataset: {'name': spec['name'], 'task': spec['task'], 'training': spec['training'], 'selection': spec['selection'], 'tavo': spec['tavo'], 'domain_adaptation': spec['domain_adaptation'], 'budgets': BUDGETS, 'targets': spec['targets']} for dataset, spec in DATASET_METHODS.items()}

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
                for budget in BUDGETS:
                    rows.append({'dataset': dataset, 'family': family, 'method': method, 'budget': budget})
    return rows
