#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from core import METHODS, project_root
PROJECT_ROOT = project_root()
TARGETS = ('NACT', 'ISPY1', 'DUKE', 'ISPY2')
BASELINE_EXPERIMENTS = (('target_only', 'Target-only'), ('source_only', 'Source-only Full'), ('target_full_source', 'Target + Full Source'))
DOMAIN_ALIGNMENT_METHODS = (('dann', 'DANN'), ('mmd', 'MMD / DAN'), ('advent', 'ADVENT'), ('seasa', 'SE-ASA'))
RANDOM_EXPERIMENTS = {50: (('random50', 'Random'),), 150: (('random150', 'Random'),), 250: (('random250', 'Random'),)}
DOMAIN_ALIGNMENT_BUDGET_EXPERIMENTS = {budget: tuple(((f'random{budget}_{method}', label) for method, label in DOMAIN_ALIGNMENT_METHODS)) for budget in (50, 150, 250)}
DOMAIN_ALIGNMENT_EXPERIMENT_KEYS = {experiment for experiments in DOMAIN_ALIGNMENT_BUDGET_EXPERIMENTS.values() for experiment, _ in experiments}
METHOD_LABELS = {'rds': 'RDS', 'gradmatch': 'GradMatch', 'less': 'LESS', 'orient': 'ORIENT', 'diversity': 'Diversity', 'kmeans': 'KMeans', 'craig': 'CRAIG', 'kcenter': 'KCenter'}
METHOD_BUDGETS = (50, 150)
OLD_RESULTS = {'target_only': {'NACT': 0.474, 'ISPY1': 0.5788, 'DUKE': 0.5103, 'ISPY2': 0.6126}, 'source_only': {'NACT': 0.686, 'ISPY1': 0.7087, 'DUKE': 0.5776, 'ISPY2': 0.6686}, 'target_full_source': {'NACT': 0.681, 'ISPY1': 0.7158, 'DUKE': 0.5302, 'ISPY2': 0.6943}, 'rds250': {'NACT': 0.6938, 'ISPY1': 0.7107, 'DUKE': 0.6338, 'ISPY2': 0.6914}, 'gradmatch250': {'NACT': 0.6904, 'ISPY1': 0.717, 'DUKE': 0.6393, 'ISPY2': 0.6906}, 'less250': {'NACT': 0.7086, 'ISPY1': 0.7202, 'DUKE': 0.631, 'ISPY2': 0.7015}, 'orient250': {'NACT': 0.6167, 'ISPY1': 0.7093, 'DUKE': 0.6127, 'ISPY2': 0.6925}, 'diversity250': {'NACT': 0.6417, 'ISPY1': 0.7064, 'DUKE': 0.5896, 'ISPY2': 0.6827}, 'kmeans250': {'NACT': 0.6744, 'ISPY1': 0.7206, 'DUKE': 0.612, 'ISPY2': 0.6851}, 'craig250': {'NACT': 0.5497, 'ISPY1': 0.6965, 'DUKE': 0.5984, 'ISPY2': 0.6971}, 'kcenter250': {'NACT': 0.6492, 'ISPY1': 0.7175, 'DUKE': 0.625, 'ISPY2': 0.6826}, 'tavo250': {'NACT': 0.7163, 'ISPY1': 0.7252, 'DUKE': 0.6255, 'ISPY2': 0.7002}}

def summary_path(project_root: Path, target: str, experiment: str, metric: str='Dice') -> Path:
    target_lc = target.lower()
    summary_name = 'summary_hd95.json' if metric.upper() == 'HD95' else 'summary.json'
    return project_root / 'outputs' / f'tavo_mamamia_{target_lc}_nnunet_{experiment}' / 'repeat_01' / 'test_preds' / summary_name

def metric_key(metric: str) -> str:
    normalized = metric.upper()
    if normalized == 'DICE':
        return 'Dice'
    if normalized == 'HD95':
        return 'HD95'
    return metric

def read_foreground_metric(path: Path, metric: str) -> float | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    value = data.get('foreground_mean', {}).get(metric_key(metric))
    return None if value is None else float(value)

def read_foreground_dice(path: Path) -> float | None:
    return read_foreground_metric(path, 'Dice')

def first_foreground_metric(project_roots: tuple[Path, ...], target: str, experiment: str, metric: str) -> float | None:
    for root in project_roots:
        value = read_foreground_metric(summary_path(root, target, experiment, metric), metric)
        if value is not None:
            return value
    return None

def fmt(value: float | None) -> str:
    return '' if value is None else f'{value:.4f}'

def fmt_value_pair(new_value: float | None, old_value: float | None) -> str:
    if new_value is not None and old_value is not None:
        return f'new: {new_value:.4f}<br>old: {old_value:.4f}'
    if new_value is not None:
        return f'new: {new_value:.4f}'
    if old_value is not None:
        return f'old: {old_value:.4f}'
    return ''

def fmt_cell(experiment: str, target: str, new_value: float | None, old_value: float | None, metric: str) -> str:
    if metric.upper() != 'DICE':
        return fmt(new_value)
    return fmt_value_pair(new_value, old_value)

def collect_rows(project_roots: tuple[Path, ...], experiments: tuple[tuple[str, str], ...], include_domain_alignment_results: bool=False, metric: str='Dice') -> list[dict[str, str]]:
    rows = []
    metric = metric.upper()
    for experiment, label in experiments:
        if experiment in DOMAIN_ALIGNMENT_EXPERIMENT_KEYS and (not include_domain_alignment_results):
            values = {target: None for target in TARGETS}
        else:
            values = {target: first_foreground_metric(project_roots, target, experiment, metric) for target in TARGETS}
        old_values = OLD_RESULTS.get(experiment, {}) if metric == 'DICE' else {}
        completed = [value for value in values.values() if value is not None]
        old_completed = [old_values[target] for target in TARGETS if target in old_values]
        avg = sum(completed) / len(completed) if completed else None
        old_avg = sum(old_completed) / len(old_completed) if old_completed else None
        rows.append({'method': label, 'experiment': experiment, **{target: fmt_cell(experiment, target, values[target], old_values.get(target), metric) for target in TARGETS}, 'Avg': fmt_value_pair(avg, old_avg) if metric == 'DICE' else fmt(avg), 'completed': str(len(completed) if completed else len(old_completed))})
    return rows

def collect(project_root: Path, include_domain_alignment_results: bool=False, metric: str='Dice', extra_project_roots: tuple[Path, ...]=()) -> dict[str, list[dict[str, str]]]:
    project_roots = (project_root, *extra_project_roots)
    sections = {'baselines': collect_rows(project_roots, BASELINE_EXPERIMENTS, metric=metric)}
    for budget in (50, 150, 250):
        rows = collect_rows(project_roots, RANDOM_EXPERIMENTS[budget], metric=metric)
        if budget in METHOD_BUDGETS:
            method_experiments = tuple(((f'{method}{budget}', METHOD_LABELS[method]) for method in METHODS)) + ((f'tavo{budget}', 'TAVO'),)
            rows.extend(collect_rows(project_roots, method_experiments, metric=metric))
        elif budget == 250:
            method_experiments = tuple(((f'{method}250', METHOD_LABELS[method]) for method in METHODS)) + (('tavo250', 'TAVO'),)
            rows.extend(collect_rows(project_roots, method_experiments, metric=metric))
        sections[f'k{budget}'] = rows
        sections[f'da_k{budget}'] = collect_rows(project_roots, DOMAIN_ALIGNMENT_BUDGET_EXPERIMENTS[budget], include_domain_alignment_results, metric)
    return sections

def print_markdown(rows: list[dict[str, str]]) -> None:
    headers = ('method', 'NACT', 'ISPY1', 'DUKE', 'ISPY2', 'Avg', 'completed')
    print('| ' + ' | '.join(headers) + ' |')
    print('|' + '|'.join(('---' for _ in headers)) + '|')
    for row in rows:
        print('| ' + ' | '.join((row[column] for column in headers)) + ' |')

def print_report(sections: dict[str, list[dict[str, str]]]) -> None:
    print('### Source-only / full-source baselines')
    print()
    print_markdown(sections['baselines'])
    for budget in (50, 150, 250):
        print()
        print(f'### Source-selection budget: K{budget}')
        print()
        print_markdown(sections[f'k{budget}'])
    print()
    print('## Domain Alignment')
    for budget in (50, 150, 250):
        print()
        print(f'### Domain alignment budget: K{budget}')
        print()
        print_markdown(sections[f'da_k{budget}'])

def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['section', 'method', 'experiment', 'NACT', 'ISPY1', 'DUKE', 'ISPY2', 'Avg', 'completed']
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

def flatten_sections(sections: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    rows = []
    for section, section_rows in sections.items():
        for row in section_rows:
            rows.append({'section': section, **row})
    return rows

def numeric_from_cell(cell: str) -> float | None:
    if not cell:
        return None
    match = re.search('new:\\s*([0-9.]+)', cell)
    if match:
        return float(match.group(1))
    try:
        return float(cell)
    except ValueError:
        return None

def refresh_row_average(row: dict[str, str], metric: str) -> None:
    values = [numeric_from_cell(row[target]) for target in TARGETS]
    values = [value for value in values if value is not None]
    if not values:
        return
    avg = sum(values) / len(values)
    row['Avg'] = f'new: {avg:.4f}' if metric.upper() == 'DICE' else f'{avg:.4f}'
    row['completed'] = str(len(values))

def preserve_existing_domain_alignment_cells(sections: dict[str, list[dict[str, str]]], existing_csv: Path, metric: str) -> None:
    if not existing_csv.exists():
        return
    existing: dict[tuple[str, str], dict[str, str]] = {}
    with existing_csv.open(newline='') as handle:
        for row in csv.DictReader(handle):
            section = row.get('section', '')
            experiment = row.get('experiment', '')
            if section.startswith('da_') and experiment:
                existing[section, experiment] = row
    for section, rows in sections.items():
        if not section.startswith('da_'):
            continue
        for row in rows:
            previous = existing.get((section, row['experiment']))
            if previous is None:
                continue
            changed = False
            for target in TARGETS:
                if not row[target] and previous.get(target):
                    row[target] = previous[target]
                    changed = True
            if changed:
                refresh_row_average(row, metric)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--extra-project-root', type=Path, action='append', default=[], help='Additional project root(s) to search after --project-root, useful for scratch DA runs.')
    parser.add_argument('--csv', type=Path, help='Optional CSV output path.')
    parser.add_argument('--markdown', type=Path, help='Optional Markdown output path.')
    parser.add_argument('--metric', choices=('Dice', 'HD95'), default='Dice')
    parser.add_argument('--append-hd95', action='store_true', help='When writing a Dice Markdown report, append separate HD95 tables after the Dice tables.')
    parser.add_argument('--include-domain-alignment-results', action='store_true', help='Fill DA rows from existing summaries. Default keeps DA rows blank because they are run externally.')
    args = parser.parse_args()
    project_root = args.project_root.expanduser().resolve()
    extra_project_roots = tuple((path.expanduser().resolve() for path in args.extra_project_root))
    sections = collect(project_root, args.include_domain_alignment_results, args.metric, extra_project_roots)
    if args.include_domain_alignment_results and args.csv:
        preserve_existing_domain_alignment_cells(sections, args.csv.expanduser().resolve(), args.metric)
    print_report(sections)
    if args.csv:
        write_csv(flatten_sections(sections), args.csv.expanduser().resolve())
    if args.markdown:
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            print('# MAMAMIA nnUNet LODO Results')
            print()
            print(f"Collected from `{project_root / 'outputs'}`.")
            if extra_project_roots:
                print('Additional result roots: ' + ', '.join((f"`{root / 'outputs'}`" for root in extra_project_roots)) + '.')
            if args.metric == 'HD95':
                print('Values are foreground mean HD95 from each `test_preds/summary_hd95.json`; lower is better and blank cells are not complete yet.')
            else:
                print('Values are foreground mean Dice from each `test_preds/summary.json`; blank cells are not complete yet.')
            print()
            print_report(sections)
            if args.append_hd95 and args.metric == 'Dice':
                hd95_sections = collect(project_root, args.include_domain_alignment_results, 'HD95', extra_project_roots)
                print()
                print('## HD95')
                print()
                print('Values are foreground mean HD95 from each `test_preds/summary_hd95.json`; lower is better and blank cells are not complete yet.')
                print()
                print_report(hd95_sections)
        args.markdown.expanduser().resolve().write_text(buffer.getvalue())
if __name__ == '__main__':
    main()
