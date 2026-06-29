#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs-root', type=Path, default=Path('configs_TAVO'))
    parser.add_argument('--outputs-root', type=Path, default=Path('outputs_TAVO'))
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    parser.add_argument('--checkpoint', action='append', default=None, help='Checkpoint name to evaluate. Defaults to last.pt to match the reported final-epoch results.')
    parser.add_argument('--include', action='append', default=[], help='Regex filter applied to config and output paths.')
    parser.add_argument('--exclude', action='append', default=[], help='Regex exclusion filter.')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--plan-output', type=Path, default=Path('analysis/brats_metrics_backfill_plan.csv'))
    parser.add_argument('--summary-output', type=Path, default=Path('analysis/brats_metrics_backfill_summary.json'))
    return parser.parse_args()

def load_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with path.open('r') as f:
            return yaml.safe_load(f)
    except Exception as exc:
        print(f'Skipping unreadable config {path}: {exc}', file=sys.stderr)
        return None

def path_text(path: Path) -> str:
    return str(path).replace('\\', '/')

def matches_filters(text: str, includes: list[str], excludes: list[str]) -> bool:
    if includes and (not any((re.search(pat, text) for pat in includes))):
        return False
    if excludes and any((re.search(pat, text) for pat in excludes)):
        return False
    return True

def infer_target_domain(*parts: str) -> str | None:
    joined = ' '.join([p for p in parts if p])
    for name in ('TCGA_LGG', 'TCGA_GBM', 'C4', 'C5', 'UPENN', 'IVYGAP', 'TCGA'):
        if re.search(f'(^|[^A-Za-z0-9]){re.escape(name)}([^A-Za-z0-9]|$)', joined):
            return name
    return None

def infer_budget(*parts: str) -> int | None:
    joined = ' '.join([p for p in parts if p])
    m = re.search('(...<![A-Za-z0-9])K(\\d+)(...![A-Za-z0-9])', joined, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search('(...<![A-Za-z0-9])(\\d+)T(...![A-Za-z0-9])', joined, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)) * 50
    return None

def infer_method(*parts: str) -> str | None:
    joined = ' '.join([p.lower() for p in parts if p])
    if '8d' in joined and ('cma' in joined or 'stagea2' in joined):
        return 'TAVO_8D'
    if '3d' in joined and ('cma' in joined or 'stagea2' in joined):
        return 'TAVO_3D'
    checks = [('rdsplus_kmeans', 'RDSPlus_KMeans'), ('rdsplus', 'RDSPlus'), ('gradmatch', 'GradMatch'), ('diversity', 'Diversity'), ('kcenter', 'KCenter'), ('kmeans', 'KMeans'), ('orient', 'ORIENT'), ('random3', 'Random3'), ('random2', 'Random2'), ('random1', 'Random1'), ('random', 'Random'), ('craig', 'CRAIG'), ('less', 'LESS'), ('target_train', 'TargetOnly'), ('s_and_t', 'FullSourceTarget'), ('source_plus_target', 'FullSourceTarget'), ('source', 'SourceOnly')]
    for needle, method in checks:
        if needle in joined:
            return method
    return None

def infer_subset_seed(*parts: str) -> str | None:
    joined = ' '.join([p for p in parts if p])
    m = re.search('(...:repeat|random)(\\d+)', joined, flags=re.IGNORECASE)
    return m.group(1) if m else None

def resolve_output_dir(project_root: Path, outputs_root: Path, save_dir_text: str) -> Path | None:
    if not save_dir_text:
        return None
    raw = Path(save_dir_text)
    run_name = raw.name
    parent_name = raw.parent.name
    candidates = [raw, project_root / raw, outputs_root / parent_name / run_name, project_root / outputs_root / parent_name / run_name, project_root / parent_name / run_name]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return outputs_root / parent_name / run_name

def build_run_plan(args: argparse.Namespace) -> list[dict[str, Any]]:
    project_root = args.project_root.resolve()
    configs_root = (project_root / args.configs_root).resolve()
    outputs_root = (project_root / args.outputs_root).resolve()
    plan: list[dict[str, Any]] = []
    for config in sorted(configs_root.rglob('*.yaml')):
        cfg = load_yaml(config)
        if not cfg:
            continue
        save_dir_text = str(cfg.get('training', {}).get('save_dir', ''))
        if 'PLACEHOLDER' in save_dir_text:
            continue
        output_dir = resolve_output_dir(project_root, outputs_root, save_dir_text)
        if output_dir is None:
            continue
        context = ' '.join([path_text(config), save_dir_text, path_text(output_dir)])
        if not matches_filters(context, args.include, args.exclude):
            continue
        for checkpoint_name in args.checkpoint:
            checkpoint = output_dir / checkpoint_name
            metrics_dir = output_dir / 'metrics' / Path(checkpoint_name).stem
            per_case = metrics_dir / 'per_case_metrics.csv'
            should_run = checkpoint.exists() and (args.overwrite or not per_case.exists())
            row = {'config': path_text(config.relative_to(project_root)), 'checkpoint': path_text(checkpoint.relative_to(project_root)) if checkpoint.exists() else path_text(checkpoint), 'output_dir': path_text(output_dir.relative_to(project_root)) if output_dir.exists() else path_text(output_dir), 'metrics_dir': path_text(metrics_dir.relative_to(project_root)) if output_dir.exists() else path_text(metrics_dir), 'legacy_results': path_text((output_dir / '3d_eval_results.txt').relative_to(project_root)) if (output_dir / '3d_eval_results.txt').exists() else '', 'target_domain': infer_target_domain(context) or '', 'method': infer_method(context) or '', 'budget': infer_budget(context) or '', 'subset_seed': infer_subset_seed(context) or '', 'checkpoint_exists': checkpoint.exists(), 'metrics_exists': per_case.exists(), 'should_run': should_run}
            plan.append(row)
            if args.limit is not None and len([p for p in plan if p['should_run']]) >= args.limit:
                return plan
    return plan

def write_plan(path: Path, plan: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ['config', 'checkpoint', 'output_dir', 'metrics_dir', 'legacy_results', 'target_domain', 'method', 'budget', 'subset_seed', 'checkpoint_exists', 'metrics_exists', 'should_run']
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(plan)

def run_one(project_root: Path, row: dict[str, Any], device: str) -> int:
    evaluator = project_root / 'eval' / 'evaluate_2d_to_3d_metrics.py'
    cmd = [sys.executable, str(evaluator), '--config', row['config'], '--checkpoint', row['checkpoint'], '--metrics-dir', row['metrics_dir'], '--checkpoint-name', Path(row['checkpoint']).stem, '--device', device]
    if row.get('legacy_results'):
        cmd.extend(['--legacy-results', row['legacy_results']])
    for key, flag in (('target_domain', '--target-domain'), ('method', '--method'), ('budget', '--budget'), ('subset_seed', '--subset-seed')):
        value = row.get(key)
        if value != '' and value is not None:
            cmd.extend([flag, str(value)])
    print('\nRunning:')
    print(' '.join(cmd))
    return subprocess.run(cmd, cwd=project_root).returncode

def main() -> None:
    args = parse_args()
    if args.checkpoint is None:
        args.checkpoint = ['last.pt']
    project_root = args.project_root.resolve()
    plan = build_run_plan(args)
    write_plan(args.plan_output, plan)
    runnable = [row for row in plan if row['should_run']]
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'plan_output': str(args.plan_output), 'total_pairs': len(plan), 'runnable_pairs': len(runnable), 'dry_run': args.dry_run, 'limit': args.limit}
    print(f'Plan rows: {len(plan)}')
    print(f'Runnable rows: {len(runnable)}')
    print(f'Plan saved to: {args.plan_output}')
    failures = []
    if not args.dry_run:
        for row in runnable:
            code = run_one(project_root, row, args.device)
            if code != 0:
                failures.append({'row': row, 'returncode': code})
                break
    summary['failures'] = failures
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open('w') as f:
        json.dump(summary, f, indent=2)
    print(f'Summary saved to: {args.summary_output}')
    if failures:
        sys.exit(1)
if __name__ == '__main__':
    main()
