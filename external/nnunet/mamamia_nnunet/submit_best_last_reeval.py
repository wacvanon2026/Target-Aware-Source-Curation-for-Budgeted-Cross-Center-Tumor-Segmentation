#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path
from core import DOMAINS, EXPERIMENTS, project_root
from collect_results import summary_path
TARGETS = tuple(DOMAINS)
DEFAULT_EXPERIMENTS = ('target_only', 'source_only', 'target_full_source', 'random50', 'random150', 'random250', 'rds50', 'less50', 'diversity50', 'kmeans50', 'kcenter50', 'rds150', 'less150', 'diversity150', 'kmeans150', 'kcenter150')

def active_mamamia_jobs() -> set[str]:
    user = os.environ.get('USER') or subprocess.check_output(['whoami'], text=True).strip()
    out = subprocess.check_output(['squeue', '-u', user, '-h', '-o', '%j'], text=True)
    names = set()
    for raw in out.splitlines():
        name = raw.strip()
        if not name.startswith('mamamia_'):
            continue
        names.add(name[:-8] if name.endswith('_resume1') else name)
    return names

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', nargs='+', default=list(TARGETS), choices=list(TARGETS))
    parser.add_argument('--experiments', nargs='+', default=list(DEFAULT_EXPERIMENTS), choices=sorted(EXPERIMENTS))
    parser.add_argument('--account', default=os.environ.get('SBATCH_ACCOUNT', 'YOUR_SLURM_ACCOUNT'))
    parser.add_argument('--time', default=os.environ.get('SBATCH_TIME', '04:00:00'))
    parser.add_argument('--constraint', default=os.environ.get('SBATCH_CONSTRAINT', 'a100|a40|l40s|v100'))
    parser.add_argument('--window', type=int, default=int(os.environ.get('BEST_LAST_WINDOW', '10')))
    parser.add_argument('--max-submit', type=int, default=0, help='Maximum jobs to submit in this invocation; 0 means no limit.')
    parser.add_argument('--submit', action='store_true', help='Actually submit jobs. Default is dry-run.')
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    root = project_root()
    script = Path(__file__).resolve().with_name('run_one.sh')
    active = active_mamamia_jobs()
    planned = 0
    submitted = 0
    for target in args.targets:
        for experiment in args.experiments:
            if args.submit and args.max_submit and (submitted >= args.max_submit):
                print(f'reached max submitted jobs: {args.max_submit}')
                print(f'planned_jobs={planned} submitted_jobs={submitted} submit={args.submit}')
                return
            if summary_path(root, target, experiment).exists() is False:
                continue
            job_name = f'mamamia_{target}_{experiment}_bestlast'
            base_name = f'mamamia_{target}_{experiment}'
            if job_name in active or base_name in active:
                print(f'skip active {target} {experiment}: {base_name} or {job_name}')
                continue
            cmd = ['sbatch', f'--account={args.account}', f'--time={args.time}', f'--constraint={args.constraint}', f'--job-name={job_name}', f'--export=ALL,FORCE=1,BEST_LAST_WINDOW={args.window},CLEAN_PREPROCESSED_ON_SUCCESS=0,CLEAN_PREPROCESSED_ON_ERROR=0', str(script), target, experiment]
            planned += 1
            print(' '.join(cmd))
            if args.submit:
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f'submit_failed returncode={exc.returncode} job_name={job_name}')
                    print(f'planned_jobs={planned} submitted_jobs={submitted} submit={args.submit}')
                    return
                submitted += 1
    print(f'planned_jobs={planned} submitted_jobs={submitted} submit={args.submit}')
if __name__ == '__main__':
    main()
