#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import subprocess
from pathlib import Path
from core import DOMAINS, EXPERIMENTS, REPO_ROOT
METHODS = {'dann': {'trainer': 'nnUNetTrainerTAVODANN', 'suffix': 'dann', 'description': 'DANN-style gradient-reversal domain classifier on nnU-Net probability maps.'}, 'mmd': {'trainer': 'nnUNetTrainerTAVOMMD', 'suffix': 'mmd', 'description': 'DAN/MMD-style RBF distribution alignment on pooled class probabilities.'}, 'dan': {'trainer': 'nnUNetTrainerTAVOMMD', 'suffix': 'mmd', 'description': 'Alias for the MMD/DAN-style trainer.'}, 'advent': {'trainer': 'nnUNetTrainerTAVOADVENT', 'suffix': 'advent', 'description': 'ADVENT-style adversarial alignment on prediction entropy maps.'}, 'seasa': {'trainer': 'nnUNetTrainerTAVOSEASA', 'suffix': 'seasa', 'description': 'SE-ASA-style entropy and semantic-distribution alignment.'}, 'se-asa': {'trainer': 'nnUNetTrainerTAVOSEASA', 'suffix': 'seasa', 'description': 'Alias for the SE-ASA-style trainer.'}}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--methods', nargs='+', default=['dann', 'mmd', 'advent', 'seasa'], choices=sorted(METHODS), help='Domain-alignment methods to run.')
    parser.add_argument('--targets', nargs='+', default=list(DOMAINS), help='Targets to run: NACT ISPY1 DUKE ISPY2, or all.')
    parser.add_argument('--experiments', nargs='+', default=['target_full_source'], choices=sorted(EXPERIMENTS), help='Existing MAMAMIA dataset rows to train on. target_full_source is the intended default because alignment needs source and target cases.')
    parser.add_argument('--account', default=os.environ.get('SBATCH_ACCOUNT', 'YOUR_SLURM_ACCOUNT'))
    parser.add_argument('--partition', default=os.environ.get('SBATCH_PARTITION', 'gpu'))
    parser.add_argument('--time', default=os.environ.get('SBATCH_TIME', '12:00:00'))
    parser.add_argument('--constraint', default=os.environ.get('SBATCH_CONSTRAINT', 'a100|a40|l40s|v100'))
    parser.add_argument('--lambda', dest='lambda_', default=os.environ.get('ADAPT_LAMBDA', '0.1'))
    parser.add_argument('--domain-lambda', default=os.environ.get('ADAPT_DOMAIN_LAMBDA'))
    parser.add_argument('--entropy-lambda', default=os.environ.get('ADAPT_ENTROPY_LAMBDA'))
    parser.add_argument('--semantic-lambda', default=os.environ.get('ADAPT_SEMANTIC_LAMBDA'))
    parser.add_argument('--disc-hidden', type=int, default=int(os.environ.get('ADAPT_DISC_HIDDEN', '64')))
    parser.add_argument('--force', action='store_true', help='Export FORCE=1 so existing summaries are recomputed.')
    parser.add_argument('--auto-build', action='store_true', help='Export AUTO_BUILD_DATASET=1 for run_one.sh.')
    parser.add_argument('--max-submit', type=int, default=0, help='Maximum jobs to submit in this invocation; 0 means no limit.')
    parser.add_argument('--submit', action='store_true', help='Actually submit jobs. Default is dry-run.')
    return parser.parse_args()

def normalize_targets(targets: list[str]) -> list[str]:
    normalized: list[str] = []
    for target in targets:
        upper = target.upper().replace('-', '')
        if upper == 'ALL':
            return list(DOMAINS)
        if upper == 'ISPY1':
            upper = 'ISPY1'
        elif upper == 'ISPY2':
            upper = 'ISPY2'
        if upper not in DOMAINS:
            raise SystemExit(f"Unknown target {target!r}; choose from {', '.join(DOMAINS)} or all")
        normalized.append(upper)
    return normalized

def export_string(args: argparse.Namespace, method: str) -> str:
    spec = METHODS[method]
    env = {'TRAINER': spec['trainer'], 'OUTPUT_SUFFIX': f"_{spec['suffix']}", 'ADAPT_METHOD': spec['suffix'], 'ADAPT_LAMBDA': str(args.lambda_), 'ADAPT_DISC_HIDDEN': str(args.disc_hidden), 'FORCE': '1' if args.force else '0', 'AUTO_BUILD_DATASET': '1' if args.auto_build else '0', 'CLEAN_PREPROCESSED_ON_SUCCESS': '0', 'CLEAN_PREPROCESSED_ON_ERROR': '0'}
    if args.domain_lambda is not None:
        env['ADAPT_DOMAIN_LAMBDA'] = args.domain_lambda
    if args.entropy_lambda is not None:
        env['ADAPT_ENTROPY_LAMBDA'] = args.entropy_lambda
    if args.semantic_lambda is not None:
        env['ADAPT_SEMANTIC_LAMBDA'] = args.semantic_lambda
    return 'ALL,' + ','.join((f'{key}={value}' for key, value in env.items()))

def main() -> None:
    args = parse_args()
    targets = normalize_targets(args.targets)
    script = Path(__file__).resolve().with_name('run_one.sh')
    repo_root = REPO_ROOT
    planned = 0
    submitted = 0
    print(f"submit={args.submit} targets={' '.join(targets)} experiments={' '.join(args.experiments)}")
    print('methods=' + ' '.join(dict.fromkeys((METHODS[m]['suffix'] for m in args.methods))))
    print('note=target_full_source is the intended row; source-only or target-only rows cannot form alignment batches.')
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        for experiment in args.experiments:
            for raw_method in args.methods:
                spec = METHODS[raw_method]
                method = spec['suffix']
                key = (target, experiment, method)
                if key in seen:
                    continue
                seen.add(key)
                job_name = f'mamamia_{target}_{experiment}_{method}'
                cmd = ['sbatch', f'--account={args.account}', f'--partition={args.partition}', f'--time={args.time}', f'--constraint={args.constraint}', f'--job-name={job_name}', f'--export={export_string(args, raw_method)}', str(script), target, experiment]
                planned += 1
                print(' '.join(cmd))
                if args.submit:
                    if args.max_submit and submitted >= args.max_submit:
                        print(f'reached max submitted jobs: {args.max_submit}')
                        print(f'planned_jobs={planned} submitted_jobs={submitted} submit={args.submit}')
                        return
                    try:
                        subprocess.run(cmd, cwd=repo_root, check=True)
                    except subprocess.CalledProcessError as exc:
                        print(f'submit_failed returncode={exc.returncode} job_name={job_name}')
                        print(f'planned_jobs={planned} submitted_jobs={submitted} submit={args.submit}')
                        return
                    submitted += 1
    print(f'planned_jobs={planned} submitted_jobs={submitted} submit={args.submit}')
if __name__ == '__main__':
    main()
