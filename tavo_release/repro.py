from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image
from . import mamamia, officehome
from .common import read_lines, write_json, write_lines
from .matrix import DATASET_METHODS, SCORE_METHODS_8D

def package_root() -> Path:
    return Path(__file__).resolve().parents[1]

def run_cli(args: list[str], cwd: Path) -> dict:
    env = os.environ.copy()
    root = str(cwd)
    env['PYTHONPATH'] = root + os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else root
    proc = subprocess.run([sys.executable, '-m', 'tavo_release.cli', *args], cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError({'args': args, 'stdout': proc.stdout, 'stderr': proc.stderr})
    return json.loads(proc.stdout)

def run_module(module: str, args: list[str], cwd: Path) -> dict:
    env = os.environ.copy()
    root = str(cwd)
    env['PYTHONPATH'] = root + os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else root
    proc = subprocess.run([sys.executable, '-m', module, *args], cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        raise RuntimeError({'module': module, 'args': args, 'stdout': proc.stdout, 'stderr': proc.stderr})
    return json.loads(proc.stdout)

def make_mamamia(root: Path) -> None:
    for domain, prefix in mamamia.DOMAINS.items():
        for idx in range(4):
            case = f'{prefix}{idx:03d}'
            case_dir = root / 'images' / case
            case_dir.mkdir(parents=True, exist_ok=True)
            for channel in ('0000', '0001', '0002'):
                (case_dir / f'{case}_{channel}.nii.gz').write_text('synthetic')
            seg = root / 'segmentations' / 'expert'
            seg.mkdir(parents=True, exist_ok=True)
            (seg / f'{case}.nii.gz').write_text('synthetic')

def make_brats(root: Path) -> None:
    for domain in ('C4', 'C5', 'TCGA_LGG', 'TCGA_GBM'):
        for idx in range(4):
            (root / f'subject_{domain}_{idx:03d}').mkdir(parents=True, exist_ok=True)

def make_officehome(root: Path) -> None:
    for domain in officehome.DOMAINS:
        for label in ('a', 'b'):
            out = root / domain / label
            out.mkdir(parents=True, exist_ok=True)
            for idx in range(45):
                Image.new('RGB', (8, 8), (idx * 12, 40, 90)).save(out / f'{idx}.png')

def make_scores(root: Path, case_ids: list[str]) -> list[str]:
    score_args = []
    for method_idx, method in enumerate(SCORE_METHODS_8D):
        values = {case: float((idx + method_idx) % max(1, len(case_ids))) for idx, case in enumerate(case_ids)}
        path = root / f'{method}.json'
        write_json(path, values)
        score_args.extend(['--score', f'{method}={path}'])
    return score_args

def copy_source_train(split_dir: Path, method: str, budget: int) -> Path:
    source = split_dir / 'random' / f'random_{budget}.txt'
    if not source.exists():
        source = split_dir / 'source_pool.txt'
    target = split_dir / f'{method}_{budget}_source_train.txt'
    write_lines(target, read_lines(source))
    return target

def run_repro_smoke(workdir: str | Path | None=None) -> dict:
    base = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix='tavo_release_repro_'))
    base.mkdir(parents=True, exist_ok=True)
    root = package_root()
    mamamia_root = base / 'data' / 'mamamia'
    brats_root = base / 'data' / 'brats'
    office_root = base / 'data' / 'officehome'
    make_mamamia(mamamia_root)
    make_brats(brats_root)
    make_officehome(office_root)
    results = {}
    results['mamamia_split'] = run_cli(['split', '--dataset', 'mamamia', '--data-root', str(mamamia_root), '--output-root', str(base / 'splits' / 'mamamia_lodo_seed42')], root)
    results['brats_split'] = run_cli(['split', '--dataset', 'brats', '--data-root', str(brats_root), '--output-root', str(base / 'splits' / 'brats'), '--target', 'C5'], root)
    results['officehome_split'] = run_cli(['split', '--dataset', 'officehome', '--data-root', str(office_root), '--output-root', str(base / 'splits' / 'officehome'), '--target', 'Art'], root)
    source_ids = read_lines(base / 'splits' / 'mamamia_lodo_seed42' / 'NACT' / 'source_pool.txt')
    score_args = make_scores(base / 'scores' / 'mamamia' / 'NACT', source_ids)
    results['select'] = run_cli(['select', *score_args, '--weight', '1', '0', '0', '0', '0', '0', '0', '0', '--budget', '5', '--output', str(base / 'splits' / 'mamamia_lodo_seed42' / 'NACT' / 'methods' / 'rds_5.txt')], root)
    results['search'] = run_cli(['search', *score_args, '--budget', '5', '--output-dir', str(base / 'outputs' / 'mamamia' / 'NACT' / 'tavo5'), '--generations', '1', '--popsize', '4'], root)
    results['mamamia_command'] = run_cli(['command', '--dataset', 'mamamia', '--dataset-id', 'NACT:random50'], root)
    office_cfg = base / 'configs' / 'officehome_Art.json'
    results['officehome_config'] = run_cli(['officehome-config', '--split-dir', str(base / 'splits' / 'officehome' / 'Art'), '--output-dir', str(base / 'outputs' / 'officehome' / 'Art'), '--output', str(office_cfg)], root)
    results['officehome_command'] = run_cli(['command', '--dataset', 'officehome', '--config', str(office_cfg)], root)
    results['officehome_train'] = run_module('tavo_release.officehome_train', ['--config', str(office_cfg), '--dry-run'], root)
    brats_cfg = base / 'configs' / 'brats_C5.yaml'
    brats_cfg.parent.mkdir(parents=True, exist_ok=True)
    brats_cfg.write_text('data:\n  train_subjects: ' + str(base / 'splits' / 'brats' / 'C5' / 'target_train.txt') + '\n  val_subjects: ' + str(base / 'splits' / 'brats' / 'C5' / 'target_val.txt') + '\n')
    results['brats_command'] = run_cli(['command', '--dataset', 'brats', '--config', str(brats_cfg), '--seeds', '0'], root)
    results['brats_train'] = run_module('tavo_release.segmentation_train', ['--config', str(brats_cfg), '--seeds', '0', '--dry-run'], root)
    da_specs = []
    for method_idx, method in enumerate(DATASET_METHODS['mamamia']['domain_adaptation']):
        da_specs.append(('mamamia', method, 'NACT', base / 'splits' / 'mamamia_lodo_seed42' / 'NACT', ['--nnunet-dataset-id', str(9000 + method_idx)]))
    for method in DATASET_METHODS['brats']['domain_adaptation']:
        da_specs.append(('brats', method, 'C5', base / 'splits' / 'brats' / 'C5', []))
    for method in DATASET_METHODS['officehome']['domain_adaptation']:
        da_specs.append(('officehome', method, 'Art', base / 'splits' / 'officehome' / 'Art', []))
    da_ok = []
    for dataset, method, target, split_dir, extra in da_specs:
        copy_source_train(split_dir, method, 50)
        cfg = base / 'configs' / f'{dataset}_{target}_{method}_50.json'
        run_cli(['da-config', '--dataset', dataset, '--method', method, '--split-dir', str(split_dir), '--output-dir', str(base / 'outputs' / dataset / target / f'{method}50'), '--budget', '50', '--output', str(cfg), '--target', target, *extra], root)
        run_cli(['da-command', '--config', str(cfg)], root)
        run_module('tavo_release.domain_adaptation_train', ['--config', str(cfg), '--dry-run'], root)
        da_ok.append(f'{dataset}:{method}')
    plan = run_cli(['plan', '--dataset', 'all', '--output-dir', str(base / 'outputs' / 'plans')], root)
    return {'status': 'ok', 'workdir': str(base), 'cli_steps': len(results) + len(da_specs) * 3 + 1, 'da_dry_runs': da_ok, 'plan_steps': plan['steps'], 'search_score': results['search']['score'], 'selected': results['select']['selected']}
