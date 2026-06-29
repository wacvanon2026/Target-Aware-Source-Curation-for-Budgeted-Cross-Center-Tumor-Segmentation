#!/usr/bin/env python3
import os, json, yaml, shutil, subprocess
import numpy as np
from pathlib import Path
import sys
EFFICIENTVIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EFFICIENTVIT_ROOT))
from scripts.search_multi.eval_early_dice_stageB import evaluate_early_dice_2d_to_3d
PROJECT_ROOT = EFFICIENTVIT_ROOT
TRAIN_ENTRY = ['python', 'scripts/train_seg_short.py']
DEVICE_ENV = {'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '8', 'OPENBLAS_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8', 'NUMEXPR_MAX_THREADS': '8'}

def load_norm_scores(score_root: Path, repeat_id: int | None):
    if repeat_id is None:
        base = score_root
    else:
        base = score_root / f'repeat{repeat_id:02d}'
    score_dicts = {}
    for f in base.glob('*_norm_dict.npy'):
        name = f.stem.replace('_norm_dict', '')
        score_dicts[name] = np.load(f, allow_pickle=True).item()
    keys = None
    for d in score_dicts.values():
        if keys is None:
            keys = set(d.keys())
        else:
            assert keys == set(d.keys())
    return score_dicts

def build_subset(score_dicts: dict, weights: dict, budget: int, out_txt: Path):
    keys = list(score_dicts.values())[0].keys()
    scores = {}
    for k in keys:
        total = 0.0
        for name, score_dict in score_dicts.items():
            if name in weights:
                total += weights[name] * score_dict[k]
        scores[k] = total
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [k for k, _ in ranked[:budget]]
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text('\n'.join(selected))
    return selected

def set_domain_split(cfg, domain_name: str, split_txt: Path):
    found = False
    for dom in cfg['data']['domains']:
        if dom.get('name') == domain_name:
            dom['split_txt'] = str(split_txt)
            found = True
    assert found

def build_yaml(template_yaml: Path, out_yaml: Path, train_subjects_txt: Path, output_dir: Path, max_iters: int, resume_ckpt: str | None=None, stage: str='A'):
    cfg = yaml.safe_load(template_yaml.read_text())
    set_domain_split(cfg, 'source', train_subjects_txt.parent)
    cfg['trainer']['max_iters'] = int(max_iters)
    cfg['training']['save_dir'] = str(output_dir)
    if stage == 'A':
        cfg['data']['skip_empty_train'] = True
        cfg['data']['skip_empty_val'] = True
    if resume_ckpt is not None:
        cfg['warmup'] = {'checkpoint': str(resume_ckpt)}
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg))

def run_training(yaml_path: Path, seeds='0'):
    env = os.environ.copy()
    env.update(DEVICE_ENV)
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    cmd = TRAIN_ENTRY + ['--config', str(yaml_path), '--seeds', str(seeds)]
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)

def compute_fast_dice(out_dir: Path, val_dataset, max_subjects: int):
    out_dir = Path(out_dir)
    config_path = out_dir / 'train_config.yaml'
    ckpt_path = out_dir / 'latest.pt'
    assert config_path.exists()
    assert ckpt_path.exists()
    print('\n Running early Dice evaluation...')
    results = evaluate_early_dice_2d_to_3d(config_path=str(config_path), checkpoint_path=str(ckpt_path), max_subjects=max_subjects, val_dataset=val_dataset)
    print('\n================== EARLY DICE ==================')
    print(f"Dice_ET    : {results['dice_ET']:.4f}")
    print(f"Dice_TC    : {results['dice_TC']:.4f}")
    print(f"Dice_WT    : {results['dice_WT']:.4f}")
    print(f"Macro Avg  : {results['dice_macro']:.4f}")
    print(f"Subjects   : {results['num_subjects']}")
    print('================================================\n')
    return float(results['dice_macro'])
