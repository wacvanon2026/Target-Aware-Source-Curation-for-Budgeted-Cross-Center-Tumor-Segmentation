#!/usr/bin/env python3
import argparse
import copy
import json
import math
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import sys
EFFICIENTVIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EFFICIENTVIT_ROOT))
from typing import Any
import numpy as np
import yaml
from torch.utils.data import ConcatDataset, Subset
PROJECT_ROOT = EFFICIENTVIT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
from scripts.search_multi.utils_cma import build_subset, compute_fast_dice, load_norm_scores
from scripts.train_seg_short import run_short_training_with_dataset
DATA_ROOT = PROJECT_ROOT / 'data/002_BraTS21'
T = 50
BUDGET_T = 3
BUDGET_CASES = BUDGET_T * T
METHODS = ['rds', 'less', 'orient', 'craig', 'gradmatch', 'kmeans', 'kcenter', 'diversity']
TARGETS = ['C4', 'C5', 'TCGA_LGG', 'TCGA_GBM']
TARGET_INFO = {'C4': {'template': PROJECT_ROOT / 'configs_TAVO/configs_C4_cma/repeat01/template.yaml', 'score_root': PROJECT_ROOT / 'data/TAVO/splits_C4_mix_scores_multi', 'repeat_id': 1, 'target_split': PROJECT_ROOT / 'data/TAVO/splits_C4_holdout/repeat_01', 'warmup': PROJECT_ROOT / 'outputs_TAVO/outputs_C4_new/source_plus_target_repeat01/best.pt'}, 'C5': {'template': PROJECT_ROOT / 'configs_TAVO/configs_C5_cma/template.yaml', 'score_root': PROJECT_ROOT / 'data/TAVO/splits_C5_mix_scores_multi', 'repeat_id': None, 'target_split': PROJECT_ROOT / 'data/TAVO/split_C5_T22', 'warmup': PROJECT_ROOT / 'outputs_TAVO/outputs_C5/baseline3_brats21_source_plus_target/best_last.pt'}, 'TCGA_LGG': {'template': PROJECT_ROOT / 'configs_TAVO/configs_TCGA_LGG_cma/template.yaml', 'score_root': PROJECT_ROOT / 'data/TAVO/splits_TCGA_LGG_mix_scores_multi', 'repeat_id': None, 'target_split': PROJECT_ROOT / 'data/TAVO/split_TCGA_LGG_T25', 'warmup': PROJECT_ROOT / 'outputs_TAVO/outputs_TCGA_LGG/baseline3_brats21_source_plus_target/best_last.pt'}, 'TCGA_GBM': {'template': PROJECT_ROOT / 'configs_TAVO/configs_TCGA_GBM_cma/template.yaml', 'score_root': PROJECT_ROOT / 'data/TAVO/splits_TCGA_GBM_mix_scores_multi', 'repeat_id': None, 'target_split': PROJECT_ROOT / 'data/TAVO/split_TCGA_GBM_T40', 'warmup': PROJECT_ROOT / 'outputs_TAVO/outputs_TCGA_GBM/baseline3_brats21_source_plus_target/best_last.pt'}}
A2_CFG = {'popsize': 20, 'n_gen': 12, 'iters_eval': 500, 'iters_refine': 1500, 'refine_topk': 8, 'max_subjects': 15, 'fitness_agg': 'median'}

def parse_seeds(raw: str) -> list[int]:
    return [int(x) for x in raw.replace(',', ' ').split() if x.strip()]

def aggregate(values: list[float], mode: str) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr)) if mode == 'mean' else float(np.median(arr))

def remap_template_split_paths(cfg: dict[str, Any]) -> dict[str, Any]:

    def remap_one(value):
        if not isinstance(value, str) or '/EfficientVit/data/TAVO/' in value:
            return value
        marker = '/EfficientVit/data/'
        if marker in value:
            candidate = Path(value.replace(marker, '/EfficientVit/data/TAVO/', 1))
            if candidate.exists() or (candidate / 'train_subjects.txt').exists() or (candidate / 'val_subjects.txt').exists():
                print(f'Remapped split path: {value} -> {candidate}')
                return str(candidate)
        return value
    data_cfg = cfg.get('data', {})
    for dom in data_cfg.get('domains', []):
        if 'split_txt' in dom:
            dom['split_txt'] = remap_one(dom['split_txt'])
    for key in ['val', 'test']:
        if key in data_cfg and 'split_txt' in data_cfg[key]:
            data_cfg[key]['split_txt'] = remap_one(data_cfg[key]['split_txt'])
    return cfg

def maybe_set_warmup(cfg: dict[str, Any], ckpt_path: Path) -> dict[str, Any]:
    if ckpt_path.exists():
        cfg['warmup'] = {'checkpoint': str(ckpt_path)}
        print(f'Using Target+Full Source warmup: {ckpt_path}')
    else:
        raise FileNotFoundError(f'Missing warmup checkpoint: {ckpt_path}')
    return cfg

def project_simplex(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    vec = np.clip(vec, 0.0, None)
    total = float(vec.sum())
    if total <= 0:
        return np.ones_like(vec) / len(vec)
    return vec / total

def build_datasets(template_cfg: dict[str, Any]):
    data_cfg = template_cfg['data']
    source_dom = next((d for d in data_cfg['domains'] if d['name'] == 'source'))
    target_dom = next((d for d in data_cfg['domains'] if d['name'] == 'target'))
    val_dom = data_cfg['val']
    img_size = data_cfg['img_size']
    full_source_dataset = BraTSSliceDataset(root_dir=source_dom['path'], split='train', img_size=img_size, split_txt_dir=source_dom.get('split_txt'), skip_empty=data_cfg['skip_empty_train'])
    target_dataset = BraTSSliceDataset(root_dir=target_dom['path'], split='train', img_size=img_size, split_txt_dir=target_dom.get('split_txt'), skip_empty=data_cfg['skip_empty_train'])
    val_dataset = BraTSSliceDataset(root_dir=val_dom['path'], split=val_dom['split'], img_size=img_size, split_txt_dir=val_dom.get('split_txt'), skip_empty=data_cfg['skip_empty_val'])
    subject_to_slice_indices = defaultdict(list)
    for idx in range(len(full_source_dataset)):
        _, _, sid, _ = full_source_dataset.samples[idx]
        subject_to_slice_indices[sid].append(idx)
    print(f'full_source slices={len(full_source_dataset)} target slices={len(target_dataset)} val slices={len(val_dataset)}')
    print(f'subject index map={len(subject_to_slice_indices)} subjects')
    return (full_source_dataset, target_dataset, val_dataset, subject_to_slice_indices)

def run_candidate(*, target: str, score_dicts: dict[str, dict[str, float]], weights_vec: np.ndarray, iter_tag: str, eval_id: int, iters: int, seeds: list[int], run_tag: str, template_cfg: dict[str, Any], full_source_dataset, target_dataset, val_dataset, subject_to_slice_indices) -> dict[str, Any]:
    info = TARGET_INFO[target]
    weights_vec = project_simplex(weights_vec)
    weights = {METHODS[i]: float(weights_vec[i]) for i in range(len(METHODS))}
    tag = '_'.join([f'{k}{v:.3f}' for k, v in weights.items()]) + f'_{iter_tag}_it{iters}_id{eval_id:04d}'
    split_dir = PROJECT_ROOT / 'data/dirichlet_ablation/brats21' / target / 'K150' / run_tag / tag
    out_dir = PROJECT_ROOT / 'outputs_dirichlet_ablation/brats21' / target / 'K150' / run_tag / tag
    train_txt = split_dir / 'train_subjects.txt'
    split_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_subjects = build_subset(score_dicts=score_dicts, weights=weights, budget=BUDGET_CASES, out_txt=train_txt)
    indices = []
    missing = 0
    for sid in selected_subjects:
        if sid in subject_to_slice_indices:
            indices.extend(subject_to_slice_indices[sid])
        else:
            missing += 1
    if missing:
        print(f'Warning: {missing} selected subjects not found in source dataset')
    train_dataset = ConcatDataset([Subset(full_source_dataset, indices), target_dataset])
    dice_per_seed = {}
    for sd in seeds:
        seed_out = out_dir / f'seed{sd}'
        seed_yaml = seed_out / 'train_config.yaml'
        ckpt_path = seed_out / 'latest.pt'
        if seed_out.exists():
            shutil.rmtree(seed_out)
        seed_out.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(template_cfg)
        cfg['trainer']['max_iters'] = int(iters)
        cfg['training']['save_dir'] = str(seed_out)
        cfg = maybe_set_warmup(cfg, info['warmup'])
        seed_yaml.write_text(yaml.safe_dump(cfg))
        run_short_training_with_dataset(cfg=cfg, seeds=[sd], train_dataset=train_dataset, val_dataset=val_dataset)
        dice = compute_fast_dice(seed_out, val_dataset, max_subjects=A2_CFG['max_subjects'])
        dice_per_seed[int(sd)] = float(dice)
        if ckpt_path.exists():
            ckpt_path.unlink()
    return {'id': int(eval_id), 'iter_tag': iter_tag, 'weights': weights, 'z': weights_vec.tolist(), 'iters': int(iters), 'fitness': aggregate(list(dice_per_seed.values()), A2_CFG['fitness_agg']), 'dice_per_seed': dice_per_seed, 'out_dir': str(out_dir), 'subset_path': str(train_txt), 'selected_count': len(selected_subjects)}

def target_split_dir(target: str) -> Path:
    split = TARGET_INFO[target]['target_split']
    for name in ['train_subjects.txt', 'val_subjects.txt', 'test_subjects.txt']:
        if not (split / name).exists():
            raise FileNotFoundError(split / name)
    return split

def write_final_config(target: str, source_split: Path, train_seed: int) -> Path:
    split = target_split_dir(target)
    method_name = 'TAVO_Dirichlet_8D'
    cfg_dir = PROJECT_ROOT / 'configs_revise_ablation/brats21' / target / method_name / 'K150'
    out_dir = PROJECT_ROOT / 'outputs_revise_ablation/brats21' / target / method_name / 'K150' / f'train_seed{train_seed:02d}'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = {'model': {'name': 'efficientvit_l1', 'in_channels': 4, 'num_classes': 4, 'pretrained': True}, 'data': {'skip_empty_train': True, 'skip_empty_val': False, 'domains': [{'name': f'BraTS21_{target}_{method_name}_K150', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(source_split.parent)}, {'name': f'BraTS21_{target}_T_train', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(split)}], 'val': {'path': str(DATA_ROOT), 'split': 'val', 'split_txt': str(split)}, 'test': {'path': str(DATA_ROOT), 'split': 'test', 'split_txt': str(split)}, 'img_size': 512, 'batch_size': 4, 'num_workers': 4}, 'optimizer': {'lr': 0.0001, 'weight_decay': 1e-05}, 'scheduler': {'T_max': 20, 'eta_min': 1e-06}, 'training': {'epochs': 20, 'seed': train_seed, 'save_dir': str(out_dir), 'feature_mode': False}}
    cfg_path = cfg_dir / f'train_seed{train_seed:02d}.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path

def finalize_subset(record: dict[str, Any], target: str, run_tag: str) -> Path:
    final_dir = PROJECT_ROOT / 'data/dirichlet_ablation/brats21' / target / 'TAVO_Dirichlet_8D' / 'K150'
    final_dir.mkdir(parents=True, exist_ok=True)
    final_txt = final_dir / 'train_subjects.txt'
    final_txt.write_text(Path(record['subset_path']).read_text())
    meta = {'target': target, 'method': 'TAVO_Dirichlet_8D', 'run_tag': run_tag, 'record': record, 'source_split': str(final_txt)}
    (final_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return final_txt

def run_final_train(cfg_path: Path, train_seed: int) -> None:
    cfg = yaml.safe_load(cfg_path.read_text())
    out_dir = Path(cfg['training']['save_dir'])
    if (out_dir / 'best_last.pt').exists():
        print(f"Skipping completed final training: {out_dir / 'best_last.pt'}")
        return
    cmd = [sys.executable, '-u', 'scripts/train_seg.py', '--config', str(cfg_path), '--seed', str(train_seed)]
    print('Running final train:', ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

def run_search(args: argparse.Namespace) -> None:
    info = TARGET_INFO[args.target]
    run_tag = f'Dirichlet_K150_searchseed{args.search_seed:02d}'
    save_dir = PROJECT_ROOT / 'outputs_dirichlet_ablation/brats21' / args.target / 'K150' / run_tag
    json_path = save_dir / 'stageA2_dirichlet.json'
    partial_path = save_dir / 'stageA2_dirichlet_partial.json'
    save_dir.mkdir(parents=True, exist_ok=True)
    template_cfg = remap_template_split_paths(yaml.safe_load(info['template'].read_text()))
    full_source_dataset, target_dataset, val_dataset, subject_to_slice_indices = build_datasets(template_cfg)
    score_dicts = load_norm_scores(info['score_root'], repeat_id=info['repeat_id'])
    missing = [m for m in METHODS if m not in score_dicts]
    if missing:
        raise KeyError(f'Missing score dicts for {missing}; available={sorted(score_dicts)}')
    rng = np.random.default_rng(args.search_seed)
    dim = len(METHODS)
    n_random = args.n_random if args.n_random > 0 else args.popsize * args.n_gen
    eval_seeds = parse_seeds(args.seeds_eval)
    refine_seeds = parse_seeds(args.seeds_refine)
    print('===== BraTS Dirichlet random-simplex 8D search =====')
    print(f'target={args.target} K150 search_seed={args.search_seed}')
    print(f'methods={METHODS}')
    print(f'corners+uniform={dim + 1} n_random={n_random} iters_eval={args.iters_eval} iters_refine={args.iters_refine}')
    print(f'eval_seeds={eval_seeds} refine_seeds={refine_seeds}')
    candidates = []
    for i in range(dim):
        v = np.zeros(dim, dtype=np.float64)
        v[i] = 1.0
        candidates.append(('corner', v))
    candidates.append(('uniform', np.ones(dim, dtype=np.float64) / dim))
    for _ in range(n_random):
        candidates.append(('dirichlet', rng.dirichlet(np.ones(dim, dtype=np.float64))))
    all_evals = []
    best_so_far = None
    for eval_id, (tag, vec) in enumerate(candidates):
        record = run_candidate(target=args.target, score_dicts=score_dicts, weights_vec=vec, iter_tag=tag, eval_id=eval_id, iters=args.iters_eval, seeds=eval_seeds, run_tag=run_tag, template_cfg=template_cfg, full_source_dataset=full_source_dataset, target_dataset=target_dataset, val_dataset=val_dataset, subject_to_slice_indices=subject_to_slice_indices)
        record['sampler'] = tag
        all_evals.append(record)
        if best_so_far is None or record['fitness'] > best_so_far['fitness']:
            best_so_far = record
        print(f"[{tag}] id={eval_id} fitness={record['fitness']:.4f} weights={record['weights']}")
        if (eval_id + 1) % 20 == 0:
            partial_path.write_text(json.dumps({'cfg': vars(args), 'methods': METHODS, 'all_evals': all_evals, 'best_so_far': best_so_far}, indent=2))
    all_sorted = sorted(all_evals, key=lambda x: x['fitness'], reverse=True)
    refine_inputs = all_sorted[:args.refine_topk]
    if best_so_far['id'] not in {r['id'] for r in refine_inputs}:
        refine_inputs.append(best_so_far)
    print('\nRefining top Dirichlet candidates...')
    refine_records = []
    eval_id = len(all_evals)
    for record in refine_inputs:
        vec = np.array([record['weights'][method] for method in METHODS], dtype=np.float64)
        refined = run_candidate(target=args.target, score_dicts=score_dicts, weights_vec=vec, iter_tag='refine', eval_id=eval_id, iters=args.iters_refine, seeds=refine_seeds, run_tag=run_tag, template_cfg=template_cfg, full_source_dataset=full_source_dataset, target_dataset=target_dataset, val_dataset=val_dataset, subject_to_slice_indices=subject_to_slice_indices)
        refined['origin_eval_id'] = record['id']
        refined['origin_fitness'] = record['fitness']
        refined['origin_sampler'] = record.get('sampler')
        refine_records.append(refined)
        print(f"[refine] origin={record['id']} refined_id={eval_id} fitness={refined['fitness']:.4f}")
        eval_id += 1
    refine_records.sort(key=lambda x: x['fitness'], reverse=True)
    payload = {'cfg': vars(args), 'methods': METHODS, 'all_evals': all_evals, 'best_so_far': best_so_far, 'top3': refine_records[:3], 'refine_records': refine_records, 'dirichlet_8d_record': refine_records[0]}
    json_path.write_text(json.dumps(payload, indent=2))
    print(f'Saved search JSON: {json_path}')
    final_subset = finalize_subset(refine_records[0], args.target, run_tag)
    cfg_path = write_final_config(args.target, final_subset, args.train_seed)
    manifest = {'target': args.target, 'method': 'TAVO_Dirichlet_8D', 'budget': 'K150', 'train_seed': args.train_seed, 'config': str(cfg_path), 'source_subset': str(final_subset), 'search_json': str(json_path), 'category': 'dirichlet_ablation'}
    (save_dir / 'final_train_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Final config: {cfg_path}')
    if args.run_final_train:
        run_final_train(cfg_path, args.train_seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=TARGETS)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--search-seed', type=int, default=0)
    parser.add_argument('--seeds-eval', default='0')
    parser.add_argument('--seeds-refine', default='0')
    parser.add_argument('--popsize', type=int, default=A2_CFG['popsize'])
    parser.add_argument('--n-gen', type=int, default=A2_CFG['n_gen'])
    parser.add_argument('--n-random', type=int, default=0)
    parser.add_argument('--iters-eval', type=int, default=A2_CFG['iters_eval'])
    parser.add_argument('--iters-refine', type=int, default=A2_CFG['iters_refine'])
    parser.add_argument('--refine-topk', type=int, default=A2_CFG['refine_topk'])
    parser.add_argument('--run-final-train', action='store_true')
    args = parser.parse_args()
    A2_CFG['popsize'] = args.popsize
    A2_CFG['n_gen'] = args.n_gen
    A2_CFG['iters_eval'] = args.iters_eval
    A2_CFG['iters_refine'] = args.iters_refine
    A2_CFG['refine_topk'] = args.refine_topk
    run_search(args)
if __name__ == '__main__':
    main()
PY
