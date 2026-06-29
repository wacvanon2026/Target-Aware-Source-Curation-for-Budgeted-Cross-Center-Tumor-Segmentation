#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path
import sys
EFFICIENTVIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EFFICIENTVIT_ROOT))
import yaml
PROJECT_ROOT = EFFICIENTVIT_ROOT
DATA_ROOT = PROJECT_ROOT / 'data/002_BraTS21'
TARGET_SPLITS = {'C4': 'splits_C4_holdout/repeat_01', 'C5': 'split_C5_T22', 'TCGA_LGG': 'split_TCGA_LGG_T25', 'TCGA_GBM': 'split_TCGA_GBM_T40'}
SEARCH_SCRIPTS = {('C4', 'TargetCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_C4_targetcrit.py', ('C4', 'SourceCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_C4_sourcecrit.py', ('C5', 'TargetCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_C5_targetcrit.py', ('C5', 'SourceCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_C5_sourcecrit.py', ('TCGA_LGG', 'TargetCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_TCGA_LGG_targetcrit.py', ('TCGA_LGG', 'SourceCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_TCGA_LGG_sourcecrit.py', ('TCGA_GBM', 'TargetCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_TCGA_GBM_targetcrit.py', ('TCGA_GBM', 'SourceCriteria'): PROJECT_ROOT / 'scripts/revise_ablation/brats_search_TCGA_GBM_sourcecrit.py'}
STAGE_SPLIT_ROOT = {t: PROJECT_ROOT / f'data/fusion_ablation/brats21/{t}_stageA2' for t in TARGET_SPLITS}
STAGE_OUT_ROOT = {t: PROJECT_ROOT / f'outputs_fusion_ablation/brats21/{t}_stageA2' for t in TARGET_SPLITS}
SCORE_ROOT = {'C4': PROJECT_ROOT / 'data/TAVO/splits_C4_mix_scores_multi', 'C5': PROJECT_ROOT / 'data/TAVO/splits_C5_mix_scores_multi', 'TCGA_LGG': PROJECT_ROOT / 'data/TAVO/splits_TCGA_LGG_mix_scores_multi', 'TCGA_GBM': PROJECT_ROOT / 'data/TAVO/splits_TCGA_GBM_mix_scores_multi'}
ALL_METHODS = ['rds', 'less', 'orient', 'craig', 'gradmatch', 'kmeans', 'kcenter', 'diversity']
METHOD_NAME = {'Uniform': 'TAVO_Uniform', 'TargetCriteria': 'TAVO_TargetCriteria', 'SourceCriteria': 'TAVO_SourceCriteria'}
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.search_multi.utils_cma import load_norm_scores, build_subset

def target_split_dir(target: str) -> Path:
    split = PROJECT_ROOT / 'data/TAVO' / TARGET_SPLITS[target]
    for name in ['train_subjects.txt', 'val_subjects.txt', 'test_subjects.txt']:
        if not (split / name).exists():
            raise FileNotFoundError(split / name)
    return split

def write_final_config(target: str, method_name: str, source_split: Path, train_seed: int) -> Path:
    split = target_split_dir(target)
    cfg_dir = PROJECT_ROOT / 'configs_revise_ablation/brats21' / target / method_name / 'K150'
    out_dir = PROJECT_ROOT / 'outputs_revise_ablation/brats21' / target / method_name / 'K150' / f'train_seed{train_seed:02d}'
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = {'model': {'name': 'efficientvit_l1', 'in_channels': 4, 'num_classes': 4, 'pretrained': True}, 'data': {'skip_empty_train': True, 'skip_empty_val': False, 'domains': [{'name': f'BraTS21_{target}_{method_name}_K150', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(source_split.parent)}, {'name': f'BraTS21_{target}_T_train', 'path': str(DATA_ROOT), 'split': 'train', 'split_txt': str(split)}], 'val': {'path': str(DATA_ROOT), 'split': 'val', 'split_txt': str(split)}, 'test': {'path': str(DATA_ROOT), 'split': 'test', 'split_txt': str(split)}, 'img_size': 512, 'batch_size': 4, 'num_workers': 4}, 'optimizer': {'lr': 0.0001, 'weight_decay': 1e-05}, 'scheduler': {'T_max': 20, 'eta_min': 1e-06}, 'training': {'epochs': 20, 'seed': train_seed, 'save_dir': str(out_dir), 'feature_mode': False}}
    cfg_path = cfg_dir / f'train_seed{train_seed:02d}.yaml'
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path

def build_uniform_subset(target: str) -> Path:
    repeat_id = 1 if target == 'C4' else None
    score_dicts = load_norm_scores(SCORE_ROOT[target], repeat_id=repeat_id)
    weights = {m: 1.0 / len(ALL_METHODS) for m in ALL_METHODS}
    out_dir = PROJECT_ROOT / 'data/fusion_ablation/brats21' / target / 'Uniform' / 'K150'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / 'train_subjects.txt'
    selected = build_subset(score_dicts, weights, 150, out_txt)
    meta = {'target': target, 'method': 'TAVO_Uniform', 'weights': weights, 'budget_cases': 150, 'selected_count': len(selected), 'train_subjects': str(out_txt)}
    (out_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return out_txt

def stage_paths(target: str, run_tag: str):
    if target == 'C4':
        out_root = STAGE_OUT_ROOT[target] / 'repeat01' / '3T' / run_tag
        split_root = STAGE_SPLIT_ROOT[target] / 'repeat01' / '3T' / run_tag
    else:
        out_root = STAGE_OUT_ROOT[target] / '3T' / run_tag
        split_root = STAGE_SPLIT_ROOT[target] / '3T' / run_tag
    return (out_root, split_root)

def run_search_if_needed(target: str, variant: str, run_tag: str, seeds_eval: str, seeds_refine: str, search_seed: int) -> Path:
    out_root, _ = stage_paths(target, run_tag)
    json_path = out_root / 'stageA2_cma.json'
    if json_path.exists():
        print(f'Search JSON exists, skipping search: {json_path}')
        return json_path
    script = SEARCH_SCRIPTS[target, variant]
    cmd = [sys.executable, '-u', str(script)]
    if target == 'C4':
        cmd += ['--repeat', '1']
    cmd += ['--budget_T', '3', '--run_tag', run_tag, '--seeds_eval', seeds_eval, '--seeds_refine', seeds_refine, '--seed', str(search_seed)]
    print('Running search:', ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not json_path.exists():
        raise FileNotFoundError(json_path)
    return json_path

def subset_from_search_json(target: str, variant: str, run_tag: str, json_path: Path) -> Path:
    payload = json.loads(json_path.read_text())
    records = payload.get('top3') or payload.get('refine_records') or []
    if not records:
        raise RuntimeError(f'No refine records in {json_path}')
    record = records[0]
    tag = Path(record['out_dir']).name
    _, split_root = stage_paths(target, run_tag)
    split_txt = split_root / tag / 'train_subjects.txt'
    if not split_txt.exists():
        raise FileNotFoundError(split_txt)
    final_dir = PROJECT_ROOT / 'data/fusion_ablation/brats21' / target / METHOD_NAME[variant] / 'K150'
    final_dir.mkdir(parents=True, exist_ok=True)
    final_txt = final_dir / 'train_subjects.txt'
    final_txt.write_text(split_txt.read_text())
    meta = {'target': target, 'variant': variant, 'method': METHOD_NAME[variant], 'search_json': str(json_path), 'selected_record': record, 'source_split': str(final_txt)}
    (final_dir / 'metadata.json').write_text(json.dumps(meta, indent=2))
    return final_txt

def run_final_train(cfg_path: Path, train_seed: int):
    cfg = yaml.safe_load(cfg_path.read_text())
    out_dir = Path(cfg['training']['save_dir'])
    if (out_dir / 'best_last.pt').exists():
        print(f"Skipping completed final training: {out_dir / 'best_last.pt'}")
        return
    cmd = [sys.executable, '-u', 'scripts/train_seg.py', '--config', str(cfg_path), '--seed', str(train_seed)]
    print('Running final train:', ' '.join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=list(TARGET_SPLITS))
    parser.add_argument('--variant', required=True, choices=['Uniform', 'TargetCriteria', 'SourceCriteria'])
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--search-seed', type=int, default=0)
    parser.add_argument('--seeds-eval', default='0')
    parser.add_argument('--seeds-refine', default='0')
    parser.add_argument('--run-final-train', action='store_true')
    args = parser.parse_args()
    method_name = METHOD_NAME[args.variant]
    if args.variant == 'Uniform':
        split_txt = build_uniform_subset(args.target)
    else:
        run_tag = f'{args.variant}_K150_TFbest_searchseed{args.search_seed:02d}'
        json_path = run_search_if_needed(args.target, args.variant, run_tag, args.seeds_eval, args.seeds_refine, args.search_seed)
        split_txt = subset_from_search_json(args.target, args.variant, run_tag, json_path)
    cfg_path = write_final_config(args.target, method_name, split_txt, args.train_seed)
    print(f'Final config: {cfg_path}')
    if args.run_final_train:
        run_final_train(cfg_path, args.train_seed)
if __name__ == '__main__':
    main()
