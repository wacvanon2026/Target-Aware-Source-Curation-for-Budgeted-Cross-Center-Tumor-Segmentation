#!/usr/bin/env python3
import json
import math
import shutil
from pathlib import Path
import sys
EFFICIENTVIT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EFFICIENTVIT_ROOT))
from typing import Dict, Any, List
from models.efficientvit_seg.dataset_brats import BraTSSliceDataset
import numpy as np
from scripts.train_seg_short import run_short_training_with_dataset
from torch.utils.data import Subset, ConcatDataset
import yaml
from collections import defaultdict
import copy
from scripts.search_multi.utils_cma import load_norm_scores, build_subset, compute_fast_dice
PROJECT_ROOT = EFFICIENTVIT_ROOT
TEMPLATE_YAML = PROJECT_ROOT / 'configs_TAVO/configs_TCGA_GBM_cma/template.yaml'
SCORE_ROOT = PROJECT_ROOT / 'data/TAVO/splits_TCGA_GBM_mix_scores_multi'
SPLIT_ROOT = PROJECT_ROOT / 'data/fusion_ablation/brats21/TCGA_GBM_stageA2'
OUT_ROOT = PROJECT_ROOT / 'outputs_fusion_ablation/brats21/TCGA_GBM_stageA2'
T = 50

def remap_template_split_paths(cfg):

    def remap_one(value):
        if not isinstance(value, str):
            return value
        if '/EfficientVit/data/TAVO/' in value:
            return value
        old = Path(value)
        marker = '/EfficientVit/data/'
        if marker in value:
            candidate = Path(value.replace(marker, '/EfficientVit/data/TAVO/', 1))
            if (candidate / 'train_subjects.txt').exists() or (candidate / 'val_subjects.txt').exists() or candidate.exists():
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

def maybe_set_warmup(cfg, ckpt_path):
    ckpt_path = str(ckpt_path)
    if Path(ckpt_path).exists():
        cfg['warmup'] = {'checkpoint': ckpt_path}
    else:
        cfg.pop('warmup', None)
        print(f'Warning: warmup checkpoint not found; running surrogate from pretrained init: {ckpt_path}')
    return cfg
WARMUP_CKPT = 'external/efficientvit/outputs_TAVO/outputs_TCGA_GBM/baseline3_brats21_source_plus_target/best_last.pt'
METHODS = ['craig', 'kmeans', 'kcenter', 'diversity']
A2_CFG = {'popsize': 20, 'mu': 10, 'n_gen': 12, 'sigma0': 0.3, 'sigma_min': 0.05, 'iters_eval': 500, 'iters_refine': 1500, 'refine_topk': 8, 'max_subjects': 15, 'fitness_agg': 'median'}

def parse_seeds(seeds: str) -> List[int]:
    return [int(s) for s in seeds.replace(',', ' ').split()]

def aggregate(dice_list: List[float], mode: str) -> float:
    arr = np.asarray(dice_list, dtype=np.float64)
    return float(np.mean(arr)) if mode == 'mean' else float(np.median(arr))

def project_simplex(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    w = np.clip(z, 0.0, None)
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / len(w)
    return w / s

def rank_weights(mu: int) -> np.ndarray:
    w = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
    return w / w.sum()

def run_candidate(score_dicts: Dict[str, Dict[str, float]], budget_T: int, run_tag: str, weights_vec: np.ndarray, iters: int, seeds: str, *, template_cfg: dict, full_source_dataset, target_dataset, val_dataset, subject_to_slice_indices) -> Dict[str, Any]:
    weights_vec = project_simplex(weights_vec)
    weights = {METHODS[i]: float(weights_vec[i]) for i in range(len(METHODS))}
    budget = budget_T * T
    tag = '_'.join([f'{k}{v:.3f}' for k, v in weights.items()]) + f'_it{iters}'
    split_dir = SPLIT_ROOT / f'{budget_T}T' / run_tag / tag
    out_dir = OUT_ROOT / f'{budget_T}T' / run_tag / tag
    train_txt = split_dir / 'train_subjects.txt'
    split_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_subjects = build_subset(score_dicts=score_dicts, weights=weights, budget=budget, out_txt=train_txt)
    indices: List[int] = []
    missing = 0
    for sid in selected_subjects:
        if sid in subject_to_slice_indices:
            indices.extend(subject_to_slice_indices[sid])
        else:
            missing += 1
    if missing > 0:
        print(f'WARNING Warning: {missing} selected subjects not found in full_source_dataset map')
    subset_source = Subset(full_source_dataset, indices)
    train_dataset = ConcatDataset([subset_source, target_dataset])
    seed_list = parse_seeds(seeds)
    dice_per_seed: Dict[int, float] = {}
    for sd in seed_list:
        seed_out = out_dir / f'seed{sd}'
        seed_yaml = seed_out / 'train_config.yaml'
        ckpt_path = seed_out / 'latest.pt'
        if seed_out.exists():
            shutil.rmtree(seed_out)
        seed_out.mkdir(parents=True, exist_ok=True)
        cfg = copy.deepcopy(template_cfg)
        cfg['trainer']['max_iters'] = int(iters)
        cfg['training']['save_dir'] = str(seed_out)
        cfg = maybe_set_warmup(cfg, WARMUP_CKPT)
        seed_yaml.write_text(yaml.safe_dump(cfg))
        run_short_training_with_dataset(cfg=cfg, seeds=[sd], train_dataset=train_dataset, val_dataset=val_dataset)
        dice = compute_fast_dice(seed_out, val_dataset, max_subjects=A2_CFG['max_subjects'])
        dice_per_seed[sd] = float(dice)
        if ckpt_path.exists():
            ckpt_path.unlink()
    fitness = aggregate(list(dice_per_seed.values()), A2_CFG['fitness_agg'])
    return {'weights': weights, 'iters': int(iters), 'fitness': float(fitness), 'dice_per_seed': dice_per_seed, 'out_dir': str(out_dir)}

def _cma_params(dim: int, mu: int, w: np.ndarray) -> Dict[str, float]:
    w = np.asarray(w, dtype=np.float64)
    w = w / np.sum(w)
    mueff = 1.0 / np.sum(w ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))
    return {'mueff': float(mueff), 'cc': float(cc), 'cs': float(cs), 'c1': float(c1), 'cmu': float(cmu), 'damps': float(damps), 'chiN': float(chiN)}

def _update_eigensystem(C: np.ndarray):
    Csym = 0.5 * (C + C.T)
    eigvals, B = np.linalg.eigh(Csym)
    eigvals = np.maximum(eigvals, 1e-12)
    D = np.sqrt(eigvals)
    invsqrtC = B @ np.diag(1.0 / D) @ B.T
    return (B, D, invsqrtC)

def stageA2_cma(budget_T: int, run_tag: str, seeds_eval: str='0', seeds_refine: str='0,1,2', seed: int=0):
    print(' Prebuilding FULL datasets once...')
    template_cfg = yaml.safe_load(TEMPLATE_YAML.read_text())
    data_cfg = template_cfg['data']
    source_dom = next((d for d in data_cfg['domains'] if d['name'] == 'source'))
    target_dom = next((d for d in data_cfg['domains'] if d['name'] == 'target'))
    val_dom = data_cfg['val']
    img_size = data_cfg['img_size']
    full_source_dataset = BraTSSliceDataset(root_dir=source_dom['path'], split='train', img_size=img_size, split_txt_dir=source_dom.get('split_txt'), skip_empty=data_cfg['skip_empty_train'])
    target_dataset = BraTSSliceDataset(root_dir=target_dom['path'], split='train', img_size=img_size, split_txt_dir=target_dom.get('split_txt'), skip_empty=data_cfg['skip_empty_train'])
    val_dataset = BraTSSliceDataset(root_dir=val_dom['path'], split=val_dom['split'], img_size=img_size, split_txt_dir=val_dom.get('split_txt'), skip_empty=data_cfg['skip_empty_val'])
    print(f' full_source slices: {len(full_source_dataset)}')
    print(f' target slices:      {len(target_dataset)}')
    print(f' val slices:         {len(val_dataset)}')
    subject_to_slice_indices = defaultdict(list)
    for idx in range(len(full_source_dataset)):
        _, _, sid, _ = full_source_dataset.samples[idx]
        subject_to_slice_indices[sid].append(idx)
    print(f' Built subject index map for {len(subject_to_slice_indices)} subjects')
    score_dicts = load_norm_scores(SCORE_ROOT, repeat_id=None)
    rng = np.random.default_rng(seed)
    dim = len(METHODS)
    lam = A2_CFG['popsize']
    mu = A2_CFG['mu']
    n_gen = A2_CFG['n_gen']
    it_eval = A2_CFG['iters_eval']
    w_rank = rank_weights(mu)
    w_rank = w_rank / w_rank.sum()
    mueff = 1.0 / np.sum(w_rank ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chiN = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))
    m = np.ones(dim) / dim
    sigma = A2_CFG['sigma0']
    sigma_min = A2_CFG['sigma_min']
    C = np.eye(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    B, D, invsqrtC = _update_eigensystem(C)
    all_evals = []
    best_so_far = None
    gens = []
    save_dir = OUT_ROOT / f'{budget_T}T' / run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    partial_json_path = save_dir / 'stageA2_cma_partial.json'
    print('\n==============================')
    print(' CMA-ES Stage A2 sourcecrit')
    print(f'dim={dim} | popsize={lam} | mu={mu} | n_gen={n_gen}')
    print('==============================\n')
    print('Evaluating corners...')
    corners = []
    for i in range(dim):
        v = np.zeros(dim)
        v[i] = 1.0
        corners.append(v)
    corners.append(np.ones(dim) / dim)
    for c in corners:
        r = run_candidate(score_dicts, budget_T, run_tag, c, it_eval, seeds_eval, template_cfg=template_cfg, full_source_dataset=full_source_dataset, target_dataset=target_dataset, val_dataset=val_dataset, subject_to_slice_indices=subject_to_slice_indices)
        print(f"[Corner] weights={r['weights']} | fitness={r['fitness']:.4f}")
        r['gen'] = -1
        r['z'] = c.tolist()
        all_evals.append(r)
        if best_so_far is None or r['fitness'] > best_so_far['fitness']:
            best_so_far = r
    for g in range(n_gen):
        print('\n----------------------------------')
        print(f' Generation {g}')
        print(f'Current sigma={sigma:.4f}')
        print(f'Current mean (projected)={project_simplex(m)}')
        print('----------------------------------')
        arz = rng.normal(size=(lam, dim))
        ary = arz * D @ B.T
        arx = m[None, :] + sigma * ary
        evals = []
        for k in range(lam):
            z = arx[k]
            r = run_candidate(score_dicts, budget_T, run_tag, z, it_eval, seeds_eval, template_cfg=template_cfg, full_source_dataset=full_source_dataset, target_dataset=target_dataset, val_dataset=val_dataset, subject_to_slice_indices=subject_to_slice_indices)
            print(f"[Gen {g} | Cand {k}] weights={r['weights']} | fitness={r['fitness']:.4f}")
            r['gen'] = g
            r['z'] = z.tolist()
            evals.append(r)
            all_evals.append(r)
        evals.sort(key=lambda x: x['fitness'], reverse=True)
        parents = evals[:mu]
        print(f"\n[Gen {g}] Best fitness this gen = {parents[0]['fitness']:.4f}")
        Z = np.array([p['z'] for p in parents])
        m_old = m.copy()
        m = np.sum(Z * w_rank[:, None], axis=0)
        y_w = (m - m_old) / max(sigma, 1e-12)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ y_w)
        ps_norm = np.linalg.norm(ps)
        hsig = 1.0 if ps_norm / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) < (1.4 + 2 / (dim + 1)) * chiN else 0.0
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        Y = (Z - m_old[None, :]) / max(sigma, 1e-12)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            yi = Y[i][:, None]
            rank_mu += w_rank[i] * (yi @ yi.T)
        C = (1 - c1 - cmu) * C + c1 * (pc[:, None] @ pc[None, :] + (1 - hsig) * cc * (2 - cc) * C) + cmu * rank_mu
        sigma = sigma * math.exp(cs / damps * (ps_norm / chiN - 1.0))
        sigma = max(sigma, sigma_min)
        B, D, invsqrtC = _update_eigensystem(C)
        eigvals = np.linalg.eigvalsh(C)
        eig_min = float(max(np.min(eigvals), 1e-12))
        eig_max = float(np.max(eigvals))
        condC = eig_max / eig_min
        traceC = float(np.sum(eigvals))
        if parents[0]['fitness'] > best_so_far['fitness']:
            best_so_far = parents[0]
        print(f'[Gen {g}] Updated sigma={sigma:.4f}')
        print(f'[Gen {g}] Eigenvalues(C)={eigvals}')
        print(f"[Gen {g}] Global best so far={best_so_far['fitness']:.4f}")
        gens.append({'gen': g, 'sigma': float(sigma), 'mean_z': m.tolist(), 'mean_proj': project_simplex(m).tolist(), 'eigvals': eigvals.tolist(), 'condC': float(condC), 'traceC': float(traceC), 'best_fitness_gen': float(parents[0]['fitness']), 'best_fitness_global': float(best_so_far['fitness'])})
        snapshot = {'cfg': A2_CFG, 'methods': METHODS, 'gens': gens, 'best_so_far': best_so_far, 'n_gen_completed': len(gens)}
        with open(partial_json_path, 'w') as f:
            json.dump(snapshot, f, indent=2)
    print('\n Refining top candidates...')
    all_sorted = sorted(all_evals, key=lambda x: x['fitness'], reverse=True)
    refine_records = []
    for r in all_sorted[:A2_CFG['refine_topk']]:
        vec = np.array([r['weights'][k] for k in METHODS])
        rr = run_candidate(score_dicts, budget_T, run_tag, vec, A2_CFG['iters_refine'], seeds_refine, template_cfg=template_cfg, full_source_dataset=full_source_dataset, target_dataset=target_dataset, val_dataset=val_dataset, subject_to_slice_indices=subject_to_slice_indices)
        print(f"[Refine] weights={rr['weights']} | fitness={rr['fitness']:.4f}")
        refine_records.append(rr)
    refine_records.sort(key=lambda x: x['fitness'], reverse=True)
    return {'cfg': A2_CFG, 'methods': METHODS, 'gens': gens, 'best_so_far': best_so_far, 'top3': refine_records[:3], 'refine_records': refine_records}
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--budget_T', type=int, required=True)
    parser.add_argument('--run_tag', type=str, required=True)
    parser.add_argument('--seeds_eval', type=str, default='0')
    parser.add_argument('--seeds_refine', type=str, default='0,1,2')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = stageA2_cma(args.budget_T, args.run_tag, args.seeds_eval, args.seeds_refine, args.seed)
    save_dir = OUT_ROOT / f'{args.budget_T}T' / args.run_tag
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / 'stageA2_cma.json', 'w') as f:
        json.dump(out, f, indent=2)
