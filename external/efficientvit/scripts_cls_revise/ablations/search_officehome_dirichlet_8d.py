#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts_cls_revise.search_officehome_tavo_8d import METHODS, PROJECT_ROOT, TARGETS, build_paths, finalize_subset, group_by_label, load_rank_scores, parse_int_list, read_items, run_candidate, run_final_training, save_json, set_seed, write_final_config

def run_dirichlet(args: argparse.Namespace) -> None:
    if args.target not in TARGETS:
        raise ValueError(f'Unknown target {args.target}; expected one of {TARGETS}')
    set_seed(args.search_seed)
    paths = build_paths(args)
    for key in ['source_train', 'target_train', 'target_val', 'target_test']:
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])
    if not paths['rank_root'].exists():
        raise FileNotFoundError(paths['rank_root'])
    source_items = read_items(paths['source_train'])
    source_by_label = group_by_label(source_items)
    if len(source_by_label) != 65:
        raise RuntimeError(f'Expected 65 source classes, found {len(source_by_label)}')
    for label, pool in source_by_label.items():
        if len(pool) < args.budget_per_class:
            raise RuntimeError(f'{label} has only {len(pool)} source images')
    score_dicts = load_rank_scores(paths['rank_root'], args.budget_per_class)
    dim = len(METHODS)
    n_random = args.n_random if args.n_random > 0 else args.popsize * args.n_gen
    rng = np.random.default_rng(args.search_seed)
    eval_seeds = parse_int_list(args.eval_seeds)
    refine_seeds = parse_int_list(args.refine_seeds)
    final_train_seeds = parse_int_list(args.final_train_seeds)
    json_path = paths['search_root'] / 'stageA2_dirichlet.json'
    partial_path = paths['search_root'] / 'stageA2_dirichlet_partial.json'
    paths['search_root'].mkdir(parents=True, exist_ok=True)
    print('===== OfficeHome Dirichlet random-simplex 8D search =====')
    print(f'target={args.target} B={args.budget_per_class} search_seed={args.search_seed}')
    print(f'methods={METHODS}')
    print(f'corners+uniform={dim + 1} n_random={n_random} eval_epochs={args.eval_epochs}')
    print(f'eval_seeds={eval_seeds} refine_seeds={refine_seeds} final_train_seeds={final_train_seeds}')
    all_evals: list[dict[str, Any]] = []
    best_so_far = None
    eval_id = 0
    candidates = []
    for i in range(dim):
        v = np.zeros(dim, dtype=np.float64)
        v[i] = 1.0
        candidates.append(('corner', v))
    candidates.append(('uniform', np.ones(dim, dtype=np.float64) / dim))
    for _ in range(n_random):
        candidates.append(('dirichlet', rng.dirichlet(np.ones(dim, dtype=np.float64))))
    for iter_tag, vec in candidates:
        record = run_candidate(source_items, score_dicts, vec, args, paths, iter_tag, eval_id, eval_seeds)
        record['sampler'] = iter_tag
        all_evals.append(record)
        if best_so_far is None or record['fitness'] > best_so_far['fitness']:
            best_so_far = record
        print(f"[{iter_tag}] id={eval_id} fitness={record['fitness']:.4f} weights={record['weights']}")
        eval_id += 1
        if eval_id % 20 == 0:
            save_json(partial_path, {'cfg': vars(args), 'methods': METHODS, 'all_evals': all_evals, 'best_so_far': best_so_far, 'n_eval': eval_id})
    all_sorted = sorted(all_evals, key=lambda x: x['fitness'], reverse=True)
    refine_inputs = all_sorted[:args.refine_topk]
    if best_so_far['id'] not in {r['id'] for r in refine_inputs}:
        refine_inputs.append(best_so_far)
    print('\nRefining top Dirichlet candidates...')
    old_eval_epochs = args.eval_epochs
    args.eval_epochs = args.refine_epochs
    refine_records = []
    for rank, record in enumerate(refine_inputs):
        vec = np.array([record['weights'][method] for method in METHODS], dtype=np.float64)
        refined = run_candidate(source_items, score_dicts, vec, args, paths, 'refine', eval_id, refine_seeds)
        refined['origin_eval_id'] = record['id']
        refined['origin_fitness'] = record['fitness']
        refined['origin_sampler'] = record.get('sampler')
        refined['sampler'] = 'refine'
        refine_records.append(refined)
        print(f"[refine] origin={record['id']} refined_id={eval_id} fitness={refined['fitness']:.4f}")
        eval_id += 1
    args.eval_epochs = old_eval_epochs
    refine_records.sort(key=lambda x: x['fitness'], reverse=True)
    best_origin_refine = next((r for r in refine_records if r.get('origin_eval_id') == best_so_far['id']), None)
    if best_origin_refine is None:
        best_origin_refine = best_so_fa
    payload = {'cfg': vars(args), 'methods': METHODS, 'all_evals': all_evals, 'best_so_far': best_so_far, 'top3': refine_records[:3], 'refine_records': refine_records, 'dirichlet_8d_record': refine_records[0], 'dirichlet_8d_best_record': best_origin_refine}
    save_json(json_path, payload)
    print(f'Saved search JSON: {json_path}')
    final_manifest = []
    final_configs = []
    variants = [('TAVO_Dirichlet_8D', refine_records[0])]
    if args.emit_best_so_far:
        variants.append(('TAVO_Dirichlet_8D_best', best_origin_refine))
    for method_name, record in variants:
        subset = finalize_subset(record, args.target, method_name, paths, args)
        for train_seed in final_train_seeds:
            cfg_path, out_dir = write_final_config(args.target, method_name, subset, paths, args, train_seed)
            final_manifest.append({'target': args.target, 'method': method_name, 'budget': f'B{args.budget_per_class}', 'train_seed': train_seed, 'config': cfg_path.as_posix(), 'source_subset': subset.as_posix(), 'search_json': json_path.as_posix(), 'category': 'dirichlet_ablation'})
            final_configs.append((f'{method_name}_seed{train_seed:02d}', cfg_path, out_dir))
    save_json(paths['search_root'] / 'final_train_manifest.json', final_manifest)
    if args.run_final_train:
        run_final_training(final_configs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=TARGETS)
    parser.add_argument('--budget-per-class', type=int, default=15)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--search-seed', type=int, default=0)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--eval-seeds', default='0')
    parser.add_argument('--refine-seeds', default='0')
    parser.add_argument('--final-train-seeds', default='0')
    parser.add_argument('--split-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--rank-root', default='data_cls_revise/source_subsets/officehome')
    parser.add_argument('--search-output-root', default='experiments_cls_revise/officehome_tavo_search_dirichlet')
    parser.add_argument('--final-subset-root', default='data_cls_revise/source_subsets/officehome_tavo_dirichlet')
    parser.add_argument('--final-config-root', default='configs_cls_revise/officehome_tavo_dirichlet')
    parser.add_argument('--final-output-root', default='experiments_cls_revise/officehome_tavo_dirichlet')
    parser.add_argument('--popsize', type=int, default=20)
    parser.add_argument('--n-gen', type=int, default=12)
    parser.add_argument('--n-random', type=int, default=0, help='If 0, use popsize*n_gen random Dirichlet samples.')
    parser.add_argument('--eval-epochs', type=int, default=5)
    parser.add_argument('--refine-epochs', type=int, default=15)
    parser.add_argument('--refine-topk', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--final-batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--final-epochs', type=int, default=30)
    parser.add_argument('--emit-best-so-far', action='store_true')
    parser.add_argument('--run-final-train', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_dirichlet(args)
if __name__ == '__main__':
    main()
