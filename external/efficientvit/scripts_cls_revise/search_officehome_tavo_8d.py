#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.classification.dataset_office import OfficeDataset
from models.classification.resnet50_cls import ResNet50Classifier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ['Art', 'Clipart', 'Product', 'RealWorld']
METHODS = ['KMeans-B', 'KCenter-B', 'FacilityLocation-B', 'CRAIG-B', 'TargetMMD-B', 'TargetGradMatch-B', 'GLISTER-B', 'ORIENT-B']

def safe_name(name: str) -> str:
    return name.replace('-', '_').replace('+', 'Plus').replace(' ', '')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_items(path: Path):
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, label = line.rsplit(' ', 1)
            items.append({'path': img_path, 'label': label})
    return items

def write_items(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(f"{item['path']} {item['label']}\n")

def group_by_label(items):
    grouped = defaultdict(list)
    for item in items:
        grouped[item['label']].append(item)
    return dict(sorted(grouped.items()))

def build_transforms():
    train_transform = transforms.Compose([transforms.Resize(256), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return (train_transform, test_transform)

def load_rank_scores(rank_root: Path, budget_per_class: int):
    by_method: dict[str, dict[str, dict[str, float]]] = {}
    for method in METHODS:
        rank_path = rank_root / 'rankings' / f'{method}_B{budget_per_class}_seed00.csv'
        if not rank_path.exists():
            raise FileNotFoundError(rank_path)
        method_scores: dict[str, dict[str, float]] = defaultdict(dict)
        with rank_path.open('r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row['label']
                path = row['path']
                class_rank = int(row['class_rank'])
                score = (budget_per_class - class_rank + 1) / float(budget_per_class)
                method_scores[label][path] = float(score)
        by_method[method] = {k: dict(v) for k, v in method_scores.items()}
    return by_method

def project_simplex(z):
    z = np.asarray(z, dtype=np.float64)
    w = np.clip(z, 0.0, None)
    total = float(w.sum())
    if total <= 0:
        return np.ones_like(w) / len(w)
    return w / total

def rank_weights(mu):
    weights = np.array([math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)], dtype=np.float64)
    return weights / weights.sum()

def parse_int_list(raw: str):
    return [int(x) for x in raw.replace(',', ' ').split() if x.strip()]

def update_eigensystem(cov):
    cov = 0.5 * (cov + cov.T)
    eigvals, basis = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-12)
    scales = np.sqrt(eigvals)
    invsqrt = basis @ np.diag(1.0 / scales) @ basis.T
    return (basis, scales, invsqrt)

def weight_tag(weights: dict[str, float], iters: str):
    parts = []
    for method in METHODS:
        key = safe_name(method).replace('_B', '').lower()
        parts.append(f'{key}{weights[method]:.3f}')
    return '_'.join(parts) + f'_{iters}'

def build_tavo_subset(source_items, score_dicts, weights_vec, budget_per_class, out_txt: Path):
    weights_vec = project_simplex(weights_vec)
    weights = {METHODS[i]: float(weights_vec[i]) for i in range(len(METHODS))}
    source_by_label = group_by_label(source_items)
    selected = []
    rows = []
    for label, pool in source_by_label.items():
        scored = []
        for item in pool:
            total = 0.0
            for method, weight in weights.items():
                total += weight * score_dicts[method].get(label, {}).get(item['path'], 0.0)
            scored.append((total, item['path'], item))
        scored.sort(key=lambda x: (-x[0], x[1]))
        chosen = scored[:budget_per_class]
        if len(chosen) < budget_per_class:
            raise RuntimeError(f'{label} only has {len(chosen)} source items for B{budget_per_class}')
        for local_rank, (score, _, item) in enumerate(chosen, start=1):
            selected.append(item)
            rows.append({'class_rank': local_rank, 'label': label, 'path': item['path'], 'score': score})
    selected = sorted(selected, key=lambda x: (x['label'], x['path']))
    write_items(out_txt, selected)
    rank_csv = out_txt.parent / 'rankings' / f'{out_txt.stem}.csv'
    rank_csv.parent.mkdir(parents=True, exist_ok=True)
    with rank_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['class_rank', 'label', 'path', 'score'])
        writer.writeheader()
        writer.writerows(rows)
    return (selected, weights, rank_csv)

def build_final_config(target, method, source_train, target_selected, target_val, target_test, save_dir, train_seed, epochs, batch_size, num_workers, lr, weight_decay):
    budget_tag = source_train.parent.name
    return {'experiment': {'name': f'officehome_{target}_{method}_{budget_tag}_split00_train{train_seed:02d}', 'save_dir': save_dir.as_posix()}, 'data': {'num_classes': 65, 'source_train': source_train.as_posix(), 'source_val': target_val.as_posix(), 'target_selected': target_selected.as_posix(), 'target_test': target_test.as_posix(), 'batch_size': batch_size, 'num_workers': num_workers}, 'model': {'backbone': 'resnet50', 'pretrained': True, 'num_classes': 65}, 'optimizer': {'lr': lr, 'weight_decay': weight_decay}, 'scheduler': {'T_max': epochs}, 'training': {'seed': train_seed, 'epochs': epochs}}

def make_loaders(source_txt, target_train_txt, target_val_txt, batch_size, num_workers):
    train_transform, test_transform = build_transforms()
    source_ds = OfficeDataset(source_txt, transform=train_transform, return_path=False)
    target_ds = OfficeDataset(target_train_txt, transform=train_transform, class_to_idx=source_ds.class_to_idx, return_path=False)
    val_ds = OfficeDataset(target_val_txt, transform=test_transform, class_to_idx=source_ds.class_to_idx, return_path=False)
    train_ds = ConcatDataset([source_ds, target_ds])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    return (train_loader, val_loader)

def eval_accuracy(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            total_loss += float(loss.item()) * imgs.size(0)
            total_correct += int((preds == labels).sum().item())
            total_seen += int(imgs.size(0))
    return (total_loss / max(total_seen, 1), total_correct / max(total_seen, 1))

def short_train_val(source_txt, target_train_txt, target_val_txt, args, seed):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader = make_loaders(source_txt=source_txt, target_train_txt=target_train_txt, target_val_txt=target_val_txt, batch_size=args.batch_size, num_workers=args.num_workers)
    model = ResNet50Classifier(num_classes=65, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.eval_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')
    best_val = -1.0
    last_val = -1.0
    for epoch in range(args.eval_epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        _, last_val = eval_accuracy(model, val_loader, criterion, device)
        best_val = max(best_val, last_val)
        print(f'    eval_epoch={epoch + 1}/{args.eval_epochs} val_acc={last_val:.4f}')
    return (float(best_val), float(last_val))

def aggregate_short_train_val(source_txt, target_train_txt, target_val_txt, args, seeds):
    per_seed = {}
    best_vals = []
    last_vals = []
    for seed in seeds:
        best_val, last_val = short_train_val(source_txt=source_txt, target_train_txt=target_train_txt, target_val_txt=target_val_txt, args=args, seed=seed)
        per_seed[int(seed)] = {'best_val_acc': float(best_val), 'last_val_acc': float(last_val)}
        best_vals.append(float(best_val))
        last_vals.append(float(last_val))
    return {'fitness': float(np.median(best_vals)), 'mean_best_val_acc': float(np.mean(best_vals)), 'last_val_acc': float(np.median(last_vals)), 'per_seed': per_seed}

def run_candidate(source_items, score_dicts, weights_vec, args, paths, iter_tag, eval_id, seeds):
    weights_vec = project_simplex(weights_vec)
    weights = {METHODS[i]: float(weights_vec[i]) for i in range(len(METHODS))}
    tag = weight_tag(weights, iter_tag)
    subset_dir = paths['search_subset_root'] / tag
    subset_txt = subset_dir / f'TAVO_candidate_{eval_id:04d}.txt'
    selected, weights, rank_csv = build_tavo_subset(source_items=source_items, score_dicts=score_dicts, weights_vec=weights_vec, budget_per_class=args.budget_per_class, out_txt=subset_txt)
    agg = aggregate_short_train_val(source_txt=subset_txt, target_train_txt=paths['target_train'], target_val_txt=paths['target_val'], args=args, seeds=seeds)
    return {'id': int(eval_id), 'iter_tag': iter_tag, 'weights': weights, 'z': weights_vec.tolist(), 'fitness': float(agg['fitness']), 'mean_best_val_acc': float(agg['mean_best_val_acc']), 'last_val_acc': float(agg['last_val_acc']), 'per_seed': agg['per_seed'], 'subset_path': subset_txt.as_posix(), 'ranking_path': rank_csv.as_posix(), 'selected_count': len(selected)}

def save_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)

def finalize_subset(record, target, method_name, paths, args):
    final_dir = paths['final_subset_root'] / target / f'seed{args.split_seed:02d}' / f'B{args.budget_per_class}'
    final_txt = final_dir / f'{method_name}_B{args.budget_per_class}_searchseed{args.search_seed:02d}.txt'
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(record['subset_path'], final_txt)
    meta_dir = final_dir / 'metadata'
    meta_dir.mkdir(parents=True, exist_ok=True)
    save_json(meta_dir / f'{method_name}_B{args.budget_per_class}_searchseed{args.search_seed:02d}.json', record)
    return final_txt

def write_final_config(target, method_name, source_subset, paths, args, train_seed):
    method_key = safe_name(method_name)
    cfg_dir = paths['final_config_root'] / target / method_key / f'B{args.budget_per_class}' / f'split{args.split_seed:02d}'
    out_dir = paths['final_output_root'] / target / method_key / f'B{args.budget_per_class}' / f'split{args.split_seed:02d}' / f'train_seed{train_seed:02d}'
    cfg_path = cfg_dir / f'train_seed{train_seed:02d}.yaml'
    cfg = build_final_config(target=target, method=method_key, source_train=source_subset, target_selected=paths['target_train'], target_val=paths['target_val'], target_test=paths['target_test'], save_dir=out_dir, train_seed=train_seed, epochs=args.final_epochs, batch_size=args.final_batch_size, num_workers=args.num_workers, lr=args.lr, weight_decay=args.weight_decay)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with cfg_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return (cfg_path, out_dir)

def run_final_training(configs):
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    env['PYTHONUNBUFFERED'] = '1'
    for method_name, cfg_path, out_dir in configs:
        last_best = out_dir / 'last_best.pt'
        print(f'\n===== Final training {method_name} =====')
        print(f'config={cfg_path}')
        print(f'output={out_dir}')
        if last_best.exists():
            print(f'Skipping completed final training: {last_best}')
            continue
        subprocess.run(['python', '-u', 'scripts_cls/train_cls.py', '--config', cfg_path.as_posix()], cwd=PROJECT_ROOT, env=env, check=True)

def build_paths(args):
    split_dir = PROJECT_ROOT / args.split_root / args.target / f'seed{args.split_seed:02d}'
    rank_root = PROJECT_ROOT / args.rank_root / args.target / f'seed{args.split_seed:02d}' / f'B{args.budget_per_class}'
    search_root = PROJECT_ROOT / args.search_output_root / args.target / f'B{args.budget_per_class}' / f'search_seed{args.search_seed:02d}'
    return {'source_train': split_dir / 'source_train.txt', 'target_train': split_dir / 'target_train_3shot.txt', 'target_val': split_dir / 'target_val_2shot.txt', 'target_test': split_dir / 'target_test.txt', 'rank_root': rank_root, 'search_root': search_root, 'search_subset_root': search_root / 'candidate_subsets', 'final_subset_root': PROJECT_ROOT / args.final_subset_root, 'final_config_root': PROJECT_ROOT / args.final_config_root, 'final_output_root': PROJECT_ROOT / args.final_output_root}

def run_search(args):
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
    lam = args.popsize
    mu = args.mu
    if mu > lam:
        raise ValueError('--mu cannot exceed --popsize')
    json_path = paths['search_root'] / 'stageA2_cma.json'
    partial_path = paths['search_root'] / 'stageA2_cma_partial.json'
    paths['search_root'].mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.search_seed)
    rank_w = rank_weights(mu)
    mueff = 1.0 / np.sum(rank_w ** 2)
    cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
    cs = (mueff + 2) / (dim + mueff + 5)
    c1 = 2 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
    chi_n = math.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim * dim))
    mean = np.ones(dim) / dim
    sigma = args.sigma0
    cov = np.eye(dim)
    ps = np.zeros(dim)
    pc = np.zeros(dim)
    basis, scales, invsqrt = update_eigensystem(cov)
    all_evals = []
    gens = []
    best_so_far = None
    eval_id = 0
    print('===== OfficeHome TAVO 8D search =====')
    print(f'target={args.target} B={args.budget_per_class} search_seed={args.search_seed}')
    print(f'methods={METHODS}')
    print(f'popsize={lam} mu={mu} n_gen={args.n_gen} eval_epochs={args.eval_epochs}')
    eval_seeds = parse_int_list(args.eval_seeds)
    refine_seeds = parse_int_list(args.refine_seeds)
    final_train_seeds = parse_int_list(args.final_train_seeds)
    print(f'eval_seeds={eval_seeds} refine_seeds={refine_seeds} final_train_seeds={final_train_seeds}')
    if args.dry_run:
        print('\nDry run: building one uniform-weight subset and final configs only.')
        uniform = np.ones(dim, dtype=np.float64) / dim
        subset_txt = paths['search_subset_root'] / 'dry_run_uniform' / 'TAVO_candidate_dry_run.txt'
        selected, weights, rank_csv = build_tavo_subset(source_items=source_items, score_dicts=score_dicts, weights_vec=uniform, budget_per_class=args.budget_per_class, out_txt=subset_txt)
        record = {'id': -1, 'iter_tag': 'dry_run', 'weights': weights, 'fitness': None, 'subset_path': subset_txt.as_posix(), 'ranking_path': rank_csv.as_posix(), 'selected_count': len(selected)}
        final_subset = finalize_subset(record, args.target, 'TAVO_8D', paths, args)
        cfg_path, out_dir = write_final_config(args.target, 'TAVO_8D', final_subset, paths, args, args.train_seed)
        payload = {'cfg': vars(args), 'methods': METHODS, 'dry_run_record': record, 'final_config': cfg_path.as_posix(), 'final_output': out_dir.as_posix()}
        save_json(paths['search_root'] / 'dry_run.json', payload)
        print(f'Dry run selected_count={len(selected)} subset={subset_txt}')
        print(f'Dry run config={cfg_path}')
        return
    corners = []
    for i in range(dim):
        v = np.zeros(dim)
        v[i] = 1.0
        corners.append(v)
    corners.append(np.ones(dim) / dim)
    print('\nEvaluating corners and uniform...')
    for vec in corners:
        record = run_candidate(source_items, score_dicts, vec, args, paths, 'eval', eval_id, eval_seeds)
        record['gen'] = -1
        all_evals.append(record)
        if best_so_far is None or record['fitness'] > best_so_far['fitness']:
            best_so_far = record
        print(f"[corner] id={eval_id} fitness={record['fitness']:.4f} weights={record['weights']}")
        eval_id += 1
        if args.max_evals and eval_id >= args.max_evals:
            break
    for gen in range(args.n_gen):
        if args.max_evals and eval_id >= args.max_evals:
            break
        print(f'\n--- Generation {gen} sigma={sigma:.4f} mean={project_simplex(mean)} ---')
        arz = rng.normal(size=(lam, dim))
        ary = arz * scales @ basis.T
        arx = mean[None, :] + sigma * ary
        evals = []
        for k in range(lam):
            if args.max_evals and eval_id >= args.max_evals:
                break
            record = run_candidate(source_items, score_dicts, arx[k], args, paths, f'gen{gen}', eval_id, eval_seeds)
            record['gen'] = gen
            evals.append(record)
            all_evals.append(record)
            if record['fitness'] > best_so_far['fitness']:
                best_so_far = record
            print(f"[gen {gen} cand {k}] id={eval_id} fitness={record['fitness']:.4f}")
            eval_id += 1
        if not evals:
            break
        evals.sort(key=lambda x: x['fitness'], reverse=True)
        parents = evals[:min(mu, len(evals))]
        if len(parents) < mu:
            break
        zmat = np.array([p['z'] for p in parents])
        mean_old = mean.copy()
        mean = np.sum(zmat * rank_w[:, None], axis=0)
        y_w = (mean - mean_old) / max(sigma, 1e-12)
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (invsqrt @ y_w)
        ps_norm = np.linalg.norm(ps)
        hsig = 1.0 if ps_norm / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w
        y = (zmat - mean_old[None, :]) / max(sigma, 1e-12)
        rank_mu = np.zeros((dim, dim))
        for i in range(mu):
            yi = y[i][:, None]
            rank_mu += rank_w[i] * (yi @ yi.T)
        cov = (1 - c1 - cmu) * cov + c1 * (pc[:, None] @ pc[None, :] + (1 - hsig) * cc * (2 - cc) * cov) + cmu * rank_mu
        sigma = max(args.sigma_min, sigma * math.exp(cs / damps * (ps_norm / chi_n - 1.0)))
        basis, scales, invsqrt = update_eigensystem(cov)
        eigvals = np.linalg.eigvalsh(cov)
        gens.append({'gen': gen, 'sigma': float(sigma), 'mean_z': mean.tolist(), 'mean_proj': project_simplex(mean).tolist(), 'eigvals': eigvals.tolist(), 'best_fitness_gen': float(parents[0]['fitness']), 'best_fitness_global': float(best_so_far['fitness'])})
        save_json(partial_path, {'cfg': vars(args), 'methods': METHODS, 'gens': gens, 'best_so_far': best_so_far, 'n_eval': eval_id})
    all_sorted = sorted(all_evals, key=lambda x: x['fitness'], reverse=True)
    refine_inputs = all_sorted[:args.refine_topk]
    if best_so_far['id'] not in {r['id'] for r in refine_inputs}:
        refine_inputs.append(best_so_far)
    print('\nRefining top candidates...')
    old_eval_epochs = args.eval_epochs
    args.eval_epochs = args.refine_epochs
    refine_records = []
    for rank, record in enumerate(refine_inputs):
        vec = np.array([record['weights'][method] for method in METHODS], dtype=np.float64)
        refined = run_candidate(source_items, score_dicts, vec, args, paths, 'refine', eval_id, refine_seeds)
        refined['origin_eval_id'] = record['id']
        refined['origin_fitness'] = record['fitness']
        refined['gen'] = 'refine'
        refine_records.append(refined)
        print(f"[refine] origin={record['id']} refined_id={eval_id} fitness={refined['fitness']:.4f}")
        eval_id += 1
    args.eval_epochs = old_eval_epochs
    refine_records.sort(key=lambda x: x['fitness'], reverse=True)
    best_origin_refine = next((r for r in refine_records if r.get('origin_eval_id') == best_so_far['id']), None)
    if best_origin_refine is None:
        best_origin_refine = best_so_far
    out_payload = {'cfg': vars(args), 'methods': METHODS, 'gens': gens, 'all_evals': all_evals, 'best_so_far': best_so_far, 'top3': refine_records[:3], 'refine_records': refine_records, 'tavo_8d_record': refine_records[0], 'tavo_8d_best_record': best_origin_refine}
    save_json(json_path, out_payload)
    print(f'\nSaved search JSON: {json_path}')
    final_manifest = []
    final_configs = []
    variants = [('TAVO_8D', refine_records[0]), ('TAVO_8D_best', best_origin_refine)]
    for method_name, record in variants:
        subset = finalize_subset(record, args.target, method_name, paths, args)
        for train_seed in final_train_seeds:
            cfg_path, out_dir = write_final_config(args.target, method_name, subset, paths, args, train_seed)
            final_manifest.append({'target': args.target, 'method': method_name, 'budget': f'B{args.budget_per_class}', 'train_seed': train_seed, 'config': cfg_path.as_posix(), 'source_subset': subset.as_posix(), 'search_json': json_path.as_posix(), 'category': 'tavo'})
            final_configs.append((f'{method_name}_seed{train_seed:02d}', cfg_path, out_dir))
    manifest_path = paths['search_root'] / 'final_train_manifest.json'
    save_json(manifest_path, final_manifest)
    print(f'Saved final train manifest: {manifest_path}')
    if args.run_final_train:
        run_final_training(final_configs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True, choices=TARGETS)
    parser.add_argument('--budget-per-class', type=int, default=25)
    parser.add_argument('--split-seed', type=int, default=0)
    parser.add_argument('--search-seed', type=int, default=0)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--eval-seeds', default='0')
    parser.add_argument('--refine-seeds', default='0')
    parser.add_argument('--final-train-seeds', default='0')
    parser.add_argument('--split-root', default='data_cls_revise/splits/officehome')
    parser.add_argument('--rank-root', default='data_cls_revise/source_subsets/officehome')
    parser.add_argument('--search-output-root', default='experiments_cls_revise/officehome_tavo_search')
    parser.add_argument('--final-subset-root', default='data_cls_revise/source_subsets/officehome_tavo')
    parser.add_argument('--final-config-root', default='configs_cls_revise/officehome_tavo')
    parser.add_argument('--final-output-root', default='experiments_cls_revise/officehome')
    parser.add_argument('--popsize', type=int, default=12)
    parser.add_argument('--mu', type=int, default=6)
    parser.add_argument('--n-gen', type=int, default=5)
    parser.add_argument('--sigma0', type=float, default=0.3)
    parser.add_argument('--sigma-min', type=float, default=0.05)
    parser.add_argument('--eval-epochs', type=int, default=3)
    parser.add_argument('--refine-epochs', type=int, default=8)
    parser.add_argument('--refine-topk', type=int, default=4)
    parser.add_argument('--max-evals', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--final-batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--final-epochs', type=int, default=30)
    parser.add_argument('--run-final-train', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    os.chdir(PROJECT_ROOT)
    run_search(args)
if __name__ == '__main__':
    main()
