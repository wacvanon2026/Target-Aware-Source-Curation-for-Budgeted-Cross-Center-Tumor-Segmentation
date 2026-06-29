#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.classification.resnet50_cls import ResNet50Classifier
METHODS = ['kmeans', 'kcenter', 'facilitylocation', 'craig', 'targetmmd', 'targetgradmatch', 'glister', 'orient']
METHOD_DISPLAY = {'kmeans': 'KMeans-B', 'kcenter': 'KCenter-B', 'facilitylocation': 'FacilityLocation-B', 'craig': 'CRAIG-B', 'targetmmd': 'TargetMMD-B', 'targetgradmatch': 'TargetGradMatch-B', 'glister': 'GLISTER-B', 'orient': 'ORIENT-B'}
GRADIENT_METHODS = {'craig', 'targetgradmatch', 'glister'}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_items(path):
    items = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, label = line.rsplit(' ', 1)
            items.append({'path': img_path, 'label': label})
    return items

def write_items(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for item in items:
            f.write(f"{item['path']} {item['label']}\n")

def group_indices(items):
    grouped = defaultdict(list)
    for idx, item in enumerate(items):
        grouped[item['label']].append(idx)
    return dict(sorted(grouped.items()))

def normalize_rows(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def check_class_budget(source_items, target_items, budget_per_class):
    source_grouped = group_indices(source_items)
    target_grouped = group_indices(target_items)
    for label in sorted(target_grouped):
        if label not in source_grouped:
            raise ValueError(f'Target class {label} is missing from source pool.')
        if len(source_grouped[label]) < budget_per_class:
            raise ValueError(f'Source class {label} has {len(source_grouped[label])} samples, need budget_per_class={budget_per_class}.')

class OfficeListDataset(Dataset):

    def __init__(self, items, class_to_idx, transform):
        self.items = items
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item['path']).convert('RGB')
        img = self.transform(img)
        label = self.class_to_idx[item['label']]
        return (img, label, item['path'], item['label'])

def build_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def load_model(num_classes, pretrained, warmup_ckpt, device):
    model = ResNet50Classifier(num_classes=num_classes, pretrained=pretrained)
    if warmup_ckpt:
        state = torch.load(warmup_ckpt, map_location='cpu')
        state_dict = state.get('model_state', state)
        cleaned = {}
        for key, value in state_dict.items():
            if key.startswith('module.'):
                key = key[len('module.'):]
            cleaned[key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f'Loaded warmup checkpoint: {warmup_ckpt}')
        if missing:
            print(f'Missing keys: {len(missing)}')
        if unexpected:
            print(f'Unexpected keys: {len(unexpected)}')
    model.to(device)
    model.eval()
    return model

def resnet_features(backbone, imgs):
    x = backbone.conv1(imgs)
    x = backbone.bn1(x)
    x = backbone.relu(x)
    x = backbone.maxpool(x)
    x = backbone.layer1(x)
    x = backbone.layer2(x)
    x = backbone.layer3(x)
    x = backbone.layer4(x)
    x = backbone.avgpool(x)
    return torch.flatten(x, 1)

@torch.no_grad()
def extract_arrays(model, items, class_to_idx, batch_size, num_workers, device, need_gradients, gradient_mode):
    dataset = OfficeListDataset(items, class_to_idx, build_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == 'cuda')
    features = []
    gradients = []
    labels = []
    paths = []
    class_names = []
    backbone = model.model
    for imgs, ys, batch_paths, batch_classes in loader:
        imgs = imgs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)
        feat = resnet_features(backbone, imgs)
        logits = backbone.fc(feat)
        prob = torch.softmax(logits, dim=1)
        features.append(feat.cpu().numpy().astype(np.float32))
        labels.append(ys.cpu().numpy().astype(np.int64))
        paths.extend(list(batch_paths))
        class_names.extend(list(batch_classes))
        if need_gradients:
            if gradient_mode == 'full':
                diff = prob.clone()
                diff.scatter_add_(1, ys.view(-1, 1), torch.full((ys.numel(), 1), -1.0, device=device))
                weight_grad = torch.einsum('bc,bd->bcd', diff, feat).flatten(1)
                grad = torch.cat([weight_grad, diff], dim=1)
            elif gradient_mode == 'true_class':
                true_prob = prob.gather(1, ys.view(-1, 1))
                coeff = true_prob - 1.0
                grad = torch.cat([coeff * feat, coeff], dim=1)
            else:
                raise ValueError(f'Unknown gradient_mode: {gradient_mode}')
            grad = grad / (grad.norm(dim=1, keepdim=True) + 1e-12)
            gradients.append(grad.cpu().numpy().astype(np.float32))
    out = {'features': np.concatenate(features, axis=0), 'labels': np.concatenate(labels, axis=0), 'paths': paths, 'class_names': class_names}
    if need_gradients:
        out['gradients'] = np.concatenate(gradients, axis=0)
    return out

def load_or_extract(cache_dir, split_name, model, items, class_to_idx, args, need_gradients, device):
    cache_dir.mkdir(parents=True, exist_ok=True)
    feature_path = cache_dir / f'{split_name}_features.npy'
    label_path = cache_dir / f'{split_name}_labels.npy'
    path_path = cache_dir / f'{split_name}_paths.txt'
    class_path = cache_dir / f'{split_name}_classes.txt'
    grad_path = cache_dir / f'{split_name}_gradients_{args.gradient_mode}.npy'
    have_base = all((p.exists() for p in [feature_path, label_path, path_path, class_path]))
    have_grad = not need_gradients or grad_path.exists()
    if args.use_cache and have_base and have_grad:
        print(f'Loading cache for {split_name}: {cache_dir}')
        out = {'features': np.load(feature_path), 'labels': np.load(label_path), 'paths': path_path.read_text(encoding='utf-8').splitlines(), 'class_names': class_path.read_text(encoding='utf-8').splitlines()}
        if need_gradients:
            out['gradients'] = np.load(grad_path)
        return out
    print(f'Extracting arrays for {split_name}: {len(items)} images')
    out = extract_arrays(model=model, items=items, class_to_idx=class_to_idx, batch_size=args.batch_size, num_workers=args.num_workers, device=device, need_gradients=need_gradients, gradient_mode=args.gradient_mode)
    np.save(feature_path, out['features'])
    np.save(label_path, out['labels'])
    path_path.write_text('\n'.join(out['paths']), encoding='utf-8')
    class_path.write_text('\n'.join(out['class_names']), encoding='utf-8')
    if need_gradients:
        np.save(grad_path, out['gradients'])
    return out

def kmeans_select(x, k, seed):
    if len(x) <= k:
        return (list(range(len(x))), [0.0] * len(x))
    x_norm = normalize_rows(x)
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    cluster_labels = km.fit_predict(x_norm)
    selected = []
    scores = []
    for cluster_id in range(k):
        local = np.where(cluster_labels == cluster_id)[0]
        if len(local) == 0:
            continue
        dists = np.linalg.norm(x_norm[local] - km.cluster_centers_[cluster_id], axis=1)
        pick = int(local[np.argmin(dists)])
        selected.append(pick)
        scores.append(float(-np.min(dists)))
    if len(selected) < k:
        remaining = [idx for idx in range(len(x)) if idx not in selected]
        random.Random(seed).shuffle(remaining)
        selected.extend(remaining[:k - len(selected)])
        scores.extend([0.0] * (k - len(scores)))
    return (selected[:k], scores[:k])

def kcenter_select(x, k):
    if len(x) <= k:
        return (list(range(len(x))), [0.0] * len(x))
    x_norm = normalize_rows(x)
    center = x_norm.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.linalg.norm(x_norm - center, axis=1)))
    selected = [first]
    scores = [0.0]
    min_dist = pairwise_distances(x_norm, x_norm[[first]], metric='euclidean').reshape(-1)
    for _ in range(k - 1):
        pick = int(np.argmax(min_dist))
        selected.append(pick)
        scores.append(float(min_dist[pick]))
        new_dist = pairwise_distances(x_norm, x_norm[[pick]], metric='euclidean').reshape(-1)
        min_dist = np.minimum(min_dist, new_dist)
    return (selected, scores)

def facility_location_select(x, k):
    if len(x) <= k:
        return (list(range(len(x))), [0.0] * len(x))
    sim = np.maximum(cosine_similarity(normalize_rows(x)), 0.0).astype(np.float32)
    selected = []
    gains = []
    current = np.zeros(sim.shape[0], dtype=np.float32)
    available = np.ones(sim.shape[0], dtype=bool)
    for _ in range(k):
        gain = np.maximum(current[:, None], sim) - current[:, None]
        gain = gain.sum(axis=0)
        gain[~available] = -np.inf
        pick = int(np.argmax(gain))
        selected.append(pick)
        gains.append(float(gain[pick]))
        current = np.maximum(current, sim[:, pick])
        available[pick] = False
    return (selected, gains)

def target_mean_greedy(source_x, target_x, k):
    if len(source_x) <= k:
        return (list(range(len(source_x))), [0.0] * len(source_x))
    source_x = normalize_rows(source_x)
    target_mu = normalize_rows(target_x).mean(axis=0)
    target_mu = target_mu / (np.linalg.norm(target_mu) + 1e-12)
    selected = []
    scores = []
    running_sum = np.zeros(source_x.shape[1], dtype=np.float32)
    available = np.ones(source_x.shape[0], dtype=bool)
    for step in range(k):
        candidate_mean = (running_sum[None, :] + source_x) / float(step + 1)
        dists = np.linalg.norm(candidate_mean - target_mu[None, :], axis=1)
        dists[~available] = np.inf
        pick = int(np.argmin(dists))
        selected.append(pick)
        scores.append(float(-dists[pick]))
        running_sum += source_x[pick]
        available[pick] = False
    return (selected, scores)

def gradient_match_greedy(source_grad, target_grad, k):
    if len(source_grad) <= k:
        return (list(range(len(source_grad))), [0.0] * len(source_grad))
    source_grad = normalize_rows(source_grad)
    target_mu = normalize_rows(target_grad).mean(axis=0)
    target_mu = target_mu / (np.linalg.norm(target_mu) + 1e-12)
    selected = []
    scores = []
    selected_sum = np.zeros(source_grad.shape[1], dtype=np.float32)
    available = np.ones(source_grad.shape[0], dtype=bool)
    for step in range(k):
        candidate_mean = (selected_sum[None, :] + source_grad) / float(step + 1)
        residual_norm = np.linalg.norm(target_mu[None, :] - candidate_mean, axis=1)
        residual_norm[~available] = np.inf
        pick = int(np.argmin(residual_norm))
        selected.append(pick)
        scores.append(float(-residual_norm[pick]))
        selected_sum += source_grad[pick]
        available[pick] = False
    return (selected, scores)

def orient_flmi_select(source_x, target_x, k, eta):
    if len(source_x) <= k:
        return (list(range(len(source_x))), [0.0] * len(source_x))
    source_x = normalize_rows(source_x)
    target_x = normalize_rows(target_x)
    k_mat = np.maximum(cosine_similarity(source_x, source_x), 0.0).astype(np.float32)
    q_mat = np.maximum(cosine_similarity(source_x, target_x), 0.0).astype(np.float32)
    try:
        from submodlib.functions.facilityLocationMutualInformation import FacilityLocationMutualInformationFunction
        obj = FacilityLocationMutualInformationFunction(n=k_mat.shape[0], num_queries=q_mat.shape[1], data_sijs=k_mat, query_sijs=q_mat, magnificationEta=eta)
        result = obj.maximize(budget=int(k), optimizer='LazyGreedy', stopIfNegativeGain=False, show_progress=False)
        selected = []
        gains = []
        for elem in result:
            if isinstance(elem, tuple):
                selected.append(int(elem[0]))
                gains.append(float(elem[1]))
            else:
                selected.append(int(elem))
                gains.append(0.0)
        return (selected, gains)
    except Exception as exc:
        print(f'submodlib FLMI unavailable, using greedy coverage fallback: {exc}')
    selected = []
    gains = []
    source_cov = np.zeros(k_mat.shape[0], dtype=np.float32)
    target_cov = np.zeros(q_mat.shape[1], dtype=np.float32)
    available = np.ones(k_mat.shape[0], dtype=bool)
    for _ in range(k):
        source_gain = (np.maximum(source_cov[:, None], k_mat) - source_cov[:, None]).sum(axis=0)
        target_gain = (np.maximum(target_cov[:, None], q_mat.T) - target_cov[:, None]).sum(axis=0)
        total_gain = source_gain + eta * target_gain
        total_gain[~available] = -np.inf
        pick = int(np.argmax(total_gain))
        selected.append(pick)
        gains.append(float(total_gain[pick]))
        source_cov = np.maximum(source_cov, k_mat[:, pick])
        target_cov = np.maximum(target_cov, q_mat[pick, :])
        available[pick] = False
    return (selected, gains)

def select_one_class(method, src_idx, tgt_idx, val_idx, arrays, k, seed, eta):
    src_feat = arrays['source']['features'][src_idx]
    tgt_feat = arrays['target_train']['features'][tgt_idx]
    src_grad = arrays['source'].get('gradients', None)
    tgt_grad = arrays['target_train'].get('gradients', None)
    val_grad = arrays['target_val'].get('gradients', None) if arrays.get('target_val') else None
    if method == 'kmeans':
        return kmeans_select(src_feat, k, seed)
    if method == 'kcenter':
        return kcenter_select(src_feat, k)
    if method == 'facilitylocation':
        return facility_location_select(src_feat, k)
    if method == 'targetmmd':
        return target_mean_greedy(src_feat, tgt_feat, k)
    if method == 'craig':
        return facility_location_select(src_grad[src_idx], k)
    if method == 'targetgradmatch':
        return gradient_match_greedy(src_grad[src_idx], tgt_grad[tgt_idx], k)
    if method == 'glister':
        reference = val_grad[val_idx] if val_grad is not None and len(val_idx) > 0 else tgt_grad[tgt_idx]
        return gradient_match_greedy(src_grad[src_idx], reference, k)
    if method == 'orient':
        source_space = src_grad[src_idx] if src_grad is not None else src_feat
        target_space = tgt_grad[tgt_idx] if tgt_grad is not None else tgt_feat
        return orient_flmi_select(source_space, target_space, k, eta)
    raise ValueError(f'Unknown method: {method}')

def run_method(method, source_items, target_items, target_val_items, arrays, budget_per_class, seed, eta):
    source_grouped = group_indices(source_items)
    target_grouped = group_indices(target_items)
    val_grouped = group_indices(target_val_items) if target_val_items else {}
    selected_items = []
    ranking_rows = []
    global_rank = 0
    for label in sorted(target_grouped):
        src_idx = source_grouped[label]
        tgt_idx = target_grouped[label]
        val_idx = val_grouped.get(label, [])
        local_selected, local_scores = select_one_class(method=method, src_idx=src_idx, tgt_idx=tgt_idx, val_idx=val_idx, arrays=arrays, k=budget_per_class, seed=seed, eta=eta)
        for local_rank, (local_id, score) in enumerate(zip(local_selected, local_scores), start=1):
            item = source_items[src_idx[local_id]]
            selected_items.append(item)
            ranking_rows.append({'global_rank': global_rank, 'class_rank': local_rank, 'label': label, 'path': item['path'], 'score': score})
            global_rank += 1
    selected_items = sorted(selected_items, key=lambda x: (x['label'], x['path']))
    return (selected_items, ranking_rows)

def save_ranking(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['global_rank', 'class_rank', 'label', 'path', 'score'])
        writer.writeheader()
        writer.writerows(rows)

def parse_methods(raw_methods):
    if len(raw_methods) == 1 and raw_methods[0].lower() == 'all':
        return METHODS
    out = []
    for method in raw_methods:
        key = method.lower().replace('-', '').replace('_', '')
        aliases = {'kmeansb': 'kmeans', 'kcenterb': 'kcenter', 'facilitylocationb': 'facilitylocation', 'flb': 'facilitylocation', 'craigb': 'craig', 'targetmmdb': 'targetmmd', 'targetgradmatchb': 'targetgradmatch', 'glisterb': 'glister', 'orientb': 'orient'}
        method = aliases.get(key, method.lower())
        if method not in METHODS:
            raise ValueError(f'Unknown method {method}. Valid methods: {METHODS}')
        out.append(method)
    return out

def main():
    parser = argparse.ArgumentParser(description='Generate class-balanced OfficeHome source subsets for source-selection baselines.')
    parser.add_argument('--source-list', required=True)
    parser.add_argument('--target-train-list', required=True)
    parser.add_argument('--target-val-list', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--cache-dir', default=None)
    parser.add_argument('--methods', nargs='+', default=['all'])
    parser.add_argument('--budget-per-class', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num-classes', type=int, default=65)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--warmup-ckpt', default=None)
    parser.add_argument('--allow-random-head-gradients', action='store_true')
    parser.add_argument('--gradient-mode', choices=['true_class', 'full'], default='full')
    parser.add_argument('--orient-eta', type=float, default=1.0)
    parser.add_argument('--no-pretrained', action='store_true')
    parser.add_argument('--use-cache', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)
    methods = parse_methods(args.methods)
    need_gradients = bool(set(methods) & GRADIENT_METHODS or 'orient' in methods)
    if need_gradients and args.warmup_ckpt is None and (not args.allow_random_head_gradients):
        raise ValueError('Gradient-based selectors (CRAIG/TargetGradMatch/GLISTER/ORIENT-gradient) need --warmup-ckpt. Pass --allow-random-head-gradients only for smoke tests.')
    source_items = read_items(args.source_list)
    target_items = read_items(args.target_train_list)
    target_val_items = read_items(args.target_val_list) if args.target_val_list else []
    check_class_budget(source_items, target_items, args.budget_per_class)
    all_labels = sorted({x['label'] for x in source_items + target_items + target_val_items})
    class_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    if len(class_to_idx) != args.num_classes:
        print(f'Warning: expected {args.num_classes} classes, found {len(class_to_idx)}')
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / 'cache'
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Methods: {[METHOD_DISPLAY[x] for x in methods]}')
    print(f'Source images: {len(source_items)}')
    print(f'Target train images: {len(target_items)}')
    print(f'Target val images: {len(target_val_items)}')
    print(f'Budget: {args.budget_per_class} per class, total {args.budget_per_class * len(class_to_idx)}')
    model = load_model(num_classes=len(class_to_idx), pretrained=not args.no_pretrained, warmup_ckpt=args.warmup_ckpt, device=device)
    arrays = {'source': load_or_extract(cache_dir, 'source', model, source_items, class_to_idx, args, need_gradients, device), 'target_train': load_or_extract(cache_dir, 'target_train', model, target_items, class_to_idx, args, need_gradients, device)}
    if target_val_items:
        arrays['target_val'] = load_or_extract(cache_dir, 'target_val', model, target_val_items, class_to_idx, args, need_gradients, device)
    else:
        arrays['target_val'] = None
    manifest = {'source_list': args.source_list, 'target_train_list': args.target_train_list, 'target_val_list': args.target_val_list, 'warmup_ckpt': args.warmup_ckpt, 'budget_per_class': args.budget_per_class, 'seed': args.seed, 'num_classes': len(class_to_idx), 'gradient_mode': args.gradient_mode, 'methods': {}}
    for method in methods:
        display = METHOD_DISPLAY[method]
        print(f'\nSelecting {display}')
        selected, ranking = run_method(method=method, source_items=source_items, target_items=target_items, target_val_items=target_val_items, arrays=arrays, budget_per_class=args.budget_per_class, seed=args.seed, eta=args.orient_eta)
        stem = f'{display}_B{args.budget_per_class}_seed{args.seed:02d}'
        subset_path = output_dir / f'{stem}.txt'
        ranking_path = output_dir / 'rankings' / f'{stem}.csv'
        meta_path = output_dir / 'metadata' / f'{stem}.json'
        write_items(subset_path, selected)
        save_ranking(ranking_path, ranking)
        counts = {label: len(indices) for label, indices in group_indices(selected).items()}
        method_meta = {'method': display, 'method_key': method, 'subset_path': subset_path.as_posix(), 'ranking_path': ranking_path.as_posix(), 'num_selected': len(selected), 'class_counts': counts}
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with meta_path.open('w', encoding='utf-8') as f:
            json.dump(method_meta, f, indent=2)
        manifest['methods'][display] = method_meta
        print(f'Saved {subset_path} ({len(selected)} images)')
    with (output_dir / 'manifest.json').open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
if __name__ == '__main__':
    main()
