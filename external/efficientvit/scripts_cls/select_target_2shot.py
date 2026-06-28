#!/usr/bin/env python3
import os
import random
import argparse
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms, models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
IMAGE_EXTS = ('.jpg', '.jpeg', '.png')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def l2_normalize_np(x, eps=1e-08):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)

def read_split_txt(txt_path):
    items = []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            path, label = line.rsplit(' ', 1)
            items.append({'path': path, 'label': label})
    return items

def save_selected_txt(items, out_path):
    with open(out_path, 'w') as f:
        for it in items:
            f.write(f"{it['path']} {it['label']}\n")

def group_by_label(items):
    g = defaultdict(list)
    for it in items:
        g[it['label']].append(it)
    return dict(sorted(g.items(), key=lambda x: x[0]))

class OfficeListDataset(Dataset):

    def __init__(self, items, transform=None, label_to_idx=None):
        self.items = items
        self.transform = transform
        labels = sorted(set([x['label'] for x in items]))
        if label_to_idx is None:
            self.label_to_idx = {c: i for i, c in enumerate(labels)}
        else:
            self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item['path']).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        y = self.label_to_idx[item['label']]
        return (img, y, item['path'], item['label'])

def build_eval_transform():
    return transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

class ResNet50FeatureExtractor(nn.Module):

    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(model.children())[:-1])
        self.out_dim = model.fc.in_features

    def forward(self, x):
        feat = self.backbone(x)
        feat = feat.flatten(1)
        return feat

@torch.no_grad()
def extract_embeddings(feature_model, items, batch_size, num_workers, transform, label_to_idx=None):
    dataset = OfficeListDataset(items, transform=transform, label_to_idx=label_to_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats = []
    labels = []
    paths = []
    class_names = []
    feature_model.eval()
    for imgs, ys, ps, cls_names in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        z = feature_model(imgs)
        feats.append(z.cpu())
        labels.append(ys.cpu())
        paths.extend(list(ps))
        class_names.extend(list(cls_names))
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return (feats, labels, paths, class_names, dataset.label_to_idx)

class MetricProjector(nn.Module):

    def __init__(self, in_dim, hidden_dim=512, out_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z, dim=1)
        return z

class LabelBalancedBatchSampler(Sampler):

    def __init__(self, labels, classes_per_batch=8, samples_per_class=4, seed=2025):
        self.labels = np.array(labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = classes_per_batch * samples_per_class
        self.rng = random.Random(seed)
        self.cls_to_indices = defaultdict(list)
        for i, y in enumerate(self.labels):
            self.cls_to_indices[int(y)].append(i)
        self.classes = list(self.cls_to_indices.keys())
        self.num_batches = max(1, len(labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = self.rng.sample(self.classes, k=min(self.classes_per_batch, len(self.classes)))
            batch = []
            for c in chosen_classes:
                idxs = self.cls_to_indices[c]
                if len(idxs) >= self.samples_per_class:
                    picked = self.rng.sample(idxs, self.samples_per_class)
                else:
                    picked = [self.rng.choice(idxs) for _ in range(self.samples_per_class)]
                batch.extend(picked)
            yield batch

    def __len__(self):
        return self.num_batches

class EmbeddingTensorDataset(Dataset):

    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

def supervised_contrastive_loss(features, labels, temperature=0.07):
    device = features.device
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits = torch.matmul(features, features.T) / temperature
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=device)
    mask = mask * logits_mask
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = mask.sum(dim=1)
    valid = pos_count > 0
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (pos_count + 1e-12)
    loss = -mean_log_prob_pos[valid].mean()
    return loss

def train_metric_projector_on_source(source_embeddings, source_labels, epochs=50, lr=0.001, wd=0.0001, hidden_dim=512, out_dim=128, classes_per_batch=8, samples_per_class=4, seed=2025):
    dataset = EmbeddingTensorDataset(source_embeddings, source_labels)
    sampler = LabelBalancedBatchSampler(labels=source_labels, classes_per_batch=classes_per_batch, samples_per_class=samples_per_class, seed=seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    projector = MetricProjector(in_dim=source_embeddings.shape[1], hidden_dim=hidden_dim, out_dim=out_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=wd)
    projector.train()
    for epoch in range(epochs):
        losses = []
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            z = projector(x)
            loss = supervised_contrastive_loss(z, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'[Metric] epoch {epoch + 1:03d}/{epochs} loss={np.mean(losses):.4f}')
    projector.eval()
    return projector

@torch.no_grad()
def project_embeddings(projector, embeddings, batch_size=1024):
    xs = torch.tensor(embeddings, dtype=torch.float32)
    outs = []
    projector.eval()
    for i in range(0, len(xs), batch_size):
        x = xs[i:i + batch_size].to(DEVICE)
        z = projector(x)
        outs.append(z.cpu())
    return torch.cat(outs, dim=0).numpy()

def select_random_per_class(items, k, seed):
    rng = random.Random(seed)
    grouped = group_by_label(items)
    selected = []
    for cls in sorted(grouped.keys()):
        pool = grouped[cls]
        if len(pool) < k:
            raise ValueError(f'class {cls} has only {len(pool)} samples, need {k}')
        picked = rng.sample(pool, k)
        selected.extend(picked)
    return sorted(selected, key=lambda x: (x['label'], x['path']))

def select_kmeans_per_class(items, embeddings, k, seed):
    grouped = group_by_label(items)
    path_to_idx = {it['path']: i for i, it in enumerate(items)}
    selected = []
    for cls in sorted(grouped.keys()):
        cls_items = grouped[cls]
        idx = [path_to_idx[it['path']] for it in cls_items]
        x = l2_normalize_np(embeddings[idx])
        if len(cls_items) < k:
            raise ValueError(f'class {cls} has only {len(cls_items)} samples, need {k}')
        if len(cls_items) == k:
            selected.extend(cls_items)
            continue
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(x)
        centers = km.cluster_centers_
        picked_local = []
        for c in range(k):
            ids = np.where(labels == c)[0]
            if len(ids) == 0:
                continue
            d = np.linalg.norm(x[ids] - centers[c], axis=1)
            best = ids[np.argmin(d)]
            picked_local.append(best)
        if len(picked_local) < k:
            remain = [i for i in range(len(cls_items)) if i not in picked_local]
            random.shuffle(remain)
            picked_local.extend(remain[:k - len(picked_local)])
        for j in picked_local[:k]:
            selected.append(cls_items[j])
    return sorted(selected, key=lambda x: (x['label'], x['path']))

def kcenter_greedy_select(x, k):
    n = len(x)
    if n <= k:
        return list(range(n))
    center = x.mean(axis=0, keepdims=True)
    first_idx = int(np.argmin(np.linalg.norm(x - center, axis=1)))
    selected = [first_idx]
    dist = pairwise_distances(x, x[[first_idx]], metric='euclidean')
    for _ in range(k - 1):
        min_dist = dist.min(axis=1)
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dist = pairwise_distances(x, x[[next_idx]], metric='euclidean')
        dist = np.minimum(dist, new_dist)
    return selected

def select_coreset_per_class(items, embeddings, k):
    grouped = group_by_label(items)
    path_to_idx = {it['path']: i for i, it in enumerate(items)}
    selected = []
    for cls in sorted(grouped.keys()):
        cls_items = grouped[cls]
        idx = [path_to_idx[it['path']] for it in cls_items]
        x = l2_normalize_np(embeddings[idx])
        if len(cls_items) < k:
            raise ValueError(f'class {cls} has only {len(cls_items)} samples, need {k}')
        local_idx = kcenter_greedy_select(x, k)
        for j in local_idx:
            selected.append(cls_items[j])
    return sorted(selected, key=lambda x: (x['label'], x['path']))

def select_metric_coreset_per_class(items, metric_embeddings, k):
    return select_coreset_per_class(items, metric_embeddings, k)
from sklearn.metrics.pairwise import cosine_similarity

def facility_location_greedy(x, k):
    n = len(x)
    if n <= k:
        return list(range(n))
    sim = cosine_similarity(x)
    selected = []
    current = np.full(n, -1000000000.0, dtype=np.float32)
    for _ in range(k):
        best_gain = -1e+18
        best_idx = -1
        for j in range(n):
            if j in selected:
                continue
            new_cov = np.maximum(current, sim[:, j])
            gain = new_cov.sum() - current.sum()
            if gain > best_gain:
                best_gain = gain
                best_idx = j
        selected.append(best_idx)
        current = np.maximum(current, sim[:, best_idx])
    return selected

def select_craig_proxy_per_class(items, embeddings, k):
    grouped = group_by_label(items)
    path_to_idx = {it['path']: i for i, it in enumerate(items)}
    selected = []
    for cls in sorted(grouped.keys()):
        cls_items = grouped[cls]
        idx = [path_to_idx[it['path']] for it in cls_items]
        x = embeddings[idx]
        if len(cls_items) < k:
            raise ValueError(f'class {cls} has only {len(cls_items)} samples, need {k}')
        local_idx = facility_location_greedy(x, k)
        for j in local_idx:
            selected.append(cls_items[j])
    return sorted(selected, key=lambda x: (x['label'], x['path']))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_train_txt', type=str, required=True)
    parser.add_argument('--target_train_txt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--use_saved_embeddings', action='store_true')
    parser.add_argument('--budget_per_class', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--metric_epochs', type=int, default=30)
    parser.add_argument('--metric_lr', type=float, default=0.001)
    parser.add_argument('--metric_wd', type=float, default=0.0001)
    parser.add_argument('--metric_hidden_dim', type=int, default=512)
    parser.add_argument('--metric_out_dim', type=int, default=128)
    args = parser.parse_args()
    set_seed(args.seed)
    ensure_dir(args.output_dir)
    source_items = read_split_txt(args.source_train_txt)
    target_items = read_split_txt(args.target_train_txt)
    print(f'source_train: {len(source_items)}')
    print(f'target_train: {len(target_items)}')
    src_embed_path = os.path.join(args.output_dir, 'source_embeddings.npy')
    tgt_embed_path = os.path.join(args.output_dir, 'target_embeddings.npy')
    if args.use_saved_embeddings and os.path.exists(src_embed_path) and os.path.exists(tgt_embed_path):
        print('Loading saved embeddings...')
        src_emb = np.load(src_embed_path)
        tgt_emb = np.load(tgt_embed_path)
        print('Loaded source embeddings:', src_emb.shape)
        print('Loaded target embeddings:', tgt_emb.shape)
    else:
        print('Loading ImageNet-pretrained ResNet50 feature extractor...')
        feature_model = ResNet50FeatureExtractor(pretrained=True).to(DEVICE)
        transform = build_eval_transform()
        print('Extracting SOURCE embeddings...')
        src_emb, src_y, src_paths, src_cls, label_to_idx = extract_embeddings(feature_model, source_items, batch_size=args.batch_size, num_workers=args.num_workers, transform=transform, label_to_idx=None)
        print('Extracting TARGET embeddings...')
        tgt_emb, tgt_y, tgt_paths, tgt_cls, _ = extract_embeddings(feature_model, target_items, batch_size=args.batch_size, num_workers=args.num_workers, transform=transform, label_to_idx=label_to_idx)
        np.save(src_embed_path, src_emb)
        np.save(tgt_embed_path, tgt_emb)
    k = args.budget_per_class
    print('Selecting CRAIG-like proxy...')
    craig_sel = select_craig_proxy_per_class(target_items, tgt_emb, k)
    save_selected_txt(craig_sel, os.path.join(args.output_dir, f'craig_proxy_{k}shot.txt'))
    print('\nDone.')
    print(f'Saved selections under: {args.output_dir}')
    print('Note: MME/DANN are NOT sample selectors and should be run as separate training baselines.')
if __name__ == '__main__':
    main()
