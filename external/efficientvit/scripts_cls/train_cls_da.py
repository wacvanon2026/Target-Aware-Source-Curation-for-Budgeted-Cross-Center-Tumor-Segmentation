#!/usr/bin/env python3
"""Train OfficeHome classification DA baselines.

The supervised objective is source CE + labeled target_train CE. Alignment uses
only source_train and target_train. target_val selects checkpoints, and
target_test is evaluated only after training for best.pt, last.pt, last_best.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.classification.dataset_office import OfficeDataset
from models.classification.resnet50_cls import ResNet50Classifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, dropout: float = 0.5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        return self.net(GradientReversalFn.apply(x, alpha))


class RandomLayer(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, output_dim: int) -> None:
        super().__init__()
        self.register_buffer("rf", torch.randn(feature_dim, output_dim))
        self.register_buffer("rp", torch.randn(num_classes, output_dim))
        self.scale = math.sqrt(float(output_dim))

    def forward(self, features: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        return (features @ self.rf) * (probs @ self.rp) / self.scale


def dann_alpha(epoch: int, step: int, epochs: int, steps_per_epoch: int, lambda_max: float) -> float:
    total_steps = max(epochs * steps_per_epoch, 1)
    p = (epoch * steps_per_epoch + step) / total_steps
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, multipliers: list[float], fixed_sigma: float | None) -> torch.Tensor:
    z = torch.cat([x, y], dim=0).float()
    dist = torch.cdist(z, z, p=2).pow(2)
    if fixed_sigma is None:
        with torch.no_grad():
            sigma = dist.detach()
            sigma = sigma[sigma > 0].median().clamp_min(1e-6)
    else:
        sigma = torch.tensor(float(fixed_sigma), device=z.device).clamp_min(1e-6)
    kernels = [torch.exp(-dist / (2.0 * sigma * m)) for m in multipliers]
    return sum(kernels)


def mmd_loss(x: torch.Tensor, y: torch.Tensor, multipliers: list[float], fixed_sigma: float | None) -> torch.Tensor:
    n = min(x.size(0), y.size(0))
    x = x[:n]
    y = y[:n]
    k = gaussian_kernel(x, y, multipliers, fixed_sigma)
    k_xx = k[:n, :n]
    k_yy = k[n:, n:]
    k_xy = k[:n, n:]
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


def coral_loss(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    n = min(source.size(0), target.size(0))
    source = source[:n].float()
    target = target[:n].float()
    source = source - source.mean(dim=0, keepdim=True)
    target = target - target.mean(dim=0, keepdim=True)
    denom = max(n - 1, 1)
    cov_s = source.t().mm(source) / denom
    cov_t = target.t().mm(target) / denom
    d = source.size(1)
    return (cov_s - cov_t).pow(2).sum() / (4.0 * d * d)


def entropy_weights(logits: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    ent = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
    weights = 1.0 + torch.exp(-ent)
    return weights / weights.mean().detach().clamp_min(1e-6)


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.view(-1), labels.float(), reduction="none")
    if weights is not None:
        loss = loss * weights.view(-1)
    return loss.mean()


def macro_f1_and_balanced_acc(y_true: list[int], y_pred: list[int], num_classes: int) -> tuple[float, float]:
    f1_values = []
    recalls = []
    for cls_idx in range(num_classes):
        tp = sum(1 for y, p in zip(y_true, y_pred) if y == cls_idx and p == cls_idx)
        fp = sum(1 for y, p in zip(y_true, y_pred) if y != cls_idx and p == cls_idx)
        fn = sum(1 for y, p in zip(y_true, y_pred) if y == cls_idx and p != cls_idx)
        support = tp + fn
        if support == 0:
            continue
        precision = tp / max(tp + fp, 1)
        recall = tp / max(support, 1)
        f1_values.append(0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall))
        recalls.append(recall)
    macro_f1 = float(sum(f1_values) / max(len(f1_values), 1))
    balanced_acc = float(sum(recalls) / max(len(recalls), 1))
    return macro_f1, balanced_acc


def save_checkpoint(state: dict[str, Any], save_dir: Path, filename: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, save_dir / filename)
    print(f"Saved checkpoint: {filename}")


def load_da_datasets(cfg: dict[str, Any]):
    train_transform, test_transform = build_transforms()
    data = cfg["data"]
    source = OfficeDataset(data["source_train"], transform=train_transform, return_path=False)
    target_train = OfficeDataset(data["target_selected"], transform=train_transform, class_to_idx=source.class_to_idx, return_path=False)
    target_val = OfficeDataset(data["source_val"], transform=test_transform, class_to_idx=source.class_to_idx, return_path=False)
    target_test = OfficeDataset(data["target_test"], transform=test_transform, class_to_idx=source.class_to_idx, return_path=False)
    print(f"Loaded source_train={len(source)} target_train={len(target_train)} target_val={len(target_val)} target_test={len(target_test)}")
    print("source_train:", data["source_train"])
    print("target_train:", data["target_selected"])
    print("target_val:", data["source_val"])
    print("target_test:", data["target_test"])
    return source, target_train, target_val, target_test


def make_loader(dataset, cfg: dict[str, Any], shuffle: bool, drop_last: bool, generator: torch.Generator) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def evaluate(model: ResNet50Classifier, loader: DataLoader, criterion: nn.Module, device: torch.device, desc: str) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc, ncols=100):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_correct += (preds == labels).sum().item()
            total_samples += bs
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    macro_f1, balanced_acc = macro_f1_and_balanced_acc(y_true, y_pred, model.model.fc.out_features)
    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "macro_f1": macro_f1,
        "balanced_acc": balanced_acc,
    }


def train_one_epoch(
    *,
    model: ResNet50Classifier,
    domain_disc: DomainDiscriminator | None,
    random_layer: RandomLayer | None,
    source_loader: DataLoader,
    target_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    cfg: dict[str, Any],
    epoch: int,
) -> dict[str, float]:
    model.train()
    if domain_disc is not None:
        domain_disc.train()

    da_cfg = cfg["da"]
    method = da_cfg["method"].lower()
    epochs = cfg["training"]["epochs"]
    lambda_max = float(da_cfg.get("lambda_max", 0.1))
    steps_per_epoch = da_cfg.get("steps_per_epoch", "auto")
    if steps_per_epoch == "auto":
        steps = max(len(source_loader), len(target_loader))
    else:
        steps = int(steps_per_epoch)

    total = {
        "loss": 0.0,
        "source_ce": 0.0,
        "target_ce": 0.0,
        "da_loss": 0.0,
        "da_weight": 0.0,
        "source_correct": 0,
        "target_correct": 0,
        "samples": 0,
    }
    source_iter = cycle(source_loader)
    target_iter = cycle(target_loader)

    for step in tqdm(range(steps), desc="Training DA", ncols=100):
        src_imgs, src_labels = next(source_iter)
        tgt_imgs, tgt_labels = next(target_iter)
        src_imgs = src_imgs.to(device, non_blocking=True)
        src_labels = src_labels.to(device, non_blocking=True)
        tgt_imgs = tgt_imgs.to(device, non_blocking=True)
        tgt_labels = tgt_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        da_weight = dann_alpha(epoch, step, epochs, steps, lambda_max)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            src_logits, src_feats = model.forward_with_features(src_imgs)
            tgt_logits, tgt_feats = model.forward_with_features(tgt_imgs)
            source_ce = criterion(src_logits, src_labels)
            target_ce = criterion(tgt_logits, tgt_labels)

            if method == "dann":
                assert domain_disc is not None
                feats = torch.cat([src_feats, tgt_feats], dim=0)
                labels = torch.cat([
                    torch.zeros(src_feats.size(0), device=device),
                    torch.ones(tgt_feats.size(0), device=device),
                ])
                da_loss = weighted_bce(domain_disc(feats, da_weight), labels)
            elif method in {"mmd", "dan", "dan_mmd"}:
                da_loss = mmd_loss(
                    src_feats,
                    tgt_feats,
                    multipliers=[float(x) for x in da_cfg.get("kernel_multipliers", [0.25, 0.5, 1.0, 2.0, 4.0])],
                    fixed_sigma=None if da_cfg.get("fixed_sigma", "auto") == "auto" else float(da_cfg["fixed_sigma"]),
                )
            elif method in {"coral", "deep_coral", "deepcoral"}:
                da_loss = coral_loss(src_feats, tgt_feats)
            elif method in {"cdan", "cdan_e", "cdan+e"}:
                assert domain_disc is not None
                src_prob = F.softmax(src_logits, dim=1)
                tgt_prob = F.softmax(tgt_logits, dim=1)
                if random_layer is None:
                    src_joint = torch.bmm(src_prob.unsqueeze(2), src_feats.unsqueeze(1)).flatten(1)
                    tgt_joint = torch.bmm(tgt_prob.unsqueeze(2), tgt_feats.unsqueeze(1)).flatten(1)
                else:
                    src_joint = random_layer(src_feats, src_prob)
                    tgt_joint = random_layer(tgt_feats, tgt_prob)
                joint = torch.cat([src_joint, tgt_joint], dim=0)
                labels = torch.cat([
                    torch.zeros(src_joint.size(0), device=device),
                    torch.ones(tgt_joint.size(0), device=device),
                ])
                weights = None
                if bool(da_cfg.get("entropy_conditioning", True)):
                    weights = torch.cat([entropy_weights(src_logits), entropy_weights(tgt_logits)], dim=0)
                da_loss = weighted_bce(domain_disc(joint, da_weight), labels, weights)
            else:
                raise ValueError(f"Unknown DA method: {method}")

            supervised_loss = source_ce + float(da_cfg.get("target_ce_weight", 1.0)) * target_ce
            if method in {"dann", "cdan", "cdan_e", "cdan+e"}:
                loss = supervised_loss + da_loss
            else:
                loss = supervised_loss + da_weight * da_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = src_imgs.size(0) + tgt_imgs.size(0)
        total["loss"] += loss.item() * bs
        total["source_ce"] += source_ce.item() * src_imgs.size(0)
        total["target_ce"] += target_ce.item() * tgt_imgs.size(0)
        total["da_loss"] += da_loss.item() * bs
        total["da_weight"] += da_weight
        total["source_correct"] += (src_logits.argmax(dim=1) == src_labels).sum().item()
        total["target_correct"] += (tgt_logits.argmax(dim=1) == tgt_labels).sum().item()
        total["samples"] += bs

    source_seen = steps * source_loader.batch_size
    target_seen = steps * target_loader.batch_size
    return {
        "loss": total["loss"] / max(total["samples"], 1),
        "source_ce": total["source_ce"] / max(source_seen, 1),
        "target_ce": total["target_ce"] / max(target_seen, 1),
        "da_loss": total["da_loss"] / max(total["samples"], 1),
        "da_weight": total["da_weight"] / max(steps, 1),
        "source_acc": total["source_correct"] / max(source_seen, 1),
        "target_acc": total["target_correct"] / max(target_seen, 1),
    }


def main(config_path: str) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(int(cfg["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Config:", config_path)

    source_ds, target_ds, val_ds, test_ds = load_da_datasets(cfg)
    g = torch.Generator()
    g.manual_seed(int(cfg["training"]["seed"]))
    source_loader = make_loader(source_ds, cfg, shuffle=True, drop_last=True, generator=g)
    target_loader = make_loader(target_ds, cfg, shuffle=True, drop_last=True, generator=g)
    val_loader = make_loader(val_ds, cfg, shuffle=False, drop_last=False, generator=g)
    test_loader = make_loader(test_ds, cfg, shuffle=False, drop_last=False, generator=g)

    model = ResNet50Classifier(cfg["model"]["num_classes"], cfg["model"].get("pretrained", True)).to(device)
    da_method = cfg["da"]["method"].lower()
    feature_dim = model.feature_dim
    domain_disc: DomainDiscriminator | None = None
    random_layer: RandomLayer | None = None
    if da_method == "dann":
        domain_disc = DomainDiscriminator(feature_dim, cfg["da"].get("domain_hidden_dim", 1024), cfg["da"].get("domain_dropout", 0.5)).to(device)
    elif da_method in {"cdan", "cdan_e", "cdan+e"}:
        random_dim = int(cfg["da"].get("random_dim", 1024))
        random_layer = RandomLayer(feature_dim, cfg["model"]["num_classes"], random_dim).to(device)
        domain_disc = DomainDiscriminator(random_dim, cfg["da"].get("domain_hidden_dim", 1024), cfg["da"].get("domain_dropout", 0.5)).to(device)

    params = list(model.parameters())
    if domain_disc is not None:
        params += list(domain_disc.parameters())
    optimizer = optim.AdamW(params, lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["scheduler"]["T_max"])
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    save_dir = Path(cfg["experiment"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    history: list[dict[str, Any]] = []
    epochs = int(cfg["training"]["epochs"])

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        train_stats = train_one_epoch(
            model=model,
            domain_disc=domain_disc,
            random_layer=random_layer,
            source_loader=source_loader,
            target_loader=target_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            cfg=cfg,
            epoch=epoch,
        )
        val_stats = evaluate(model, val_loader, criterion, device, "Validating target_val")
        scheduler.step()
        print(
            f"Train loss {train_stats['loss']:.4f} | Source acc {train_stats['source_acc']:.4f} | "
            f"Target train acc {train_stats['target_acc']:.4f} | "
            f"DA raw {train_stats['da_loss']:.4f} | DA weight {train_stats['da_weight']:.4f} | "
            f"Val acc {val_stats['acc']:.4f} | Val macroF1 {val_stats['macro_f1']:.4f}"
        )
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "domain_state": domain_disc.state_dict() if domain_disc is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "train_stats": train_stats,
            "val_stats": val_stats,
            "val_acc": val_stats["acc"],
            "val_loss": val_stats["loss"],
            "config": cfg,
        }
        save_checkpoint(checkpoint, save_dir, f"epoch_{epoch + 1:03d}.pt")
        if val_stats["acc"] > best_metric:
            best_metric = val_stats["acc"]
            save_checkpoint(checkpoint, save_dir, "best.pt")
        history.append({"epoch": epoch + 1, "train": train_stats, "val": val_stats})
        with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    last_epoch = history[-1]["epoch"]
    shutil.copy(save_dir / f"epoch_{last_epoch:03d}.pt", save_dir / "last.pt")
    recent = history[-5:]
    last_best_epoch = sorted(recent, key=lambda x: (-x["val"]["acc"], x["val"]["loss"]))[0]["epoch"]
    shutil.copy(save_dir / f"epoch_{last_best_epoch:03d}.pt", save_dir / "last_best.pt")
    print(f"Last-best epoch = {last_best_epoch}")

    final_results: dict[str, dict[str, float]] = {}
    for ckpt_name in ["best.pt", "last.pt", "last_best.pt"]:
        print(f"\nEvaluating {ckpt_name} on target_test")
        ckpt = torch.load(save_dir / ckpt_name, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        stats = evaluate(model, test_loader, criterion, device, f"Testing {ckpt_name}")
        final_results[ckpt_name] = stats
        print(
            f"{ckpt_name} Target Acc: {stats['acc']:.4f} | "
            f"Macro-F1: {stats['macro_f1']:.4f} | Balanced Acc: {stats['balanced_acc']:.4f}"
        )

    with open(save_dir / "final_results.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print("\n===== FINAL RESULTS =====")
    for ckpt_name, stats in final_results.items():
        print(f"{ckpt_name:<12} acc={stats['acc']} macro_f1={stats['macro_f1']} balanced_acc={stats['balanced_acc']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
