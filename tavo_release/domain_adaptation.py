from __future__ import annotations

import json
from pathlib import Path

from .common import write_json
from .matrix import dataset_methods


MAMAMIA_TRAINERS = {
    "dann": "nnUNetTrainerTAVODANN",
    "mmd": "nnUNetTrainerTAVOMMD",
    "advent": "nnUNetTrainerTAVOADVENT",
    "seasa": "nnUNetTrainerTAVOSEASA",
}
BRATS_ENTRYPOINTS = {
    "dann": "external/efficientvit/scripts/train_seg_da.py",
    "mmd": "external/efficientvit/scripts/train_seg_da.py",
    "advent": "external/efficientvit/scripts/train_seg_da.py",
    "seasa": "external/efficientvit/scripts/train_seg_da.py",
}
OFFICEHOME_ENTRYPOINTS = {
    "dann": "external/efficientvit/scripts_cls/train_cls_da.py",
    "mmd": "external/efficientvit/scripts_cls/train_cls_da.py",
    "coral": "external/efficientvit/scripts_cls/train_cls_da.py",
    "cdan": "external/efficientvit/scripts_cls/train_cls_da.py",
}

BRATS_DA_METHODS = {
    "dann": "dann",
    "mmd": "dan_mmd",
    "advent": "advent_advent",
    "seasa": "se_asa",
}


def build_config(dataset: str, method: str, split_dir: str | Path, output_dir: str | Path, budget: int, output: str | Path, nnunet_dataset_id: int | str | None = None, target: str | None = None) -> Path:
    if method not in dataset_methods(dataset, "domain_adaptation"):
        raise ValueError(f"unknown domain adaptation method: {method}")
    cfg = {
        "dataset": dataset,
        "method": method,
        "budget": int(budget),
        "splits": {
            "source_train": str(Path(split_dir) / f"{method}_{budget}_source_train.txt"),
            "target_train": str(Path(split_dir) / "target_train.txt"),
            "target_val": str(Path(split_dir) / "target_val.txt"),
            "target_test": str(Path(split_dir) / "target_test.txt"),
        },
        "implementation": implementation(dataset, method),
        "training": {"seed": 0, "epochs": 30, "output_dir": str(Path(output_dir)), "save_checkpoints": True},
    }
    if dataset == "officehome":
        cfg.update({
            "experiment": {"name": f"officehome_{target or 'target'}_{method}_{budget}", "save_dir": str(Path(output_dir))},
            "data": {
                "num_classes": 65,
                "source_train": cfg["splits"]["source_train"],
                "source_val": cfg["splits"]["target_val"],
                "target_selected": cfg["splits"]["target_train"],
                "target_test": cfg["splits"]["target_test"],
                "batch_size": 64,
                "num_workers": 8,
            },
            "model": {"backbone": "resnet50", "pretrained": True, "num_classes": 65},
            "optimizer": {"lr": 3e-4, "weight_decay": 1e-4},
            "scheduler": {"T_max": 30},
            "da": {
                "method": method,
                "lambda_max": 0.1,
                "target_ce_weight": 1.0,
                "steps_per_epoch": "auto",
                "domain_hidden_dim": 1024,
                "domain_dropout": 0.5,
                "kernel_multipliers": [0.25, 0.5, 1.0, 2.0, 4.0],
                "fixed_sigma": "auto",
                "entropy_conditioning": True,
                "random_dim": 1024,
            },
        })
    if dataset == "brats":
        cfg.update(brats_efficientvit_config(method, target or "target", split_dir, output_dir, output, cfg["splits"]))
    if target is not None:
        cfg["target"] = target
    if nnunet_dataset_id is not None:
        cfg["nnunet_dataset_id"] = int(nnunet_dataset_id)
    return write_json(output, cfg)


def implementation(dataset: str, method: str) -> dict[str, str]:
    if dataset == "mamamia":
        return {"backend": "nnunet", "trainer": MAMAMIA_TRAINERS[method], "source_module": "tavo_release.mamamia_nnunet_trainers"}
    if dataset == "brats":
        return {"backend": "efficientvit", "entrypoint": BRATS_ENTRYPOINTS[method]}
    if dataset == "officehome":
        return {"backend": "classification", "entrypoint": OFFICEHOME_ENTRYPOINTS[method]}
    raise ValueError(dataset)


def build_train_command(config: str | Path) -> list[str]:
    cfg = read_config(config)
    impl = cfg["implementation"]
    dataset = cfg["dataset"]
    method = cfg["method"]
    budget = str(cfg["budget"])
    target = cfg.get("target", "target")
    if dataset == "mamamia":
        if "nnunet_dataset_id" not in cfg:
            raise ValueError("nnunet_dataset_id is required for MAMA-MIA domain adaptation")
        dataset_id = str(cfg["nnunet_dataset_id"])
        configuration = str(cfg.get("configuration", "2d"))
        fold = str(cfg.get("fold", 0))
        return ["python", "-m", "tavo_release.mamamia_nnunet_train", dataset_id, configuration, fold, "-tr", impl["trainer"]]
    if dataset == "brats":
        return ["env", "PYTHONPATH=external/efficientvit", "python", impl["entrypoint"], "--config", str(Path(config))]
    if dataset == "officehome":
        return ["env", "PYTHONPATH=external/efficientvit", "python", impl["entrypoint"], "--config", str(Path(config))]
    raise ValueError(dataset)


def read_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def brats_efficientvit_config(method: str, target: str, split_dir: str | Path, output_dir: str | Path, output: str | Path, splits: dict[str, str]) -> dict:
    sidecar = Path(output).with_suffix("")
    source_dir = write_brats_split_dir(sidecar / "source", "train", splits["source_train"])
    target_train_dir = write_brats_split_dir(sidecar / "target_train", "train", splits["target_train"])
    target_val_dir = write_brats_split_dir(sidecar / "target_val", "val", splits["target_val"])
    target_test_dir = write_brats_split_dir(sidecar / "target_test", "test", splits["target_test"])
    root = "external/efficientvit/data/002_BraTS21"
    da_cfg = {
        "method": BRATS_DA_METHODS[method],
        "lambda_max": 0.1,
        "lambda_schedule": "dann_logistic",
        "target_seg_weight": 1.0,
        "steps_per_epoch": "auto",
        "feature_layer": "backbone_last",
        "domain_hidden_dim": 256,
        "domain_dropout": 0.5,
        "kernel_multipliers": [0.25, 0.5, 1.0, 2.0, 4.0],
        "fixed_sigma": "auto",
    }
    if method == "advent":
        da_cfg.update({
            "lambda_max": 0.001,
            "lambda_schedule": "fixed",
            "lr_d": 1e-4,
            "beta1_d": 0.9,
            "beta2_d": 0.99,
            "output_discriminator_ndf": 64,
        })
    if method == "seasa":
        da_cfg.update({
            "lambda_max": 0.003,
            "lambda_schedule": "fixed",
            "lr_d": 1e-4,
            "beta1_d": 0.9,
            "beta2_d": 0.99,
            "output_discriminator_ndf": 64,
            "seasa_lambda_class": 0.1,
            "seasa_lambda_selective": 0.01,
            "seasa_class_center_momentum": 0.01,
            "seasa_num_aug": 3,
            "seasa_consistency_threshold": 2,
            "seasa_fourier_beta": 0.01,
            "seasa_noise_std": 0.03,
        })
    return {
        "model": {"name": "efficientvit_l1", "in_channels": 4, "num_classes": 4, "pretrained": True},
        "data": {
            "skip_empty_train": True,
            "skip_empty_val": False,
            "skip_empty_align": True,
            "source": {"name": f"BraTS21_{target}_{method}_source", "path": root, "split": "train", "split_txt": str(source_dir)},
            "target": {"name": f"BraTS21_{target}_target_train", "path": root, "split": "train", "split_txt": str(target_train_dir)},
            "target_align": {"name": f"BraTS21_{target}_target_align", "path": root, "split": "train", "split_txt": str(target_train_dir)},
            "val": {"path": root, "split": "val", "split_txt": str(target_val_dir)},
            "test": {"path": root, "split": "test", "split_txt": str(target_test_dir)},
            "img_size": 512,
            "batch_size": 4,
            "num_workers": 4,
        },
        "optimizer": {"lr": 1e-4, "weight_decay": 1e-5},
        "scheduler": {"T_max": 30, "eta_min": 1e-6},
        "training": {"epochs": 30, "seed": 0, "save_dir": str(Path(output_dir)), "auto_eval": False, "keep_epoch_checkpoints": False, "feature_mode": False},
        "da": da_cfg,
    }


def write_brats_split_dir(out_dir: Path, split: str, source_file: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source_file)
    if source.exists():
        (out_dir / f"{split}_subjects.txt").write_text(source.read_text())
    return out_dir
