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
    "dann": "external/efficientvit/scripts/run_dann.py",
    "mmd": "external/efficientvit/scripts/run_mmd.py",
    "advent": "external/efficientvit/scripts/run_advent.py",
    "seasa": "external/efficientvit/scripts/run_seasa.py",
}
OFFICEHOME_ENTRYPOINTS = {
    "dann": "external/efficientvit/scripts_cls/train_cls.py",
    "mmd": "external/efficientvit/scripts_cls/train_cls.py",
    "coral": "external/efficientvit/scripts_cls/train_cls.py",
    "cdan": "external/efficientvit/scripts_cls/train_cls.py",
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
        "training": {"output_dir": str(Path(output_dir)), "save_checkpoints": True},
    }
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
        return ["nnUNetv2_train", dataset_id, configuration, fold, "-tr", impl["trainer"]]
    if dataset == "brats":
        return ["python", impl["entrypoint"], "--target", target, "--budget", budget]
    if dataset == "officehome":
        return ["python", impl["entrypoint"], "--method", method, "--target", target, "--budget_per_class", str(cfg.get("budget_per_class", 2))]
    raise ValueError(dataset)


def read_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
