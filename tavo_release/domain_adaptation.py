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
    "aada": "external/efficientvit/scripts/run_aada.py",
    "mme": "external/efficientvit/scripts/run_mme.py",
    "lada": "external/efficientvit/scripts/run_lada.py",
    "clue": "external/efficientvit/scripts/run_clue.py",
}
OFFICEHOME_ENTRYPOINTS = {
    "dann": "external/efficientvit/scripts_cls/train_cls.py",
    "aada": "external/efficientvit/scripts_cls/select_aada_2shot.py",
    "mme": "external/efficientvit/scripts_cls/select_mme_2shot.py",
    "lada": "external/efficientvit/scripts_cls/select_lada_2shot.py",
    "adamatch": "external/efficientvit/scripts_cls/select_adamatch_2shot.py",
}


def build_config(dataset: str, method: str, split_dir: str | Path, output_dir: str | Path, budget: int, output: str | Path) -> Path:
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
        dataset_id = str(cfg.get("nnunet_dataset_id", "DATASET_ID"))
        configuration = str(cfg.get("configuration", "2d"))
        fold = str(cfg.get("fold", 0))
        return ["nnUNetv2_train", dataset_id, configuration, fold, "-tr", impl["trainer"]]
    if dataset == "brats":
        return ["python", impl["entrypoint"], "--target", target, "--budget", budget]
    if dataset == "officehome":
        if method == "dann":
            return ["python", impl["entrypoint"], "--config", str(cfg.get("train_config", "configs/officehome_target_dann.json"))]
        return ["python", impl["entrypoint"], "--target", target, "--budget_per_class", str(cfg.get("budget_per_class", 2))]
    raise ValueError(dataset)


def read_config(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())
