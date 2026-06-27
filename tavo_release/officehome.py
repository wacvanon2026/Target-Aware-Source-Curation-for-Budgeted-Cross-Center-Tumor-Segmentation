from __future__ import annotations

import zipfile
from pathlib import Path

from .common import download_file, list_images, stable_shuffle, stratified_split, write_json, write_lines


DOMAINS = ("Art", "Clipart", "Product", "Real_World")


def download_officehome(url: str, output_dir: str | Path, filename: str = "officehome.zip", overwrite: bool = False) -> Path:
    return download_file(url, Path(output_dir) / filename, overwrite=overwrite)


def extract_archive(archive: str | Path, output_dir: str | Path) -> Path:
    archive = Path(archive)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if archive.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out)
    else:
        raise ValueError(f"unsupported archive type: {archive}")
    return out


def image_label_pairs(domain_root: str | Path) -> list[tuple[str, str]]:
    pairs = []
    root = Path(domain_root)
    for image in list_images(root):
        if image.parent == root:
            continue
        pairs.append((str(image), image.parent.name))
    return sorted(pairs)


def build_splits(data_root: str | Path, output_root: str | Path, target_domain: str, seed: int = 42, source_ratios=(9, 1), target_ratios=(7, 1, 2)) -> dict[str, int]:
    data = Path(data_root)
    out = Path(output_root) / target_domain
    sources = [d for d in DOMAINS if d != target_domain]
    source_pairs = []
    for source in sources:
        source_pairs.extend(image_label_pairs(data / source))
    source_train, source_val = stratified_split(source_pairs, source_ratios, seed)
    target_train, target_val, target_test = stratified_split(image_label_pairs(data / target_domain), target_ratios, seed)
    write_pair_file(out / "source_train.txt", source_train)
    write_pair_file(out / "source_val.txt", source_val)
    write_pair_file(out / "target_train.txt", target_train)
    write_pair_file(out / "target_val.txt", target_val)
    write_pair_file(out / "target_test.txt", target_test)
    shuffled = stable_shuffle([path for path, _ in source_train], seed)
    for budget in (50, 150, 250, 500):
        allowed = set(shuffled[:budget])
        write_pair_file(out / "random" / f"random_{budget}.txt", [x for x in source_train if x[0] in allowed])
    summary = {
        "source_train": len(source_train),
        "source_val": len(source_val),
        "target_train": len(target_train),
        "target_val": len(target_val),
        "target_test": len(target_test),
    }
    write_json(out / "summary.json", summary)
    return summary


def write_pair_file(path: str | Path, pairs: list[tuple[str, str]]) -> Path:
    return write_lines(path, [f"{p} {label}" for p, label in pairs])


def build_train_command(config: str | Path, entrypoint: str = "python -m tavo_release.officehome_train") -> list[str]:
    return entrypoint.split() + ["--config", str(Path(config))]


def build_config(output: str | Path, split_dir: str | Path, output_dir: str | Path, backbone: str = "resnet50", epochs: int = 30, batch_size: int = 64) -> Path:
    cfg = {
        "dataset": "officehome",
        "splits": {
            "source_train": str(Path(split_dir) / "source_train.txt"),
            "source_val": str(Path(split_dir) / "source_val.txt"),
            "target_train": str(Path(split_dir) / "target_train.txt"),
            "target_val": str(Path(split_dir) / "target_val.txt"),
            "target_test": str(Path(split_dir) / "target_test.txt"),
        },
        "model": {"backbone": backbone},
        "training": {"epochs": epochs, "batch_size": batch_size, "output_dir": str(Path(output_dir))},
    }
    return write_json(output, cfg)
