from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


TRAINER_FILES = (
    "nnUNetTrainerTAVOSaveEveryEpoch.py",
    "nnUNetTrainerTAVODomainAlignment.py",
)


def trainer_source_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "external" / "nnunet" / "mamamia_nnunet"


def installed_trainer_dir() -> Path:
    import nnunetv2.training.nnUNetTrainer as trainer_pkg

    paths = list(getattr(trainer_pkg, "__path__", []))
    if not paths:
        raise RuntimeError("could not locate installed nnunetv2.training.nnUNetTrainer package")
    return Path(paths[0])


def install_trainers() -> list[Path]:
    src_dir = trainer_source_dir()
    dst_dir = installed_trainer_dir()
    copied = []
    for name in TRAINER_FILES:
        src = src_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst = dst_dir / name
        try:
            if not dst.exists() or src.read_bytes() != dst.read_bytes():
                shutil.copy2(src, dst)
            copied.append(dst)
        except PermissionError as exc:
            raise PermissionError(
                f"Cannot install {src.name} into {dst_dir}. "
                "Use a writable Python environment or copy the files manually."
            ) from exc
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_id")
    parser.add_argument("configuration")
    parser.add_argument("fold")
    parser.add_argument("-tr", "--trainer", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--only-install-trainers", action="store_true")
    args, extra = parser.parse_known_args(argv)

    copied = install_trainers()
    for path in copied:
        print(f"Installed nnU-Net trainer: {path}")
    if args.only_install_trainers:
        return 0

    cmd = ["nnUNetv2_train", args.dataset_id, args.configuration, args.fold, "-tr", args.trainer]
    if args.continue_training:
        cmd.append("--c")
    if args.device:
        cmd.extend(["-device", args.device])
    cmd.extend(extra)
    print("Launching:", " ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
