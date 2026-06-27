from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


FORBIDDEN_TEXT = tuple(
    [
        "/" + "project" + "2" + "/",
        "/" + "scratch",
        "/" + "home" + "1" + "/",
        "/" + "home" + "/",
        "mi" + "aot",
        "xi" + "wenc",
        "rui" + "shan",
    ]
)
SKIP_DIRS = {"data", "datasets", "raw", "preprocessed", "results", "outputs", "runs", "logs", "checkpoints", "models", "weights", "wandb", ".git", "__pycache__", ".pytest_cache", ".mypy_cache"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".txt", ".yaml", ".yml", ".json", ".toml"}
COMMENT_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json", ".toml"}
BLOCKED_BINARY_SUFFIXES = {".pt", ".pth", ".ckpt", ".pkl", ".npy", ".npz", ".nii", ".gz", ".zip", ".tar", ".tgz", ".7z", ".h5", ".hdf5"}


@dataclass(frozen=True)
class Workspace:
    root: Path
    data: Path
    outputs: Path
    splits: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "Workspace":
        base = Path(root).expanduser().resolve()
        return cls(base, base / "data", base / "outputs", base / "splits")


def read_lines(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


def write_lines(path: str | Path, values: Iterable[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(str(v) for v in values) + "\n")
    return p


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, value) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return p


def write_csv(path: str | Path, rows: Sequence[dict]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return p


def list_images(root: str | Path) -> list[Path]:
    base = Path(root)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in base.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)


def list_cases(root: str | Path) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def stable_shuffle(values: Sequence[str], seed: int) -> list[str]:
    out = list(values)
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


def ratio_split(values: Sequence[str], ratios: Sequence[int], seed: int) -> list[list[str]]:
    shuffled = stable_shuffle(values, seed)
    total = sum(ratios)
    n = len(shuffled)
    cuts = []
    acc = 0
    for ratio in ratios[:-1]:
        acc += ratio
        cuts.append(round(n * acc / total))
    parts = []
    prev = 0
    for cut in cuts:
        parts.append(shuffled[prev:cut])
        prev = cut
    parts.append(shuffled[prev:])
    return parts


def stratified_split(items: Sequence[tuple[str, str]], ratios: Sequence[int], seed: int) -> list[list[tuple[str, str]]]:
    by_label: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        by_label.setdefault(item[1], []).append(item)
    parts = [[] for _ in ratios]
    for label in sorted(by_label):
        subparts = ratio_split([x[0] for x in by_label[label]], ratios, seed)
        for idx, paths in enumerate(subparts):
            parts[idx].extend((path, label) for path in paths)
    return [sorted(part) for part in parts]


def symlink_or_copy(src: str | Path, dst: str | Path, copy: bool = False) -> Path:
    source = Path(src).resolve()
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()
    if copy:
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())
    return target


def download_file(url: str, dst: str | Path, overwrite: bool = False) -> Path:
    target = Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return target
    tmp = target.with_suffix(target.suffix + ".tmp")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(target)
    return target


def run_command(cmd: Sequence[str], cwd: str | Path | None = None, dry_run: bool = False, env: dict[str, str] | None = None) -> int:
    if dry_run:
        print(" ".join(str(x) for x in cmd))
        return 0
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run([str(x) for x in cmd], cwd=cwd, env=merged, check=True).returncode


def skipped(path: Path, base: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(base).parts[:-1])


def scannable_files(root: str | Path, suffixes: set[str] | None = None) -> list[Path]:
    base = Path(root)
    files = []
    for path in base.rglob("*"):
        if skipped(path, base):
            continue
        if not path.is_file():
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        files.append(path)
    return files


def scan_for_forbidden_paths(root: str | Path) -> list[tuple[str, str]]:
    hits = []
    base = Path(root)
    for path in scannable_files(base, TEXT_SUFFIXES):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for needle in FORBIDDEN_TEXT:
            if needle in text:
                hits.append((str(path.relative_to(base)), needle))
    return hits


def scan_for_large_or_binary(root: str | Path, max_bytes: int = 1_000_000) -> list[tuple[str, int]]:
    hits = []
    base = Path(root)
    for path in scannable_files(base):
        size = path.stat().st_size
        if size > max_bytes or path.suffix.lower() in BLOCKED_BINARY_SUFFIXES or path.name.endswith(".nii.gz"):
            hits.append((str(path.relative_to(base)), size))
    return hits


def scan_for_comment_syntax(root: str | Path) -> list[tuple[str, int, str]]:
    hits = []
    base = Path(root)
    doc_marker = '"' * 3
    for path in scannable_files(base, COMMENT_SUFFIXES):
        try:
            lines = path.read_text(errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            stripped = line.strip()
            if stripped.startswith("#") or doc_marker in line:
                hits.append((str(path.relative_to(base)), idx, stripped[:120]))
    return hits


def scan_git_authors(root: str | Path) -> list[tuple[str, str, str]]:
    base = Path(root)
    proc = subprocess.run(["git", "log", "--format=%H%x09%an%x09%ae", "--all"], cwd=base, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode != 0:
        return []
    hits = []
    for line in proc.stdout.splitlines():
        commit, name, email = line.split("\t", 2)
        if name != "Anonymous Authors" or email != "anonymous@example.com":
            hits.append((commit, name, email))
    return hits


def release_audit(root: str | Path) -> dict:
    base = Path(root)
    return {
        "forbidden_hits": scan_for_forbidden_paths(base),
        "large_files": scan_for_large_or_binary(base),
        "comment_hits": scan_for_comment_syntax(base),
        "git_author_hits": scan_git_authors(base),
    }
