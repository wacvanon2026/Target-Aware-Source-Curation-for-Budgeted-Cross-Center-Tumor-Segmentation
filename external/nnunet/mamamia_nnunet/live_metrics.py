#!/usr/bin/env python3
"""Summarize live MAMAMIA nnUNet training metrics from Slurm logs."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.parent.name == "nnunet" and _SCRIPT_DIR.parent.parent.name == "external":
    PROJECT_ROOT = _SCRIPT_DIR.parents[2]
else:
    PROJECT_ROOT = _SCRIPT_DIR.parents[1]
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs" / "mamamia"

EPOCH_RE = re.compile(r": Epoch (\d+)$")
JOB_RE = re.compile(r"^(?P<name>.+)_(?P<job_id>\d+)\.out$")


@dataclass
class QueueEntry:
    job_id: str
    name: str
    state: str
    elapsed: str
    reason: str


@dataclass
class Metrics:
    log_path: Path
    job_id: str
    name: str
    state: str = ""
    elapsed: str = ""
    epoch: str = ""
    learning_rate: str = ""
    train_loss: str = ""
    val_loss: str = ""
    pseudo_dice: str = ""
    best_ema_pseudo_dice: str = ""
    test_dice: str = ""
    predictions: str = ""
    last_stage: str = ""


def prediction_progress(name: str) -> str:
    short_name = name.replace("mamamia_", "", 1)
    if "_" not in short_name:
        return ""
    target, experiment = short_name.split("_", 1)
    target_lc = target.lower()
    project_root = PROJECT_ROOT
    pred_dir = (
        project_root
        / "outputs"
        / f"tavo_mamamia_{target_lc}_nnunet_{experiment}"
        / "repeat_01"
        / "test_preds"
    )
    split_path = PROJECT_ROOT / "splits" / "mamamia_lodo_seed42" / target / "target_test.txt"
    if not pred_dir.exists():
        return ""

    done = len(list(pred_dir.glob("*.nii.gz")))
    total = ""
    if split_path.exists():
        total = str(sum(1 for line in split_path.read_text().splitlines() if line.strip()))
    return f"{done}/{total}" if total else str(done)


def queue_entries() -> dict[str, QueueEntry]:
    try:
        output = subprocess.check_output(
            ["squeue", "-u", subprocess.check_output(["whoami"], text=True).strip(), "-h", "-o", "%i|%j|%T|%M|%R"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    entries: dict[str, QueueEntry] = {}
    for line in output.splitlines():
        parts = line.split("|", 4)
        if len(parts) != 5:
            continue
        job_id, name, state, elapsed, reason = parts
        entries[job_id] = QueueEntry(job_id, name, state, elapsed, reason)
    return entries


def parse_log(path: Path, queue: dict[str, QueueEntry]) -> Metrics:
    match = JOB_RE.match(path.name)
    name = match.group("name") if match else path.stem
    job_id = match.group("job_id") if match else ""
    entry = queue.get(job_id)
    metrics = Metrics(
        log_path=path,
        job_id=job_id,
        name=name,
        state=entry.state if entry else "",
        elapsed=entry.elapsed if entry else "",
        predictions=prediction_progress(name),
    )

    for line in path.read_text(errors="ignore").splitlines():
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            metrics.epoch = epoch_match.group(1)
        if "Current learning rate:" in line:
            metrics.learning_rate = line.split("Current learning rate:", 1)[1].strip()
        elif ": train_loss " in line:
            metrics.train_loss = line.split(": train_loss ", 1)[1].strip()
        elif ": val_loss " in line:
            metrics.val_loss = line.split(": val_loss ", 1)[1].strip()
        elif ": Pseudo dice " in line:
            metrics.pseudo_dice = line.split(": Pseudo dice ", 1)[1].strip()
        elif "New best EMA pseudo Dice:" in line:
            metrics.best_ema_pseudo_dice = line.split("New best EMA pseudo Dice:", 1)[1].strip()
        elif "Test Dice (foreground_mean):" in line:
            metrics.test_dice = line.split("Test Dice (foreground_mean):", 1)[1].strip()
        elif "===== " in line:
            metrics.last_stage = line.split("===== ", 1)[1].rsplit(" =====", 1)[0]

    return metrics


def latest_logs(log_dir: Path, limit: int, include_finished: bool, queue: dict[str, QueueEntry]) -> list[Path]:
    paths = sorted(log_dir.glob("mamamia_*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    if include_finished:
        return paths[:limit]

    active_ids = set(queue)
    active_paths = []
    for path in paths:
        match = JOB_RE.match(path.name)
        if match and match.group("job_id") in active_ids:
            active_paths.append(path)
    return active_paths[:limit]


def print_table(rows: list[Metrics]) -> None:
    headers = (
        "job",
        "state",
        "time",
        "name",
        "epoch",
        "lr",
        "train",
        "val",
        "pseudo",
        "best",
        "test",
        "preds",
        "stage",
    )
    data = [
        (
            row.job_id,
            row.state,
            row.elapsed,
            row.name.replace("mamamia_", "", 1),
            row.epoch,
            row.learning_rate,
            row.train_loss,
            row.val_loss,
            row.pseudo_dice,
            row.best_ema_pseudo_dice,
            row.test_dice,
            row.predictions,
            row.last_stage,
        )
        for row in rows
    ]
    widths = [len(header) for header in headers]
    for row in data:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    def fmt(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(width) for value, width in zip(row, widths))

    print(fmt(headers))
    print(fmt(tuple("-" * width for width in widths)))
    for row in data:
        print(fmt(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--limit", type=int, default=30, help="Maximum logs to report.")
    parser.add_argument("--all", action="store_true", help="Include finished logs, not only active Slurm jobs.")
    args = parser.parse_args()

    log_dir = args.log_dir.expanduser().resolve()
    if not log_dir.exists():
        raise SystemExit(f"Missing log directory: {log_dir}")

    queue = queue_entries()
    paths = latest_logs(log_dir, args.limit, args.all, queue)
    rows = [parse_log(path, queue) for path in paths]
    if not rows:
        print("No matching MAMAMIA nnUNet logs found.")
        return
    print_table(rows)


if __name__ == "__main__":
    main()
