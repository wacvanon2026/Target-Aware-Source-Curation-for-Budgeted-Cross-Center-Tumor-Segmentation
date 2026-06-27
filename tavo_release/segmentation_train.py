from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    paths = []
    data = cfg.get("data", {})
    for key in ("train_subjects", "val_subjects", "test_subjects"):
        if key in data:
            paths.append(Path(data[key]))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError({"missing": missing})
    if args.dry_run:
        print(json.dumps({"status": "ok", "mode": "dry_run", "config": args.config, "seeds": args.seeds}, indent=2))
        return 0
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install torch or rerun with --dry-run") from exc
    print(json.dumps({"status": "ready", "torch": torch.__version__, "config": args.config, "seeds": args.seeds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
