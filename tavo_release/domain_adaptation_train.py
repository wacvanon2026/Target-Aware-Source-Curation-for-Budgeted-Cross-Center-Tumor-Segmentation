from __future__ import annotations

import argparse
import json
from pathlib import Path

from .domain_adaptation import read_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = read_config(args.config)
    missing = [name for name, value in cfg["splits"].items() if not Path(value).exists()]
    if missing:
        raise FileNotFoundError({"missing_splits": missing})
    if args.dry_run:
        print(json.dumps({"status": "ok", "mode": "dry_run", "dataset": cfg["dataset"], "method": cfg["method"], "budget": cfg["budget"]}, indent=2))
        return 0
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("install torch or rerun with --dry-run") from exc
    print(json.dumps({"status": "ready", "torch": torch.__version__, "dataset": cfg["dataset"], "method": cfg["method"], "budget": cfg["budget"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
