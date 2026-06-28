from __future__ import annotations
import argparse
import json
from pathlib import Path

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    required = ['source_train', 'source_val', 'target_train', 'target_val', 'target_test']
    missing = [name for name in required if not Path(cfg['splits'][name]).exists()]
    if missing:
        raise FileNotFoundError({'missing_splits': missing})
    if args.dry_run:
        print(json.dumps({'status': 'ok', 'mode': 'dry_run', 'config': args.config}, indent=2))
        return 0
    try:
        import torch
        import torchvision
    except ImportError as exc:
        raise RuntimeError('install torch and torchvision or rerun with --dry-run') from exc
    print(json.dumps({'status': 'ready', 'torch': torch.__version__, 'torchvision': torchvision.__version__, 'config': args.config}, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
