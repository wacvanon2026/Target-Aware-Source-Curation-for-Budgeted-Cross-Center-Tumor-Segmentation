#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
CKPTS = ['best.pt', 'last.pt', 'last_best.pt']

def read_manifest(path: Path) -> list[dict]:
    if path.suffix == '.json':
        return json.loads(path.read_text())
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def parse_log_fallback(output_dir: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    log_path = output_dir / 'train.log'
    if not log_path.exists():
        return results
    pattern = re.compile('^(best\\.pt|last\\.pt|last_best\\.pt)\\s+Target Acc:\\s+([0-9.]+)')
    for line in log_path.read_text(errors='ignore').splitlines():
        match = pattern.search(line)
        if match:
            results[match.group(1)] = {'acc': float(match.group(2))}
    return results

def load_results(output_dir: Path) -> dict[str, dict[str, float]]:
    path = output_dir / 'final_results.json'
    if path.exists():
        return json.loads(path.read_text())
    return parse_log_fallback(output_dir)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='configs_cls_revise/officehome_da/manifest_seed00.csv')
    parser.add_argument('--out', default='analysis_cls_revise/officehome_da_seed0_summary.csv')
    args = parser.parse_args()
    rows = read_manifest(Path(args.manifest))
    out_rows = []
    for row in rows:
        output_dir = Path(row['output_dir'])
        results = load_results(output_dir)
        out = {'target': row['target'], 'method': row['method'], 'da_method': row['da_method'], 'budget': row['budget'], 'train_seed': row['train_seed'], 'config': row['config'], 'output_dir': row['output_dir'], 'complete': str((output_dir / 'final_results.json').exists())}
        for ckpt in CKPTS:
            stats = results.get(ckpt, {})
            prefix = ckpt.replace('.pt', '')
            out[f'{prefix}_acc'] = stats.get('acc', '')
            out[f'{prefix}_macro_f1'] = stats.get('macro_f1', '')
            out[f'{prefix}_balanced_acc'] = stats.get('balanced_acc', '')
        out_rows.append(out)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'Wrote {len(out_rows)} rows to {out_path}')
if __name__ == '__main__':
    main()
