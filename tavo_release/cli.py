from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from . import brats, docs, domain_adaptation, mamamia, matrix, officehome, pathways, repro, selection_routes, tavo_routes
from .common import download_file, release_audit, run_command, scan_for_forbidden_paths, scan_for_large_or_binary, write_json
from .pipeline import audit_plan, write_plan
from .tavo import run_score_file_search, write_selection


def add_split_parser(subparsers):
    parser = subparsers.add_parser("split")
    parser.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--target")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def parse_score_args(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        name, path = item.split("=", 1)
        out[name] = path
    return out


def cmd_split(args):
    if args.dataset == "mamamia":
        return mamamia.build_lodo_splits(args.data_root, args.output_root, seed=args.seed)
    if args.dataset == "brats":
        if not args.target:
            raise SystemExit("--target is required for brats")
        return brats.build_domain_splits(args.data_root, args.output_root, args.target, seed=args.seed)
    if args.dataset == "officehome":
        if not args.target:
            raise SystemExit("--target is required for officehome")
        return officehome.build_splits(args.data_root, args.output_root, args.target, seed=args.seed)
    raise SystemExit(args.dataset)


def cmd_download(args):
    archive = download_file(args.url, Path(args.output_dir) / args.filename, overwrite=args.overwrite)
    extracted = None
    if args.extract:
        if args.dataset == "officehome":
            extracted = officehome.extract_archive(archive, Path(args.output_dir) / "extracted")
        else:
            extracted = "extract manually or provide a dataset-specific archive layout"
    return {"archive": str(archive), "extracted": str(extracted) if extracted else None}


def cmd_select(args):
    scores = parse_score_args(args.score)
    selected = write_selection(scores, np.asarray(args.weight, dtype=float), args.budget, args.output)
    return {"selected": len(selected), "output": args.output}


def cmd_search(args):
    scores = parse_score_args(args.score)
    result = run_score_file_search(scores, args.budget, args.output_dir, seed=args.seed, generations=args.generations, popsize=args.popsize)
    return result["best"]


def cmd_tavo_command(args):
    return {"search": tavo_routes.search_command(args.dataset, args.target, args.budget, args.output_dir, args.score_root)}


def cmd_selection_route(args):
    return selection_routes.selection_route(args.dataset, args.target, args.method, args.budget, source=args.source, pathways_path=args.pathways)


def cmd_route_inventory(args):
    routes = selection_routes.family_inventory(args.family, args.dataset, pathways_path=args.pathways)
    return {"count": len(routes), "routes": routes}


def cmd_route_audit(args):
    result = selection_routes.route_audit(args.pathways)
    if not result["ok"]:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def cmd_command(args):
    if args.dataset == "mamamia":
        commands = mamamia.nnunet_commands(mamamia.resolve_dataset(args.dataset_id), trainer=args.trainer, fold=args.fold, configuration=args.configuration)
        return commands
    if args.dataset == "officehome":
        return {"train": officehome.build_train_command(args.config)}
    if args.dataset == "brats":
        return {"train": brats.build_train_command(args.config, seeds=args.seeds)}
    raise SystemExit(args.dataset)


def cmd_matrix(args):
    if args.experiments:
        return {"experiments": matrix.all_experiments()}
    return matrix.method_matrix()


def cmd_da_config(args):
    path = domain_adaptation.build_config(args.dataset, args.method, args.split_dir, args.output_dir, args.budget, args.output, nnunet_dataset_id=args.nnunet_dataset_id, target=args.target)
    return {"config": str(path)}


def cmd_da_command(args):
    return {"train": domain_adaptation.build_train_command(args.config)}


def cmd_pathway_audit(args):
    result = pathways.audit_pathways(args.pathways)
    if not result["ok"]:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def cmd_collect(args):
    if args.dataset == "mamamia":
        rows = mamamia.collect_metric_jsons(args.results_root)
        if args.output:
            write_json(args.output, rows)
        return {"rows": len(rows), "output": args.output}
    rows = []
    for path in Path(args.results_root).rglob("*.json"):
        try:
            value = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            row = {"path": str(path)}
            row.update({k: v for k, v in value.items() if isinstance(v, int | float | str)})
            rows.append(row)
    if args.output:
        write_json(args.output, rows)
    return {"rows": len(rows), "output": args.output}


def make_toy_mamamia(root: Path):
    for domain, prefix in mamamia.DOMAINS.items():
        for idx in range(4):
            case = f"{prefix}{idx:03d}"
            case_dir = root / "images" / case
            case_dir.mkdir(parents=True, exist_ok=True)
            for channel in ("0000", "0001", "0002"):
                (case_dir / f"{case}_{channel}.nii.gz").write_text("synthetic")
            seg = root / "segmentations" / "expert"
            seg.mkdir(parents=True, exist_ok=True)
            (seg / f"{case}.nii.gz").write_text("synthetic")


def make_toy_officehome(root: Path):
    for domain in officehome.DOMAINS:
        for cls in ("a", "b"):
            d = root / domain / cls
            d.mkdir(parents=True, exist_ok=True)
            for idx in range(10):
                im = Image.new("RGB", (8, 8), (idx * 20, 40, 80))
                im.save(d / f"{idx}.png")


def make_scores(root: Path):
    ids = [f"case_{i}" for i in range(8)]
    paths = {}
    for idx, name in enumerate(("rds", "less", "orient")):
        values = {case: float((i + idx) % len(ids)) for i, case in enumerate(ids)}
        path = root / f"{name}.json"
        write_json(path, values)
        paths[name] = str(path)
    return paths


def cmd_smoke(args):
    base = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="tavo_release_smoke_"))
    base.mkdir(parents=True, exist_ok=True)
    mamamia_root = base / "mamamia_data"
    office_root = base / "officehome_data"
    make_toy_mamamia(mamamia_root)
    make_toy_officehome(office_root)
    mamamia_summary = mamamia.build_lodo_splits(mamamia_root, base / "splits" / "mamamia", seed=1)
    office_summary = officehome.build_splits(office_root, base / "splits" / "officehome", "Art", seed=1)
    raw = mamamia.materialize_nnunet_raw(mamamia_root, base / "splits" / "mamamia", base / "nnunet_raw", "NACT", "random", 50)
    mamamia_cmd = mamamia.nnunet_commands(mamamia.resolve_dataset("NACT:random50"))["train"]
    office_cfg = officehome.build_config(base / "officehome_config.json", base / "splits" / "officehome" / "Art", base / "officehome_output")
    office_cmd = officehome.build_train_command(office_cfg) + ["--dry-run"]
    brats_split = base / "splits" / "brats" / "target"
    brats_split.mkdir(parents=True, exist_ok=True)
    (brats_split / "train.txt").write_text("a\n")
    (brats_split / "val.txt").write_text("b\n")
    brats_cfg = base / "brats_config.yaml"
    brats_cfg.write_text("data:\n  train_subjects: " + str(brats_split / "train.txt") + "\n  val_subjects: " + str(brats_split / "val.txt") + "\n")
    brats_cmd = brats.build_train_command(brats_cfg) + ["--dry-run"]
    run_command(office_cmd)
    run_command(brats_cmd)
    score_paths = make_scores(base / "scores")
    selected = write_selection(score_paths, [0.5, 0.3, 0.2], 3, base / "selection.txt")
    best = run_score_file_search(score_paths, 3, base / "search", generations=2, popsize=4, seed=1)["best"]
    plan = write_plan("all", base / "plans")
    hits = scan_for_forbidden_paths(Path(__file__).resolve().parents[1])
    large = scan_for_large_or_binary(Path(__file__).resolve().parents[1])
    result = {
        "workdir": str(base),
        "mamamia_targets": sorted(mamamia_summary),
        "officehome": office_summary,
        "nnunet_raw_exists": raw.exists(),
        "mamamia_train_command": mamamia_cmd,
        "officehome_dry_run_command": office_cmd,
        "brats_dry_run_command": brats_cmd,
        "selected": selected,
        "best_score": best["score"],
        "plan_steps": plan["steps"],
        "forbidden_hits": hits,
        "large_files": large,
    }
    if hits or large:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def cmd_check(args):
    root = Path(args.root)
    result = release_audit(root)
    if any(result.values()):
        raise SystemExit(json.dumps(result, indent=2))
    return result


def cmd_plan(args):
    return write_plan(args.dataset, args.output_dir)


def cmd_plan_audit(args):
    result = audit_plan()
    if not result["ok"]:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def cmd_docs_audit(args):
    result = docs.readme_audit(args.readme)
    if not result["ok"]:
        raise SystemExit(json.dumps(result, indent=2))
    return result


def cmd_repro_smoke(args):
    return repro.run_repro_smoke(args.workdir)


def cmd_officehome_config(args):
    return {"config": str(officehome.build_config(args.output, args.split_dir, args.output_dir, backbone=args.backbone, epochs=args.epochs, batch_size=args.batch_size))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tavo-release")
    sub = parser.add_subparsers(dest="cmd", required=True)
    download = sub.add_parser("download")
    download.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    download.add_argument("--url", required=True)
    download.add_argument("--output-dir", required=True)
    download.add_argument("--filename", default="dataset.zip")
    download.add_argument("--extract", action="store_true")
    download.add_argument("--overwrite", action="store_true")
    add_split_parser(sub)
    select = sub.add_parser("select")
    select.add_argument("--score", action="append", required=True)
    select.add_argument("--weight", type=float, nargs="+", required=True)
    select.add_argument("--budget", type=int, required=True)
    select.add_argument("--output", required=True)
    search = sub.add_parser("search")
    search.add_argument("--score", action="append", required=True)
    search.add_argument("--budget", type=int, required=True)
    search.add_argument("--output-dir", required=True)
    search.add_argument("--seed", type=int, default=0)
    search.add_argument("--generations", type=int, default=12)
    search.add_argument("--popsize", type=int, default=20)
    tavo_command = sub.add_parser("tavo-command")
    tavo_command.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    tavo_command.add_argument("--target", required=True)
    tavo_command.add_argument("--budget", type=int, required=True)
    tavo_command.add_argument("--score-root", default="scores")
    tavo_command.add_argument("--output-dir")
    selection_route = sub.add_parser("selection-route")
    selection_route.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    selection_route.add_argument("--target", required=True)
    selection_route.add_argument("--method", required=True)
    selection_route.add_argument("--budget", type=int, required=True)
    selection_route.add_argument("--source", default="source")
    selection_route.add_argument("--pathways", default="configs/pathways.json")
    route_inventory = sub.add_parser("route-inventory")
    route_inventory.add_argument("--dataset", choices=["mamamia", "brats", "officehome", "all"], default="all")
    route_inventory.add_argument("--family", choices=["selection", "tavo", "domain_adaptation"], default="selection")
    route_inventory.add_argument("--pathways", default="configs/pathways.json")
    route_audit = sub.add_parser("route-audit")
    route_audit.add_argument("--pathways", default="configs/pathways.json")
    command = sub.add_parser("command")
    command.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    command.add_argument("--dataset-id", default="1301")
    command.add_argument("--trainer", default="nnUNetTrainer")
    command.add_argument("--fold", type=int, default=0)
    command.add_argument("--configuration", default="2d")
    command.add_argument("--config", default="config.yaml")
    command.add_argument("--seeds", default="0")
    da_config = sub.add_parser("da-config")
    da_config.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    da_config.add_argument("--method", required=True)
    da_config.add_argument("--split-dir", required=True)
    da_config.add_argument("--output-dir", required=True)
    da_config.add_argument("--budget", type=int, required=True)
    da_config.add_argument("--output", required=True)
    da_config.add_argument("--nnunet-dataset-id")
    da_config.add_argument("--target")
    da_command = sub.add_parser("da-command")
    da_command.add_argument("--config", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--dataset", choices=["mamamia", "brats", "officehome"], required=True)
    collect.add_argument("--results-root", required=True)
    collect.add_argument("--output")
    oh_cfg = sub.add_parser("officehome-config")
    oh_cfg.add_argument("--split-dir", required=True)
    oh_cfg.add_argument("--output-dir", required=True)
    oh_cfg.add_argument("--output", required=True)
    oh_cfg.add_argument("--backbone", default="resnet50")
    oh_cfg.add_argument("--epochs", type=int, default=30)
    oh_cfg.add_argument("--batch-size", type=int, default=64)
    plan = sub.add_parser("plan")
    plan.add_argument("--dataset", choices=["mamamia", "brats", "officehome", "all"], required=True)
    plan.add_argument("--output-dir", required=True)
    sub.add_parser("plan-audit")
    docs_audit = sub.add_parser("docs-audit")
    docs_audit.add_argument("--readme", default="README.md")
    repro_smoke = sub.add_parser("repro-smoke")
    repro_smoke.add_argument("--workdir")
    methods = sub.add_parser("matrix")
    methods.add_argument("--experiments", action="store_true")
    audit = sub.add_parser("pathway-audit")
    audit.add_argument("--pathways", default="configs/pathways.json")
    smoke = sub.add_parser("smoke")
    smoke.add_argument("--workdir")
    check = sub.add_parser("check")
    check.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args(argv)
    result = {
        "split": cmd_split,
        "download": cmd_download,
        "select": cmd_select,
        "search": cmd_search,
        "tavo-command": cmd_tavo_command,
        "selection-route": cmd_selection_route,
        "route-inventory": cmd_route_inventory,
        "route-audit": cmd_route_audit,
        "command": cmd_command,
        "da-config": cmd_da_config,
        "da-command": cmd_da_command,
        "collect": cmd_collect,
        "officehome-config": cmd_officehome_config,
        "plan": cmd_plan,
        "plan-audit": cmd_plan_audit,
        "docs-audit": cmd_docs_audit,
        "repro-smoke": cmd_repro_smoke,
        "matrix": cmd_matrix,
        "pathway-audit": cmd_pathway_audit,
        "smoke": cmd_smoke,
        "check": cmd_check,
    }[args.cmd](args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
