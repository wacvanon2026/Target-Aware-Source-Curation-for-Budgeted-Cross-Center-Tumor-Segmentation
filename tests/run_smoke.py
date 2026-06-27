from pathlib import Path
from tempfile import TemporaryDirectory

from tavo_release.common import release_audit
from tavo_release.cli import main
from tavo_release.docs import readme_audit
from tavo_release.domain_adaptation import build_config, build_train_command
from tavo_release.pathways import audit_pathways
from tavo_release.pipeline import audit_plan, combined_plan
from tavo_release.selection_routes import route_audit, route_inventory, selection_route
from tavo_release.tavo_routes import search_command


with TemporaryDirectory(prefix="tavo_release_test_") as tmp:
    code = main(["smoke", "--workdir", str(Path(tmp))])
    repro_code = main(["repro-smoke", "--workdir", str(Path(tmp) / "repro")])
    if repro_code != 0:
        raise SystemExit(repro_code)
    audit = audit_pathways("configs/pathways.json")
    if not audit["ok"]:
        raise SystemExit(audit["errors"])
    moved_audit = audit_pathways(Path("configs") / "pathways.json")
    if not moved_audit["ok"]:
        raise SystemExit(moved_audit["errors"])
    release = release_audit(".")
    if any(release.values()):
        raise SystemExit(release)
    for dataset, target in (("mamamia", "NACT"), ("brats", "C5"), ("officehome", "Art")):
        if search_command(dataset, target, 50).count("--score") != 8:
            raise SystemExit(f"{dataset} TAVO route is not 8D")
    coreset = selection_route("officehome", "Art", "coreset", 50)
    if coreset["route_type"] not in {"entrypoint", "config_pattern"}:
        raise SystemExit("OfficeHome coreset route is missing")
    inventory = route_inventory("all")
    if len(inventory) != 423:
        raise SystemExit("selection route inventory count changed")
    routed = route_audit()
    if not routed["ok"]:
        raise SystemExit(routed)
    expected_counts = {"selection": 423, "tavo": 39, "domain_adaptation": 168}
    for family, expected in expected_counts.items():
        if routed["families"][family]["count"] != expected:
            raise SystemExit(routed)
    da_routes = route_inventory("mamamia", family="domain_adaptation")
    if not all("--nnunet-dataset-id" in route["config_command"] for route in da_routes):
        raise SystemExit("MAMA-MIA DA routes lost nnUNet dataset ids")
    brats_da = route_inventory("brats", family="domain_adaptation")
    if not all("--target" in route["config_command"] for route in brats_da):
        raise SystemExit("BraTS DA routes lost targets")
    names = {step["name"] for step in combined_plan()}
    for name in ("mamamia_NACT_rds50_selection", "mamamia_NACT_tavo50_search", "mamamia_NACT_tavo50"):
        if name not in names:
            raise SystemExit(f"missing plan step: {name}")
    planned = audit_plan()
    if not planned["ok"]:
        raise SystemExit(planned)
    documented = readme_audit("README.md")
    if not documented["ok"]:
        raise SystemExit(documented)
    cfg = build_config("mamamia", "dann", Path(tmp), Path(tmp) / "out", 50, Path(tmp) / "da.json", nnunet_dataset_id=9000)
    if "9000" not in build_train_command(cfg):
        raise SystemExit("MAMA-MIA DA command lost nnUNet dataset id")
    raise SystemExit(code)
