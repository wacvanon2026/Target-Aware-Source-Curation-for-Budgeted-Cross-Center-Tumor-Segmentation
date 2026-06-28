import json
import os
import shlex
import importlib.util
from pathlib import Path

from tavo_release.common import forbidden_text_hits, release_audit, scan_git_messages, scan_tracked_forbidden_text, scan_tracked_release_files, tracked_file_issues
from tavo_release.brats import build_domain_splits
from tavo_release.cli import main
from tavo_release.docs import readme_audit
from tavo_release.domain_adaptation import build_config, build_train_command
from tavo_release.pathways import audit_pathways
from tavo_release.pipeline import audit_plan, combined_plan
from tavo_release.selection_routes import route_audit, route_command_errors, route_inventory, selection_route
from tavo_release.tavo_routes import search_command


def test_smoke(tmp_path: Path):
    assert main(["smoke", "--workdir", str(tmp_path)]) == 0


def test_repro_smoke(tmp_path: Path):
    assert main(["repro-smoke", "--workdir", str(tmp_path)]) == 0


def test_pathway_audit():
    result = audit_pathways("configs/pathways.json")
    assert result["ok"], result["errors"]
    result = audit_pathways(Path("configs") / "pathways.json")
    assert result["ok"], result["errors"]


def test_pathway_entrypoints_exist():
    data = json.loads(Path("configs/pathways.json").read_text())
    missing = []
    for spec in data["pathways"]:
        for field in ("selection_entrypoints", "domain_adaptation_entrypoints"):
            for method, entrypoint in spec.get(field, {}).items():
                if entrypoint.endswith(".py") and not Path(entrypoint).exists():
                    missing.append((spec["dataset"], field, method, entrypoint))
        for entrypoint in spec.get("tavo_entrypoints", []):
            for token in shlex.split(entrypoint):
                if token.endswith(".py") and not Path(token).exists():
                    missing.append((spec["dataset"], "tavo_entrypoints", "tavo", token))
    assert missing == []


def test_mamamia_helper_release_layout():
    core_path = Path("external/nnunet/mamamia_nnunet/core.py")
    spec = importlib.util.spec_from_file_location("mamamia_release_core", core_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.REPO_ROOT == Path.cwd().resolve()
    assert module.dataset_root() == Path("data/mamamia").resolve()
    assert module.nnunet_raw_root() == Path("outputs/nnunet/nnUNet_raw").resolve()
    for script in Path("external/nnunet/mamamia_nnunet").glob("*.sh"):
        assert os.access(script, os.X_OK), script


def test_pathway_audit_rejects_extra_methods(tmp_path: Path):
    data = json.loads(Path("configs/pathways.json").read_text())
    data["pathways"][0]["selection_methods"].append("stale_method")
    path = tmp_path / "pathways.json"
    path.write_text(json.dumps(data))
    result = audit_pathways(path)
    assert not result["ok"]
    assert any("extra selection_methods" in error for error in result["errors"])


def test_brats_split_rejects_missing_target_domain(tmp_path: Path):
    (tmp_path / "BraTS2021_00000").mkdir()
    try:
        build_domain_splits(tmp_path, tmp_path / "out", "C5")
    except ValueError as exc:
        assert "no BraTS cases found for target C5" in str(exc)
    else:
        raise AssertionError("expected BraTS split failure")


def test_brats_split_accepts_explicit_lists(tmp_path: Path):
    (tmp_path / "C5_target_train.txt").write_text("a\n")
    (tmp_path / "C5_target_val.txt").write_text("b\n")
    (tmp_path / "C5_target_test.txt").write_text("c\n")
    (tmp_path / "C5_source_pool.txt").write_text("d\n")
    result = build_domain_splits(tmp_path, tmp_path / "out", "C5")
    assert result == {"target_train": 1, "target_val": 1, "target_test": 1, "source_pool": 1}


def test_tavo_routes_are_8d():
    for dataset, target in (("mamamia", "NACT"), ("brats", "C5"), ("officehome", "Art")):
        cmd = search_command(dataset, target, 50)
        assert cmd.count("--score") == 8
        assert cmd[0:4] == ["python", "-m", "tavo_release.cli", "search"]


def test_selection_route_inventory_covers_release_methods():
    random_route = selection_route("mamamia", "NACT", "random", 50)
    assert random_route["path"] == "splits/mamamia_lodo_seed42/NACT/random/random_50.txt"
    rds_route = selection_route("mamamia", "NACT", "rds", 50)
    assert "splits/mamamia_lodo_seed42/NACT/methods/rds_50.txt" in rds_route["command"]
    route = selection_route("officehome", "Art", "kmeans", 50)
    assert route["route_type"] == "score_file"
    assert len(route_inventory("all")) == 351
    audit = route_audit()
    assert audit["ok"]
    assert audit["families"]["selection"]["count"] == 351
    assert audit["families"]["tavo"]["count"] == 39
    assert audit["families"]["domain_adaptation"]["count"] == 156
    da_routes = route_inventory("mamamia", family="domain_adaptation")
    assert all("--nnunet-dataset-id" in route["config_command"] for route in da_routes)
    da_routes = route_inventory("brats", family="domain_adaptation")
    assert all("--target" in route["config_command"] for route in da_routes)
    assert not any(route_command_errors("domain_adaptation", route) for route in da_routes)


def test_combined_plan_covers_mamamia_selection_and_tavo_search():
    names = {step["name"] for step in combined_plan()}
    assert "mamamia_NACT_rds50_selection" in names
    assert "mamamia_NACT_tavo50_search" in names
    assert "mamamia_NACT_tavo50" in names
    result = audit_plan()
    assert result["ok"], result["errors"]


def test_readme_commands_match_release_surface():
    result = readme_audit("README.md")
    assert result["ok"], result["errors"]


def test_release_audit():
    result = release_audit(".")
    assert not any(result.values()), result


def test_tracked_text_audit_covers_hidden_files():
    assert scan_tracked_forbidden_text(".") == []
    assert scan_git_messages(".") == []
    needle = "/" + "project" + "2" + "/"
    assert forbidden_text_hits(".gitignore", needle + "x") == [(".gitignore", needle)]


def test_tracked_release_audit_rejects_runtime_files(tmp_path: Path):
    assert scan_tracked_release_files(".") == []
    assert "runtime_directory" in tracked_file_issues(".", Path("outputs") / "tracked_probe.txt")


def test_mamamia_da_command_requires_dataset_id(tmp_path: Path):
    cfg = build_config("mamamia", "dann", tmp_path, tmp_path / "out", 50, tmp_path / "da.json", nnunet_dataset_id=9000)
    assert "9000" in build_train_command(cfg)


def test_da_config_preserves_target(tmp_path: Path):
    cfg = build_config("brats", "mmd", tmp_path, tmp_path / "out", 50, tmp_path / "brats_da.json", target="C5")
    command = build_train_command(cfg)
    assert command[:3] == ["env", "PYTHONPATH=external/efficientvit", "python"]
    assert "external/efficientvit/scripts/train_seg_da.py" in command
    assert "--config" in command
    cfg = build_config("officehome", "coral", tmp_path, tmp_path / "out", 50, tmp_path / "office_da.json", target="Art")
    command = build_train_command(cfg)
    assert command[:3] == ["env", "PYTHONPATH=external/efficientvit", "python"]
    assert "external/efficientvit/scripts_cls/train_cls_da.py" in command
    assert "--config" in command
