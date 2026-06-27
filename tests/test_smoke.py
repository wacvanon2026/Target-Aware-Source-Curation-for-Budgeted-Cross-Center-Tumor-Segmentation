from pathlib import Path

from tavo_release.common import release_audit
from tavo_release.cli import main
from tavo_release.domain_adaptation import build_config, build_train_command
from tavo_release.pathways import audit_pathways
from tavo_release.selection_routes import route_audit, route_inventory, selection_route
from tavo_release.tavo_routes import search_command


def test_smoke(tmp_path: Path):
    assert main(["smoke", "--workdir", str(tmp_path)]) == 0


def test_pathway_audit():
    result = audit_pathways("configs/pathways.json")
    assert result["ok"], result["errors"]


def test_tavo_routes_are_8d():
    for dataset, target in (("mamamia", "NACT"), ("brats", "C5"), ("officehome", "Art")):
        cmd = search_command(dataset, target, 50)
        assert cmd.count("--score") == 8
        assert cmd[0:4] == ["python", "-m", "tavo_release.cli", "search"]


def test_selection_route_inventory_covers_extra_officehome_methods():
    route = selection_route("officehome", "Art", "coreset", 50)
    assert route["route_type"] in {"entrypoint", "config_pattern"}
    assert len(route_inventory("all")) == 423
    assert route_audit()["ok"]


def test_release_audit():
    result = release_audit(".")
    assert not any(result.values()), result


def test_mamamia_da_command_requires_dataset_id(tmp_path: Path):
    cfg = build_config("mamamia", "dann", tmp_path, tmp_path / "out", 50, tmp_path / "da.json", nnunet_dataset_id=9000)
    assert "9000" in build_train_command(cfg)
