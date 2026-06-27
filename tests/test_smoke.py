from pathlib import Path

from tavo_release.cli import main
from tavo_release.pathways import audit_pathways
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
