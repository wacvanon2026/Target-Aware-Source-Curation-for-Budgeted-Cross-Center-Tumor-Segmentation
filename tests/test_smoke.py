from pathlib import Path

from tavo_release.cli import main


def test_smoke(tmp_path: Path):
    assert main(["smoke", "--workdir", str(tmp_path)]) == 0
