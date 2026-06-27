from pathlib import Path
from tempfile import TemporaryDirectory

from tavo_release.cli import main


with TemporaryDirectory(prefix="tavo_release_test_") as tmp:
    code = main(["smoke", "--workdir", str(Path(tmp))])
    raise SystemExit(code)
