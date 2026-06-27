from pathlib import Path
from tempfile import TemporaryDirectory

from tavo_release.common import release_audit
from tavo_release.cli import main
from tavo_release.pathways import audit_pathways
from tavo_release.tavo_routes import search_command


with TemporaryDirectory(prefix="tavo_release_test_") as tmp:
    code = main(["smoke", "--workdir", str(Path(tmp))])
    audit = audit_pathways("configs/pathways.json")
    if not audit["ok"]:
        raise SystemExit(audit["errors"])
    release = release_audit(".")
    if any(release.values()):
        raise SystemExit(release)
    for dataset, target in (("mamamia", "NACT"), ("brats", "C5"), ("officehome", "Art")):
        if search_command(dataset, target, 50).count("--score") != 8:
            raise SystemExit(f"{dataset} TAVO route is not 8D")
    raise SystemExit(code)
