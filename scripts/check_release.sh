set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ROOT" python -m tavo_release.cli check --root "$ROOT"
cd "$ROOT"
PYTHONPATH="$ROOT" python -m tavo_release.cli pathway-audit --pathways configs/pathways.json
PYTHONPATH="$ROOT" python -m tavo_release.cli route-audit --pathways configs/pathways.json
