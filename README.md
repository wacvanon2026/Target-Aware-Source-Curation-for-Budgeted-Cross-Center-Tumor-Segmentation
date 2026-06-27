# Target-Aware Source Curation Code Release

This folder consolidates the code surfaces used for the paper experiments across MAMA-MIA, BraTS, and OfficeHome.

The release is organized around one package, `tavo_release`, and one CLI:

```bash
PYTHONPATH=. python -m tavo_release.cli smoke
```

The code uses relative paths by default. Put datasets under `data` or pass explicit `--data-root` values. Generated outputs belong under `outputs`, `splits`, or external paths supplied at runtime. The repository intentionally excludes datasets, checkpoints, preprocessed caches, logs, and trained weights.

List the full released method matrix:

```bash
PYTHONPATH=. python -m tavo_release.cli matrix
```

Write a combined execution plan:

```bash
PYTHONPATH=. python -m tavo_release.cli plan --dataset all --output-dir outputs/plans
```

The pathway registry is `configs/pathways.json`. It records the three experiment tracks, the eight 8D TAVO source valuation criteria, the dataset-specific TAVO entrypoints, and the domain-adaptation entrypoints. The `external/efficientvit` and `external/nnunet` directories are runtime integration points for code trees supplied by the runner; they are not data or checkpoint folders.

Generic archive download:

```bash
PYTHONPATH=. python -m tavo_release.cli download --dataset officehome --url "$OFFICEHOME_ARCHIVE_URL" --output-dir data/downloads --filename officehome.zip
```

## MAMA-MIA

Build leave-one-domain-out splits:

```bash
PYTHONPATH=. python -m tavo_release.cli split --dataset mamamia --data-root data/mamamia --output-root splits/mamamia_lodo_seed42
```

Materialization is exposed through `tavo_release.mamamia.materialize_nnunet_raw`. It symlinks image and label files into nnUNet raw format and writes `dataset.json`.

Generate nnUNet commands:

```bash
PYTHONPATH=. python -m tavo_release.cli command --dataset mamamia --dataset-id 1301
```

Generate a domain-adaptation config and command:

```bash
PYTHONPATH=. python -m tavo_release.cli da-config --dataset mamamia --method dann --split-dir splits/mamamia_lodo_seed42/NACT --output-dir outputs/mamamia/NACT/dann50 --budget 50 --output configs/generated/mamamia_NACT_dann_50.json --nnunet-dataset-id 9000
PYTHONPATH=. python -m tavo_release.cli da-command --config configs/generated/mamamia_NACT_dann_50.json
```

Collect result summaries:

```bash
PYTHONPATH=. python -m tavo_release.cli collect --dataset mamamia --results-root outputs/nnUNet_results --output outputs/mamamia_results.json
```

Write a full MAMA-MIA plan:

```bash
PYTHONPATH=. python -m tavo_release.cli plan --dataset mamamia --output-dir outputs/plans
```

## BraTS

Build target-domain splits:

```bash
PYTHONPATH=. python -m tavo_release.cli split --dataset brats --data-root data/brats --output-root splits/brats --target C5
```

Create method selections from score dictionaries:

```bash
PYTHONPATH=. python -m tavo_release.cli select --score rds=scores/rds.json --score less=scores/less.json --score orient=scores/orient.json --weight 0.4 0.4 0.2 --budget 50 --output splits/brats/C5/methods/tavo_50.txt
```

Run CMA-ES score fusion:

```bash
PYTHONPATH=. python -m tavo_release.cli search --score rds=scores/brats/C5/rds.json --score less=scores/brats/C5/less.json --score orient=scores/brats/C5/orient.json --score craig=scores/brats/C5/craig.json --score gradmatch=scores/brats/C5/gradmatch.json --score kmeans=scores/brats/C5/kmeans.json --score kcenter=scores/brats/C5/kcenter.json --score diversity=scores/brats/C5/diversity.json --budget 50 --output-dir outputs/brats/C5/tavo50
```

BraTS selection and DA scripts are integrated through the EfficientViT pathway entries in `configs/pathways.json`.

```bash
PYTHONPATH=. python -m tavo_release.cli da-config --dataset brats --method aada --split-dir splits/brats/C5 --output-dir outputs/brats/C5/aada50 --budget 50 --output configs/generated/brats_C5_aada_50.json
PYTHONPATH=. python -m tavo_release.cli da-command --config configs/generated/brats_C5_aada_50.json
```

## OfficeHome

Download from a user-provided archive URL:

```python
import os
from tavo_release.officehome import download_officehome, extract_archive
archive = download_officehome(os.environ["OFFICEHOME_ARCHIVE_URL"], "data/downloads")
extract_archive(archive, "data/officehome")
```

Build Art as the target domain:

```bash
PYTHONPATH=. python -m tavo_release.cli split --dataset officehome --data-root data/officehome --output-root splits/officehome --target Art
```

OfficeHome classification uses the classification pathway entries in `configs/pathways.json`.

```bash
PYTHONPATH=. python -m tavo_release.cli da-config --dataset officehome --method mme --split-dir splits/officehome/Art --output-dir outputs/officehome/Art/mme50 --budget 50 --output configs/generated/officehome_Art_mme_50.json
PYTHONPATH=. python -m tavo_release.cli da-command --config configs/generated/officehome_Art_mme_50.json
```

## Release Checks

Run:

```bash
bash scripts/smoke.sh
bash scripts/check_release.sh
PYTHONPATH=. python tests/run_smoke.py
```

The release folder ignores datasets, checkpoints, model weights, and generated outputs.
