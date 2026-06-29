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

The released matrix covers MAMA-MIA and BraTS segmentation budgets `50`, `150`, `250`, and `500`. OfficeHome uses class-balanced per-class budgets `1`, `3`, `5`, `8`, `15`, `25`, and `40`. MAMA-MIA targets are `NACT`, `ISPY1`, `DUKE`, and `ISPY2`; BraTS targets are `C4`, `C5`, `TCGA_LGG`, and `TCGA_GBM`; OfficeHome targets are `Art`, `Clipart`, `Product`, and `RealWorld`.

Write a combined execution plan:

```bash
PYTHONPATH=. python -m tavo_release.cli plan --dataset all --output-dir outputs/plans
```

Audit released pathway coverage:

```bash
PYTHONPATH=. python -m tavo_release.cli pathway-audit --pathways configs/pathways.json
PYTHONPATH=. python -m tavo_release.cli route-audit --pathways configs/pathways.json
```

The pathway registry is `configs/pathways.json`. It records the three experiment tracks, the eight 8D TAVO source valuation criteria, the dataset-specific TAVO entrypoints, and the domain-adaptation entrypoints. The `external/efficientvit` and `external/nnunet` directories contain the released training integrations used by those pathways; they are source-code folders, not data or checkpoint folders.

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
PYTHONPATH=. python -m tavo_release.cli da-config --dataset mamamia --method dann --split-dir splits/mamamia_lodo_seed42/NACT --output-dir outputs/mamamia/NACT/dann50 --budget 50 --output configs/generated/mamamia_NACT_dann_50.json --target NACT --nnunet-dataset-id 9000
PYTHONPATH=. python -m tavo_release.cli da-command --config configs/generated/mamamia_NACT_dann_50.json
```

The generated MAMA-MIA DA command launches `python -m tavo_release.mamamia_nnunet_train`, which installs the released TAVO nnU-Net trainer files into the active Python environment before calling `nnUNetv2_train`. Use a writable environment or copy the files in `external/nnunet/mamamia_nnunet` into the installed `nnunetv2/training/nnUNetTrainer` package manually.

Collect result summaries:

```bash
PYTHONPATH=. python -m tavo_release.cli collect --dataset mamamia --results-root outputs/nnUNet_results --output outputs/mamamia_results.json
```

Write a full MAMA-MIA plan:

```bash
PYTHONPATH=. python -m tavo_release.cli plan --dataset mamamia --output-dir outputs/plans
```

MAMA-MIA includes random, RDS, LESS, ORIENT, CRAIG, GradMatch, KMeans, KCenter, Diversity, TAVO-8D CMA-ES, and the DANN, MMD, ADVENT, and SE-ASA nnU-Net trainer integrations.

## BraTS

Build target-domain splits:

```bash
PYTHONPATH=. python -m tavo_release.cli split --dataset brats --data-root data/brats --output-root splits/brats --target C5
```

For BraTS, `data/brats` must either contain domain-coded case directories such as `case_C5_001` or explicit split lists such as `C5_target_train.txt`, `C5_target_val.txt`, `C5_target_test.txt`, and `C5_source_pool.txt`.

Create method selections from score dictionaries:

```bash
PYTHONPATH=. python -m tavo_release.cli select --score rds=scores/brats/C5/rds.json --score less=scores/brats/C5/less.json --score orient=scores/brats/C5/orient.json --score craig=scores/brats/C5/craig.json --score gradmatch=scores/brats/C5/gradmatch.json --score kmeans=scores/brats/C5/kmeans.json --score kcenter=scores/brats/C5/kcenter.json --score diversity=scores/brats/C5/diversity.json --weight 1 0 0 0 0 0 0 0 --budget 50 --output splits/brats/C5/methods/rds_50.txt
```

Run CMA-ES score fusion:

```bash
PYTHONPATH=. python -m tavo_release.cli search --score rds=scores/brats/C5/rds.json --score less=scores/brats/C5/less.json --score orient=scores/brats/C5/orient.json --score craig=scores/brats/C5/craig.json --score gradmatch=scores/brats/C5/gradmatch.json --score kmeans=scores/brats/C5/kmeans.json --score kcenter=scores/brats/C5/kcenter.json --score diversity=scores/brats/C5/diversity.json --budget 50 --output-dir outputs/brats/C5/tavo50
```

BraTS selection, TAVO, and DA scripts are integrated through the EfficientViT pathway entries in `configs/pathways.json`. EfficientViT pretrained weights are not stored in git; if the expected checkpoint is absent, the released segmentation model initializes without pretrained weights.

The restored EfficientViT search implementation includes the C4 repeated-target CMA scripts under `external/efficientvit/scripts/search_C4_multi`, the C5/TCGA CMA scripts under `external/efficientvit/scripts/search_multi`, and BraTS ablation utilities under `external/efficientvit/scripts/revise_ablation`. BraTS DA is implemented by `external/efficientvit/scripts/train_seg_da.py` with DANN, DAN/MMD, ADVENT, and SE-ASA modes.

```bash
PYTHONPATH=. python -m tavo_release.cli da-config --dataset brats --method mmd --split-dir splits/brats/C5 --output-dir outputs/brats/C5/mmd50 --budget 50 --output configs/generated/brats_C5_mmd_50.json --target C5
PYTHONPATH=. python -m tavo_release.cli da-command --config configs/generated/brats_C5_mmd_50.json
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

OfficeHome source selection uses its classification-specific 8D criteria: KMeans, KCenter, FacilityLocation, CRAIG, TargetMMD, TargetGradMatch, GLISTER, and ORIENT. The revised source-selection and TAVO scripts live in `external/efficientvit/scripts_cls_revise`; DA uses DANN, MMD, CORAL, and CDAN through `external/efficientvit/scripts_cls/train_cls_da.py`.

```bash
PYTHONPATH=. python -m tavo_release.cli da-config --dataset officehome --method coral --split-dir splits/officehome/Art --output-dir outputs/officehome/Art/coral15 --budget 15 --output configs/generated/officehome_Art_coral_15.json --target Art
PYTHONPATH=. python -m tavo_release.cli da-command --config configs/generated/officehome_Art_coral_15.json
```

## Release Checks

Run:

```bash
PYTHONPATH=. python -m tavo_release.cli repro-smoke
PYTHONPATH=. python -m tavo_release.cli check --root .
```

The release folder ignores datasets, checkpoints, model weights, and generated outputs.
