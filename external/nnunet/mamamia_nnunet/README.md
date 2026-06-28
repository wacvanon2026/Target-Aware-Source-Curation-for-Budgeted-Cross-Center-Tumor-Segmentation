# MAMAMIA nnUNet LODO Table Subset

This folder contains a reproducible nnUNet setup for the MAMAMIA leave-one-domain-out experiments.

The baseline segmentation workflow is the shared core:

- `core.py`: target domains, split ratios, budgets, path defaults, and experiment IDs
- `build_splits.py`: deterministic LODO train/val/test and random source splits
- `build_datasets.py`: symlinked nnUNet raw datasets
- `run_one.sh` and `submit_table2_subset.sh`: nnUNet train/predict/evaluate runner
- `collect_results.py`: table collection from `summary.json`

This repo extends that core with source-selection materialization:

- `derive_tumorseg2025_selections.py`: Tumor Seg 2025-style base method rankings
- `materialize_method_selections.py`: converts rankings into current MAMAMIA LODO split files
- `artifacts/tumorseg2025_selections`: committed method-selection artifacts used by the split materializer

No raw imaging data belongs in this Git repo. By default, paths are inferred
from the repo location:

```text
<repo>/../data_selection
```

For the current checkout at `.`, that
resolves to `.`. Override with
`PROJECT_ROOT=/some/other/data_selection` when copying the repo elsewhere.

That means a copied checkout can keep the same layout without editing scripts:

```text
<new-parent>/
  mamamia_new/
  data_selection/
```

If the data lives somewhere else, set `PROJECT_ROOT` at submit/runtime.

## Split Protocol

`build_splits.py` creates deterministic target-domain splits under:

```text
splits/mamamia_lodo_seed42
```

For each target domain:

- target train/val/test uses the paper-style `2:1:7` ratio
- source pool is all non-target domains
- random source selections are nested `K50`, `K150`, and `K250` samples from the source pool

The rounding rule is largest remainder with train/val/test tie order. On the local data this gives:

| target | local target N | train | val | test | source pool |
|---|---:|---:|---:|---:|---:|
| NACT | 64 | 13 | 6 | 45 | 1442 |
| ISPY1 | 171 | 34 | 17 | 120 | 1335 |
| DUKE | 291 | 58 | 29 | 204 | 1215 |
| ISPY2 | 980 | 196 | 98 | 686 | 526 |

## nnUNet IDs

| target | target-only | source-only | target+source | random50 | random150 | random250 |
|---|---:|---:|---:|---:|---:|---:|
| NACT | 1301 | 1302 | 1303 | 1304 | 1305 | 1306 |
| ISPY1 | 1311 | 1312 | 1313 | 1314 | 1315 | 1316 |
| DUKE | 1321 | 1322 | 1323 | 1324 | 1325 | 1326 |
| ISPY2 | 1331 | 1332 | 1333 | 1334 | 1335 | 1336 |

Method-selection datasets use the same builder and runner, but live in separate
ID ranges so they do not collide with the baseline rows:

| target | basic K50 IDs | basic K150 IDs | TAVO K50 | TAVO K150 |
|---|---:|---:|---:|---:|
| NACT | 1401-1408 | 1409-1416 | 1417 | 1418 |
| ISPY1 | 1501-1508 | 1509-1516 | 1517 | 1518 |
| DUKE | 1601-1608 | 1609-1616 | 1617 | 1618 |
| ISPY2 | 1701-1708 | 1709-1716 | 1717 | 1718 |

Within each budget, offsets follow:
`RDS`, `GradMatch`, `LESS`, `ORIENT`, `Diversity`, `KMeans`, `CRAIG`, `KCenter`.
TAVO uses separate offsets after the basic rows so running or completed basic
method datasets are never renumbered.
Basic method rows first materialize from derived Tumor Seg 2025-style selection
artifacts under `artifacts/tumorseg2025_selections`. The derivation script
recreates the original method rules from the local MAMAMIA embeddings and
gradients, then the materializer applies the current LODO source-pool filter.
ORIENT uses `submodlib`'s `FacilityLocationMutualInformationFunction` with
LazyGreedy, matching the Tumor Seg 2025 implementation path. GradMatch and
CRAIG also follow the Tumor Seg 2025 code path by operating on case embeddings;
CRAIG uses `submodlib`'s `FacilityLocationFunction` with LazyGreedy. LESS is the
gradient-based method. For ISPY2, local gradient arrays are not available, so
LESS falls back to the existing proxy selection artifacts.
TAVO rows materialize from the target-specific meta/CMA-ES selection artifacts
where available and are intentionally submitted separately from `basic_methods`.

To refresh the Tumor Seg 2025-style artifacts:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate data_selection_3_10
python scripts/mamamia_nnunet/derive_tumorseg2025_selections.py
python scripts/mamamia_nnunet/materialize_method_selections.py
```

## Environment Setup

The Slurm runner activates a Conda environment before calling nnUNet. By
default it uses your existing environment name:

```bash
CONDA_ENV=data_selection_3_10
```

That is convenient locally, but collaborators should not rely on being able to
read or activate another user's Conda environment. They should either create
their own environment with the same packages, or set `CONDA_ENV` to a shared
environment maintained by the cluster/project.

To create an independent environment against the repo-adjacent
`data_selection` tree:

```bash
CONDA_ENV=mamamia_nnunet scripts/mamamia_nnunet/setup_env.sh
```

If the new environment does not already have GPU PyTorch, install the
cluster-appropriate PyTorch build by setting `INSTALL_TORCH=1` and overriding
`TORCH_INSTALL_CMD` if needed:

```bash
CONDA_ENV=mamamia_nnunet \
INSTALL_TORCH=1 \
TORCH_INSTALL_CMD='python -m pip install torch' \
scripts/mamamia_nnunet/setup_env.sh
```

Then submit jobs with the same environment name:

```bash
CONDA_ENV=mamamia_nnunet ./scripts/mamamia_nnunet/submit_table2_subset.sh --submit
```

The setup helper installs the Python packages only. The raw MAMAMIA data and
the local `externals/MAMA-MIA/nnUNet` checkout must still exist under
`PROJECT_ROOT`, which defaults to `<repo>/../data_selection`.

## Prepare And Validate

```bash
python scripts/mamamia_nnunet/build_splits.py --targets all
python scripts/mamamia_nnunet/materialize_method_selections.py
python scripts/mamamia_nnunet/build_datasets.py all --targets all --skip-existing
./scripts/mamamia_nnunet/submit_table2_subset.sh
```

The wrapper above does not submit training jobs unless `--submit` is passed.

## Submit Later

All targets and all six rows:

```bash
./scripts/mamamia_nnunet/submit_table2_subset.sh --submit
```

Low-storage mode submits the same rows as a dependency chain, one job at a time,
and removes each row's preprocessed dataset after success or runner error while
preserving raw data, logs, outputs, and checkpoints:

```bash
./scripts/mamamia_nnunet/submit_table2_subset.sh --skip-build --chain --submit
```

By default this submits 12-hour jobs on account `YOUR_SLURM_ACCOUNT` and GPUs matching
`a100|a40|l40s|v100`. The runner installs
and uses the repo-owned `nnUNetTrainerTAVOSaveEveryEpoch`, which saves:

- `checkpoint_epoch_###.pth`
- `checkpoint_latest.pth`
- `checkpoint_best.pth`
- `checkpoint_best_last.pth`, copied from the highest `mean_fg_dice`
  `checkpoint_epoch_###.pth` among the last 10 completed epochs before
  prediction
- `checkpoint_final.pth` when training finishes
- `lr_scheduler_state` inside every checkpoint, along with model, optimizer,
  grad-scaler, epoch, logger, and best-EMA state

If a job times out after at least one epoch, resubmitting the same target/row resumes
from `checkpoint_latest.pth` using nnUNet's `--c` flag. If `checkpoint_final.pth`
already exists, training is skipped and prediction/evaluation runs. Reported
test scores are generated with `checkpoint_best_last.pth` by passing
`-chk checkpoint_best_last.pth` to `nnUNetv2_predict`; set
`BEST_LAST_WINDOW=N` to change the last-epoch window.

To re-evaluate rows that were already predicted with nnUNet's default final
checkpoint, submit prediction/evaluation-only jobs. The helper only targets rows
with an existing `summary.json` and skips rows that already have an active Slurm
job:

```bash
python scripts/mamamia_nnunet/submit_best_last_reeval.py --submit
```

Override submission time or GPU features with:

```bash
SBATCH_TIME=18:00:00 ./scripts/mamamia_nnunet/submit_table2_subset.sh --submit
SBATCH_ACCOUNT=other_account ./scripts/mamamia_nnunet/submit_table2_subset.sh --submit
SBATCH_CONSTRAINT='a100|a40|l40s|v100|p100' ./scripts/mamamia_nnunet/submit_table2_subset.sh --submit
PROJECT_ROOT=/path/to/data_selection ./scripts/mamamia_nnunet/submit_table2_subset.sh --submit
```

Only prioritized random rows:

```bash
./scripts/mamamia_nnunet/submit_table2_subset.sh --submit random50 random150
```

All K50/K150 basic method rows:

```bash
./scripts/mamamia_nnunet/submit_table2_subset.sh basic_methods --submit --skip-build
```

TAVO K50/K150 rows, after the basic rows are complete:

```bash
./scripts/mamamia_nnunet/submit_table2_subset.sh tavo --submit --skip-build
```

One target only:

```bash
./scripts/mamamia_nnunet/submit_table2_subset.sh --targets NACT --submit random50 random150
```

## Domain Alignment Baselines

The same nnUNet runner also supports domain-alignment baselines on top of the
shared MAMAMIA dataset core. These rows should usually be run on
`target_full_source`, because DANN/MMD/ADVENT/SE-ASA need source and target
cases in the training split to form alignment batches.

The adaptation trainers are:

| method | trainer | output suffix | alignment signal |
|---|---|---|---|
| DANN | `nnUNetTrainerTAVODANN` | `_dann` | gradient-reversal domain classifier on nnUNet probability maps |
| MMD / DAN | `nnUNetTrainerTAVOMMD` | `_mmd` | RBF MMD over pooled class-probability distributions |
| ADVENT | `nnUNetTrainerTAVOADVENT` | `_advent` | adversarial alignment on prediction entropy maps |
| SE-ASA | `nnUNetTrainerTAVOSEASA` | `_seasa` | target entropy minimization plus semantic distribution alignment |

These implementations follow the Tumor Seg 2025 adaptation patterns but attach
to nnUNet logits/probabilities, which keeps the interface architecture
independent and lets the same runner handle all MAMAMIA nnUNet rows.

Dry-run all four methods for one target:

```bash
python scripts/mamamia_nnunet/submit_domain_alignment.py \
  --targets NACT \
  --experiments target_full_source
```

Submit all four methods for all targets:

```bash
python scripts/mamamia_nnunet/submit_domain_alignment.py \
  --targets all \
  --experiments target_full_source \
  --submit
```

Short debug-partition smoke test for one job:

```bash
CONDA_ENV=mamamia_nnunet \
SKIP_PLAN_PREPROCESS=1 \
TRAIN_DEVICE=cpu \
STOP_AFTER_TRAIN=1 \
NNUNET_NUM_EPOCHS=1 \
NNUNET_ITERATIONS_PER_EPOCH=2 \
NNUNET_VAL_ITERATIONS_PER_EPOCH=1 \
NNUNET_BATCH_SIZE=2 \
python scripts/mamamia_nnunet/submit_domain_alignment.py \
  --methods dann \
  --targets NACT \
  --experiments gradmatch50 \
  --partition debug \
  --time 01:00:00 \
  --constraint 'a40|p100' \
  --max-submit 1 \
  --submit
```

If the raw nnUNet datasets have not been built yet, add `--auto-build`, or build
them first with:

```bash
python scripts/mamamia_nnunet/build_datasets.py target_full_source --targets all --skip-existing
```

Hyperparameters can be set through the submitter or environment:

```bash
python scripts/mamamia_nnunet/submit_domain_alignment.py \
  --methods dann advent \
  --lambda 0.05 \
  --domain-lambda 0.1 \
  --disc-hidden 128 \
  --targets NACT \
  --experiments target_full_source \
  --submit
```

Outputs are written beside the regular nnUNet rows with the method suffix, for
example:

```text
outputs/tavo_mamamia_nact_nnunet_target_full_source_dann/repeat_01/test_preds/summary.json
```

The training log prints `*_seg_loss`, `*_align_loss`, and
`aligned_batch_fraction` at each epoch. If `aligned_batch_fraction` is zero, the
chosen experiment row did not present both source and target cases in training
batches, so the adaptation term was inactive.

## Collect Results

As jobs finish, collect available foreground-mean Dice values from each test
`summary.json`:

```bash
python scripts/mamamia_nnunet/collect_results.py \
  --csv reports/mamamia_nnunet_lodo_results.csv \
  --markdown reports/mamamia_nnunet_lodo_results.md
```

For live Slurm training metrics, summarize the latest epoch logged by each
active MAMAMIA nnUNet job:

```bash
python scripts/mamamia_nnunet/live_metrics.py
```
