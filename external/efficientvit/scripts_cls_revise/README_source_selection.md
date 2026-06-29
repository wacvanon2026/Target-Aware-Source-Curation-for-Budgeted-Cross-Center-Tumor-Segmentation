# OfficeHome Source Selection Baselines

This folder contains the revised classification source-selection pipeline.
The task definition matches the TAVO revision setting: a small labeled target
train/validation split is fixed, and each method selects a class-balanced source
subset from the remaining source domains.

## Splits

Generate leave-one-domain-out OfficeHome splits:

```bash
python scripts_cls_revise/generate_officehome_splits.py \
  --data-root data_cls/office_home \
  --output-root data_cls_revise/splits/officehome \
  --target-shots 1 3 5 \
  --val-shots 2 \
  --seed 0
```

This writes, for each target domain:

- `source_train.txt`: all images from the other three domains.
- `target_train_1shot.txt`, `target_train_3shot.txt`,
  `target_train_5shot.txt`: nested labeled target train splits.
- `target_val_2shot.txt`: 2 labeled target images per class.
- `target_test.txt`: held-out target images.

When multiple target-shot values are requested, the train splits are generated
from a single per-class random order. For example, `target_train_3shot.txt` is a
prefix of `target_train_5shot.txt`. Validation and test images are placed after
the largest requested train split, so they never overlap with any train-shot
setting.

## Selectors

Run source selection for one target and budget:

```bash
python scripts_cls_revise/select_source_officehome.py \
  --source-list data_cls_revise/splits/officehome/Art/seed00/source_train.txt \
  --target-train-list data_cls_revise/splits/officehome/Art/seed00/target_train_3shot.txt \
  --target-val-list data_cls_revise/splits/officehome/Art/seed00/target_val_2shot.txt \
  --output-dir data_cls_revise/source_subsets/officehome/Art/seed00/B3 \
  --cache-dir data_cls_revise/source_subsets/officehome/Art/seed00/cache \
  --methods all \
  --budget-per-class 3 \
  --warmup-ckpt experiments_cls_revise/officehome/Art/warmup_full/split00/train_seed00/best.pt \
  --seed 0 \
  --use-cache
```

Methods:

- `KMeans-B`: per-class KMeans medoids over frozen ResNet features.
- `KCenter-B`: per-class farthest-first coreset over frozen ResNet features.
- `FacilityLocation-B`: per-class facility-location greedy selection over cosine
  feature similarity.
- `CRAIG-B`: per-class facility-location greedy selection over last-layer
  gradient embeddings, following the CRAIG gradient-facility-location objective.
- `TargetMMD-B`: class-conditional moment-matching source selection. It greedily
  selects source features whose mean approaches the labeled target feature mean.
- `TargetGradMatch-B`: class-conditional gradient matching against labeled target
  train gradients.
- `GLISTER-B`: validation-aware gradient matching against target validation
  gradients, approximating the one-step validation-loss objective used by GLISTER.
- `ORIENT-B`: class-conditional source-target submodular mutual information. It
  uses `submodlib` FLMI when available and falls back to a target-coverage greedy
  variant otherwise.

Gradient-based selectors require a warmup checkpoint by default. This keeps the
classification setup faithful to CRAIG/GradMatch/GLISTER, where selection is
defined using model gradients. For smoke tests only, pass
`--allow-random-head-gradients`.

## Training Configs

Generate training configs for the 8 source-selection methods:

```bash
python scripts_cls_revise/generate_officehome_selection_configs.py \
  --split-root data_cls_revise/splits/officehome \
  --subset-root data_cls_revise/source_subsets/officehome \
  --config-root configs_cls_revise/officehome \
  --output-root experiments_cls_revise/officehome \
  --split-seed 0 \
  --target-shots 3 \
  --val-shots 2 \
  --budgets 1 3 5 \
  --train-seeds 0 1 2 \
  --epochs 30
```

The generated configs use `target_val_2shot.txt` as the validation set and
`target_test.txt` only for final evaluation.
