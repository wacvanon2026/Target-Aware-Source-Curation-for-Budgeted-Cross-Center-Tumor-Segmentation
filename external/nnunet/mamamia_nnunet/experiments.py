#!/usr/bin/env python3
"""Compatibility wrapper for the shared MAMAMIA nnUNet core."""

from __future__ import annotations

from core import (  # noqa: F401
    ALIASES,
    BUDGETS,
    DOMAINS,
    EXPERIMENTS,
    GROUPS,
    METHOD_BUDGETS,
    METHODS,
    RATIO,
    TARGET_BASES,
    TAVO_BUDGET_OFFSETS,
    Experiment,
    dataset_basename,
    dataset_id,
    dataset_name,
    dataset_root,
    expand_experiments,
    get_experiment,
    nnunet_preprocessed_root,
    nnunet_raw_root,
    nnunet_results_root,
    nnunet_root,
    normalize_experiment,
    normalize_target,
    project_root,
    split_root,
)
