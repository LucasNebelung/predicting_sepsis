# data_provider/data_factory.py
"""
Minimal data_factory for your Sepsis CSV pipeline ONLY.

Supports task_name == 'classification' and args.data == 'SepsisCSV'.

Key guarantees:
- Train/Val split happens ONLY within train_fit CSV
- Patient-level stratification: max(SepsisLabel) per Patient_ID
- No leakage across folds (patients never overlap)
- No "future peeking": dataset is responsible for causal windows; we sort by [Patient_ID, ICULOS, Hour] if present

Expected args (most match your existing run.py):
  --task_name classification
  --data SepsisCSV
  --root_path <base folder>
  --train_dir <relative or absolute path to train_fit.csv>
  --thresh_dir <relative or absolute path to train_thresh.csv>
  --test_dir <relative or absolute path to test.csv>

Optional split args:
  --k_fold <int>  (>=2 enables StratifiedKFold)
  --fold <int>    (0..k_fold-1)
  --val_ratio <float> (used only if k_fold not set)

Optional dataset args (passed through):
  --seq_len, --sample_step
  --group_col (default Patient_ID)
  --label_col (default SepsisLabel)
  --assert_monotonic_time (default 1)
  --feature_cols (if you inject it programmatically; CLI typically won't pass lists)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from torch.utils.data import DataLoader


def _sepsis_patient_labels_from_csv(train_fit_csv: str, group_col: str, label_col: str):
    """
    Mirrors LIGHTGBMv2.ipynb split logic:
      y_patient = df.groupby(Patient_ID)[SepsisLabel].max().astype(int)
    Returns:
      patient_ids: np.ndarray
      patient_labels: np.ndarray (0/1)
    """
    df = pd.read_csv(train_fit_csv, usecols=[group_col, label_col])
    y_pat = df.groupby(group_col)[label_col].max().astype(int)
    patient_ids = y_pat.index.to_numpy()
    patient_labels = y_pat.values.astype(np.int64)
    return patient_ids, patient_labels


def _resolve_path(root_path: str, maybe_rel: str) -> str:
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(root_path, maybe_rel)


def data_provider(args, flag):
    """
    Entry point used by the library. Minimal: SepsisCSV classification only.
    """
    flag_u = str(flag).upper()

    if getattr(args, "task_name", None) != "classification":
        raise ValueError(
            f"This minimal data_factory only supports task_name='classification'. Got: {args.task_name}"
        )

    if getattr(args, "data", None) != "SepsisCSV":
        raise ValueError(
            f"This minimal data_factory only supports data='SepsisCSV'. Got: {args.data}"
        )

    # Lazy import so module path resolution is clean
    from data_provider.sepsis_csv_loader import SepsisCSVWindowDataset, SepsisCSVConfig

    # Paths (you pass these via --train_dir/--thresh_dir/--test_dir)
    train_csv = getattr(args, "train_dir", None)
    test_csv = getattr(args, "test_dir", None)
    thresh_csv = getattr(args, "thresh_dir", None)

    if not train_csv:
        raise ValueError("--train_dir must point to train_fit CSV")
    if not test_csv:
        raise ValueError("--test_dir must point to test CSV")
    if not thresh_csv:
        raise ValueError("--thresh_dir must point to train_thresh CSV")

    train_path = _resolve_path(args.root_path, train_csv)
    test_path = _resolve_path(args.root_path, test_csv)
    thresh_path = _resolve_path(args.root_path, thresh_csv)

    if not os.path.exists(train_path):
        raise ValueError(f"TRAIN_FIT csv not found: {train_path}")
    if not os.path.exists(test_path):
        raise ValueError(f"TEST csv not found: {test_path}")
    if not os.path.exists(thresh_path):
        raise ValueError(f"TRAIN_THRESH csv not found: {thresh_path}")

    # Column names (defaults match your notebook)
    group_col = str(getattr(args, "group_col", "Patient_ID"))
    label_col = str(getattr(args, "label_col", "SepsisLabel"))

    # Build stratification labels using ONLY train_fit
    patient_ids, y_pat = _sepsis_patient_labels_from_csv(train_path, group_col, label_col)

    # Split parameters
    seed = int(getattr(args, "seed", 2021))
    val_ratio = float(getattr(args, "val_ratio", 0.2))
    k_fold = int(getattr(args, "k_fold", 0) or 0)
    fold = int(getattr(args, "fold", 0) or 0)

    # DataLoader parameters
    batch_size = int(getattr(args, "batch_size", 64))
    num_workers = int(getattr(args, "num_workers", 0))
    drop_last = False

    # Dataset config (NO preprocessing here)
    base_cfg = dict(
        group_col=group_col,
        label_col=label_col,
        time_cols_priority=("ICULOS", "Hour"),  # notebook-style sorting if present
        seq_len=int(getattr(args, "seq_len", 48)),
        sample_step=int(getattr(args, "sample_step", 1)),
        feature_cols=getattr(args, "feature_cols", None),  # optional programmatic override
        assert_monotonic_time=bool(int(getattr(args, "assert_monotonic_time", 1))),
        keep_in_memory=True,
    )

    # TRAIN / VAL come ONLY from train_fit
    if flag_u in ("TRAIN", "VAL"):
        if k_fold and k_fold > 1:
            if fold < 0 or fold >= k_fold:
                raise ValueError(f"--fold must be in [0, {k_fold-1}], got {fold}")
            skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=seed)
            splits = list(skf.split(np.zeros(len(y_pat)), y_pat))
            tr_idx, va_idx = splits[fold]
        else:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            tr_idx, va_idx = next(sss.split(np.zeros(len(y_pat)), y_pat))

        train_pids = patient_ids[tr_idx].tolist()
        val_pids = patient_ids[va_idx].tolist()

        # hard leakage guard
        if set(train_pids) & set(val_pids):
            raise RuntimeError("Patient leakage detected: train/val patient sets overlap.")

        if flag_u == "TRAIN":
            ds_cfg = SepsisCSVConfig(csv_path=train_path, **base_cfg)
            data_set = SepsisCSVWindowDataset(ds_cfg, patient_ids=train_pids)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=drop_last,
            )
            return data_set, data_loader

        else:
            ds_cfg = SepsisCSVConfig(csv_path=train_path, **base_cfg)
            data_set = SepsisCSVWindowDataset(ds_cfg, patient_ids=val_pids)
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=drop_last,
            )
            return data_set, data_loader

    # THRESH is held out (no split)
    if flag_u in ("THRESH", "TRAIN_THRESH"):
        ds_cfg = SepsisCSVConfig(csv_path=thresh_path, **base_cfg)
        data_set = SepsisCSVWindowDataset(ds_cfg, patient_ids=None)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=drop_last,
        )
        return data_set, data_loader

    # TEST
    ds_cfg = SepsisCSVConfig(csv_path=test_path, **base_cfg)
    data_set = SepsisCSVWindowDataset(ds_cfg, patient_ids=None)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader
