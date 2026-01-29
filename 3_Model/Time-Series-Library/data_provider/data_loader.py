# data_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from data_provider.sepsis_csv_loader import SepsisCSVConfig, SepsisCSVWindowDataset



@dataclass(frozen=True)
class CVConfig:
    train_fit_csv: str
    train_thresh_csv: str
    test_csv: Optional[str] = None

    group_col: str = "Patient_ID"
    label_col: str = "SepsisLabel"

    n_splits: int = 5
    fold: int = 0
    seed: int = 42


@dataclass(frozen=True)
class LoaderConfig:
    seq_len: int = 48
    sample_step: int = 1

    batch_size: int = 64
    num_workers: int = 4
    drop_last: bool = True

    # If you want to lock features, pass the list; otherwise infer like notebook
    feature_cols: Optional[List[str]] = None

    # Sorting/assertion behavior matches notebook
    time_cols_priority: Tuple[str, ...] = ("ICULOS", "Hour")
    assert_monotonic_time: bool = True

    pin_memory: bool = True
    persistent_workers: bool = True


def _patient_level_strat_labels(train_fit_csv: str, group_col: str, label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exactly notebook logic:
      patient_y = df.groupby(Patient_ID)[SepsisLabel].max().astype(int)
    """
    df = pd.read_csv(train_fit_csv, usecols=[group_col, label_col])
    patient_y = df.groupby(group_col)[label_col].max().astype(int)
    patient_ids = patient_y.index.to_numpy()
    patient_labels = patient_y.values.astype(int)
    return patient_ids, patient_labels


def make_fold_patient_ids(cv: CVConfig) -> Tuple[List, List]:
    patient_ids, patient_labels = _patient_level_strat_labels(cv.train_fit_csv, cv.group_col, cv.label_col)

    if cv.n_splits < 2:
        raise ValueError(f"n_splits must be >=2, got {cv.n_splits}")
    if not (0 <= cv.fold < cv.n_splits):
        raise ValueError(f"fold must be in [0, {cv.n_splits-1}], got {cv.fold}")

    skf = StratifiedKFold(n_splits=cv.n_splits, shuffle=True, random_state=cv.seed)
    splits = list(skf.split(patient_ids, patient_labels))
    tr_idx, va_idx = splits[cv.fold]

    tr_pids = patient_ids[tr_idx].tolist()
    va_pids = patient_ids[va_idx].tolist()

    if set(tr_pids) & set(va_pids):
        raise RuntimeError("Patient leakage detected: train and val patient sets overlap.")

    return tr_pids, va_pids


def build_loaders(cv: CVConfig, lc: LoaderConfig):
    train_pids, val_pids = make_fold_patient_ids(cv)

    base_cfg = dict(
        group_col=cv.group_col,
        label_col=cv.label_col,
        time_cols_priority=lc.time_cols_priority,
        seq_len=lc.seq_len,
        sample_step=lc.sample_step,
        feature_cols=lc.feature_cols,
        assert_monotonic_time=lc.assert_monotonic_time,
        keep_in_memory=True,
    )

    train_ds = SepsisCSVWindowDataset(SepsisCSVConfig(csv_path=cv.train_fit_csv, **base_cfg), patient_ids=train_pids)
    val_ds   = SepsisCSVWindowDataset(SepsisCSVConfig(csv_path=cv.train_fit_csv, **base_cfg), patient_ids=val_pids)

    thresh_ds = SepsisCSVWindowDataset(SepsisCSVConfig(csv_path=cv.train_thresh_csv, **base_cfg), patient_ids=None)

    train_loader = DataLoader(
        train_ds,
        batch_size=lc.batch_size,
        shuffle=True,
        num_workers=lc.num_workers,
        drop_last=lc.drop_last,
        pin_memory=lc.pin_memory,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=lc.batch_size,
        shuffle=False,
        num_workers=lc.num_workers,
        drop_last=False,
        pin_memory=lc.pin_memory,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
    )
    thresh_loader = DataLoader(
        thresh_ds,
        batch_size=lc.batch_size,
        shuffle=False,
        num_workers=lc.num_workers,
        drop_last=False,
        pin_memory=lc.pin_memory,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
    )

    test_loader = None
    if cv.test_csv:
        test_ds = SepsisCSVWindowDataset(SepsisCSVConfig(csv_path=cv.test_csv, **base_cfg), patient_ids=None)
        test_loader = DataLoader(
            test_ds,
            batch_size=lc.batch_size,
            shuffle=False,
            num_workers=lc.num_workers,
            drop_last=False,
            pin_memory=lc.pin_memory,
            persistent_workers=lc.persistent_workers and lc.num_workers > 0,
        )

    return train_loader, val_loader, thresh_loader, test_loader
