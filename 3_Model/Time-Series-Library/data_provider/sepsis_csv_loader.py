# sepsis_csv_loader.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SepsisCSVConfig:
    csv_path: str
    group_col: str = "Patient_ID"
    label_col: str = "SepsisLabel"

    # Used only for sorting + monotonic assertions (if present in df)
    time_cols_priority: Tuple[str, ...] = ("ICULOS", "Hour")

    seq_len: int = 48
    sample_step: int = 1  # 1 uses every row; >1 subsamples the global row stream
    feature_cols: Optional[List[str]] = None  # if None: infer like notebook

    assert_monotonic_time: bool = True
    keep_in_memory: bool = True


class SepsisCSVWindowDataset(Dataset):
    """
    One sample = one timestep t from one patient, represented as a causal window ending at t.

    Mirrors LIGHTGBMv2.ipynb:
      - drop Unnamed: 0
      - sort by [Patient_ID, ICULOS, Hour] if present
      - X = df.drop([SepsisLabel, Patient_ID])
    """

    def __init__(self, cfg: SepsisCSVConfig, patient_ids: Optional[Sequence] = None):
        super().__init__()
        self.cfg = cfg
        self.patient_id_filter = set(patient_ids) if patient_ids is not None else None

        df = pd.read_csv(cfg.csv_path)

        # Notebook cleanup
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        # Filter to split patients (train/val) if provided
        if self.patient_id_filter is not None:
            if cfg.group_col not in df.columns:
                raise KeyError(f"Missing group_col={cfg.group_col} in CSV.")
            df = df[df[cfg.group_col].isin(self.patient_id_filter)].copy()

        # Sort like notebook: [Patient_ID, ICULOS, Hour] if exist
        sort_cols = [cfg.group_col]
        for c in cfg.time_cols_priority:
            if c in df.columns:
                sort_cols.append(c)
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Feature columns inference like notebook:
        #   X = df.drop([SepsisLabel, Patient_ID])
        if cfg.feature_cols is None:
            drop_cols = [c for c in [cfg.label_col, cfg.group_col] if c in df.columns]
            feature_cols = [c for c in df.columns if c not in drop_cols]
        else:
            feature_cols = list(cfg.feature_cols)

        self.feature_cols = feature_cols

        # Build patient offsets for fast access
        if cfg.group_col not in df.columns:
            raise KeyError(f"Missing group_col={cfg.group_col} in CSV.")

        grp = df.groupby(cfg.group_col, sort=False)
        self.patient_ids = list(grp.indices.keys())
        sizes = grp.size().to_numpy(dtype=np.int64)

        self.patient_lengths = sizes
        self.patient_offsets = np.cumsum(np.concatenate([[0], sizes[:-1]]))

        self.total_raw = int(sizes.sum())
        self.sample_step = max(1, int(cfg.sample_step))
        self.total_len = (self.total_raw + self.sample_step - 1) // self.sample_step

        self._df = df if cfg.keep_in_memory else None

        # Optional time monotonicity checks (after sorting)
        if cfg.assert_monotonic_time:
            self._assert_time_monotonic()

        # Cache one patient at a time
        self._cache_patient_idx: Optional[int] = None
        self._cache_X: Optional[np.ndarray] = None
        self._cache_y: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.total_len

    def _locate(self, idx: int) -> Tuple[int, int]:
        raw_idx = int(idx) * self.sample_step
        if raw_idx >= self.total_raw:
            raw_idx = self.total_raw - 1
        pi = int(np.searchsorted(self.patient_offsets, raw_idx, side="right") - 1)
        t = int(raw_idx - self.patient_offsets[pi])
        return pi, t

    def _load_patient_cached(self, patient_idx: int) -> None:
        if self._cache_patient_idx == patient_idx:
            return
        cfg = self.cfg
        df = self._df
        if df is None:
            raise RuntimeError("keep_in_memory=False not implemented in this minimal loader.")

        start = int(self.patient_offsets[patient_idx])
        L = int(self.patient_lengths[patient_idx])
        end = start + L
        dfp = df.iloc[start:end]

        # Features: numeric coercion, no preprocessing beyond that
        X = dfp[self.feature_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        # Row-level labels (like notebook y_tr = tr[TARGET])
        if cfg.label_col in dfp.columns:
            y = dfp[cfg.label_col].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        else:
            y = np.zeros((len(dfp),), dtype=np.int64)

        self._cache_patient_idx = patient_idx
        self._cache_X = X
        self._cache_y = y

    def __getitem__(self, idx: int):
        cfg = self.cfg
        patient_idx, t = self._locate(idx)
        self._load_patient_cached(patient_idx)

        assert self._cache_X is not None and self._cache_y is not None
        X = self._cache_X
        y = self._cache_y

        # causal window ending at t (no future leakage)
        start = max(0, t - cfg.seq_len + 1)
        chunk = X[start:t + 1]
        L = chunk.shape[0]

        # Right-align: most recent timestep is always last position
        window = np.zeros((cfg.seq_len, X.shape[1]), dtype=np.float32)
        mask = np.zeros((cfg.seq_len,), dtype=np.float32)
        window[-L:] = chunk
        mask[-L:] = 1.0

        label = int(y[t])
        return (
            torch.from_numpy(window),               # (seq_len, n_features)
            torch.tensor(label, dtype=torch.long),  # scalar
            torch.from_numpy(mask),                 # (seq_len,)
        )

    def _assert_time_monotonic(self) -> None:
        """After sorting, verify time columns are non-decreasing within each patient."""
        cfg = self.cfg
        df = self._df
        if df is None:
            return

        check_cols = [c for c in cfg.time_cols_priority if c in df.columns]
        if not check_cols:
            return

        # check each patient slice
        for i, pid in enumerate(self.patient_ids):
            start = int(self.patient_offsets[i])
            L = int(self.patient_lengths[i])
            end = start + L
            dfp = df.iloc[start:end]

            for c in check_cols:
                v = pd.to_numeric(dfp[c], errors="coerce").to_numpy()
                if np.any(np.isnan(v)):
                    # If preprocessing happens elsewhere, NaNs may exist; skip strict check for that col
                    continue
                if np.any(np.diff(v) < 0):
                    raise ValueError(
                        f"Time column '{c}' is not monotonic for patient {pid} "
                        f"in {cfg.csv_path}. This can cause leakage."
                    )
