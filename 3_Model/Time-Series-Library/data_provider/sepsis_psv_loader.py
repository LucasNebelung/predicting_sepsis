import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

LABEL_COL = "SepsisLabel"

MODE_NO = "NO_PREPROC_NO_FE"
MODE_LOW = "LOW_PREPROC_NO_FE"
MODE_HIGH = "HIGH_PREPROC_NO_FE"
VALID_MODES = {MODE_NO, MODE_LOW, MODE_HIGH}

# ============================================================
# Keep these features as VALUES, but DO NOT add recency_* for them
# (per your request)
# ============================================================
NO_RECENCY_COLS = {
    "Age",
    "Gender",
    "Unit1",
    "Unit2",
    "HospAdmTime",
    "ICULOS",
}


def _read_psv_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|")
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


def _ensure_feature_order(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[feature_cols]


def _get_mode(args) -> str:
    mode = getattr(args, "preproc_mode", None)
    if mode is None:
        impute = str(getattr(args, "impute_method", "zero")).lower()
        return MODE_LOW if impute == "ffill" else MODE_NO

    mode = str(mode).upper()
    if mode not in VALID_MODES:
        raise ValueError(f"--preproc_mode must be one of {sorted(VALID_MODES)} (got {mode})")
    return mode


def _cache_root(args) -> str:
    cache_parent = getattr(args, "npz_cache_dir", "NPZ_Patients_preprocessed")
    return os.path.join(args.root_path, cache_parent)


def _cache_path(args, psv_path: str) -> str:
    """
    dataset/NPZ_Patients_preprocessed/<MODE>/<split_folder>/pXXXX.npz
    """
    mode = _get_mode(args)
    cache_root = _cache_root(args)

    split_folder = os.path.basename(os.path.dirname(psv_path))
    out_dir = os.path.join(cache_root, mode, split_folder)
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(psv_path))[0]
    return os.path.join(out_dir, base + ".npz")


def _recency_from_missing(missing: np.ndarray, decay: float) -> np.ndarray:
    """
    missing: (T, F) bool (True if missing)
    recency: (T, F) float in [0,1]
      observed now -> 1
      not observed -> prev * decay
      never observed -> 0
    """
    T, F = missing.shape
    rec = np.zeros((T, F), dtype=np.float32)
    for t in range(T):
        obs = ~missing[t]
        if t == 0:
            rec[t, obs] = 1.0
        else:
            rec[t] = rec[t - 1] * decay
            rec[t, obs] = 1.0
    return rec


def _make_X_model(args, X_raw: np.ndarray, missing: np.ndarray, feature_cols: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Build model-ready X and the matching column list.

    - NO: NaN->0, columns = feature_cols
    - LOW: ffill then NaN->0, columns = feature_cols
    - HIGH: ffill then NaN->0, append recency ONLY for columns not in NO_RECENCY_COLS
            columns = feature_cols + recency_cols_subset
    """
    mode = _get_mode(args)

    # Values are ALWAYS all feature_cols (including Age/Gender/Unit/HospAdmTime/ICULOS)
    base_cols = list(feature_cols)

    if mode == MODE_NO:
        X = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)
        return X, base_cols

    # LOW or HIGH: causal ffill
    X_df = pd.DataFrame(X_raw)
    X_df = X_df.ffill().fillna(0.0)
    X_vals = X_df.to_numpy(dtype=np.float32)

    if mode == MODE_LOW:
        return X_vals, base_cols

    # HIGH: append recency only for dynamic columns
    decay = float(getattr(args, "recency_decay", 0.9))

    recency_indices = [i for i, c in enumerate(base_cols) if c not in NO_RECENCY_COLS]
    recency_cols = [f"recency_{base_cols[i]}" for i in recency_indices]

    if len(recency_indices) == 0:
        # degenerate case: no recency features
        return X_vals, base_cols

    missing_sub = missing[:, recency_indices]
    rec = _recency_from_missing(missing_sub, decay=decay)  # (T, F_dyn)

    X = np.concatenate([X_vals, rec], axis=1).astype(np.float32)
    return X, base_cols + recency_cols


def build_psv_cache_one(args, psv_path: str, feature_cols: list[str]) -> str:
    """
    Cache NPZ stores:
      X_raw: (T, F) raw values (may contain NaNs)
      X:     (T, D) model values (D=F for NO/LOW, D=F+F_dyn for HIGH)
      y:     (T,)
      missing: (T, F) bool based on X_raw NaNs
      feature_cols: base cols (F)
      model_cols: actual columns of X (D)
    """
    out = _cache_path(args, psv_path)
    if os.path.exists(out) and int(getattr(args, "rebuild_cache", 0)) == 0:
        return out

    df = _read_psv_df(psv_path)
    has_label = LABEL_COL in df.columns

    feat_df = df.drop(columns=[LABEL_COL], errors="ignore")
    feat_df = _ensure_feature_order(feat_df, feature_cols)

    X_raw = feat_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    missing = np.isnan(X_raw)

    if has_label:
        y = df[LABEL_COL].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    else:
        y = np.zeros((len(df),), dtype=np.int64)

    X, model_cols = _make_X_model(args, X_raw, missing, feature_cols)

    np.savez_compressed(
        out,
        X_raw=X_raw,
        X=X,
        y=y,
        missing=missing,
        feature_cols=np.array(feature_cols, dtype=object),
        model_cols=np.array(model_cols, dtype=object),
        no_recency_cols=np.array(sorted(NO_RECENCY_COLS), dtype=object),
        mode=np.array([_get_mode(args)], dtype=object),
    )
    return out


def build_psv_cache_dir(args, psv_dir: str):
    files = sorted([os.path.join(psv_dir, f) for f in os.listdir(psv_dir) if f.endswith(".psv")])
    if not files:
        raise ValueError(f"No .psv files in {psv_dir}")

    df0 = _read_psv_df(files[0])
    feature_cols = [c for c in df0.columns if c != LABEL_COL]

    for p in files:
        build_psv_cache_one(args, p, feature_cols)


class SepsisPSVWindowDataset(Dataset):
    """
    One sample = one hour t from one patient, with causal window ending at t.
    """
    def __init__(self, args, files: list[str], flag: str):
        super().__init__()
        self.args = args
        self.flag = str(flag).upper()
        self.files = list(files)
        if not self.files:
            raise ValueError("SepsisPSVWindowDataset: empty files list")

        self.seq_len = int(getattr(args, "seq_len", 48))
        self.sample_step = int(max(1, getattr(args, "sample_step", 1)))
        self.mode = _get_mode(args)

        # Optional missing mask channel:
        # If you’re using HIGH (values+recency), you asked to NOT add missing mask.
        self.add_missing_mask = int(getattr(args, "add_missing_mask", 0)) == 1
        if self.mode == MODE_HIGH:
            self.add_missing_mask = False

        # base feature order from first file
        df0 = _read_psv_df(self.files[0])
        self.feature_cols = [c for c in df0.columns if c != LABEL_COL]

        # Determine model input columns (needed by Exp_Classification)
        # Try to match cached model_cols logic
        base_cols = list(self.feature_cols)
        if self.mode == MODE_HIGH:
            recency_cols = [f"recency_{c}" for c in base_cols if c not in NO_RECENCY_COLS]
            cols = base_cols + recency_cols
        else:
            cols = base_cols

        if self.add_missing_mask:
            cols = cols + [f"missing_{c}" for c in base_cols]

        self.feature_df = pd.DataFrame(columns=cols)
        self.class_names = [0, 1]
        self.max_seq_len = self.seq_len

        # global index mapping
        lengths = []
        for p in self.files:
            df = _read_psv_df(p)
            lengths.append(len(df))
        self.file_lengths = np.asarray(lengths, dtype=np.int64)
        self.file_offsets = np.cumsum(np.concatenate([[0], self.file_lengths[:-1]]))
        self.total_raw = int(self.file_lengths.sum())
        self.total_len = (self.total_raw + self.sample_step - 1) // self.sample_step

        self._cache_file_idx = None
        self._cache_X = None
        self._cache_y = None
        self._cache_missing = None

    def __len__(self):
        return self.total_len

    def _locate(self, idx: int):
        raw_idx = int(idx) * self.sample_step
        if raw_idx >= self.total_raw:
            raw_idx = self.total_raw - 1
        fi = np.searchsorted(self.file_offsets, raw_idx, side="right") - 1
        t = int(raw_idx - self.file_offsets[fi])
        return int(fi), int(t)

    def _load_patient_cached(self, file_idx: int):
        if self._cache_file_idx == file_idx:
            return

        psv_path = self.files[file_idx]

        if int(getattr(self.args, "cache_psv", 1)) == 1:
            import time
            t0 = time.time()

            npz_path = build_psv_cache_one(self.args, psv_path, self.feature_cols)

            built_now = False
            try:
                built_now = (os.path.getmtime(npz_path) >= t0)
            except Exception:
                pass

            if not hasattr(self, "_printed_cache_status"):
                mode = getattr(self.args, "preproc_mode", None) or getattr(self.args, "impute_method", None) or "UNKNOWN"
                rebuild = int(getattr(self.args, "rebuild_cache", 0))
                if built_now:
                    print(f"[SepsisPSV] ⚠️  Using NPZ cache (BUILT NOW). mode={mode}, rebuild_cache={rebuild}")
                else:
                    print(f"[SepsisPSV] ✅ Using NPZ cache (HIT). mode={mode}, rebuild_cache={rebuild}")
                self._printed_cache_status = True

            npz = np.load(npz_path, allow_pickle=True)

            X = npz["X"].astype(np.float32)
            y = npz["y"].astype(np.int64)
            missing = npz["missing"].astype(bool)

            if self.add_missing_mask:
                X = np.concatenate([X, missing.astype(np.float32)], axis=1).astype(np.float32)

        else:
            if not hasattr(self, "_printed_cache_status"):
                mode = getattr(self.args, "preproc_mode", None) or getattr(self.args, "impute_method", None) or "UNKNOWN"
                print(f"[SepsisPSV] 🚫 Using raw PSV (no cache). mode={mode}")
                self._printed_cache_status = True

            df = _read_psv_df(psv_path)
            feat_df = df.drop(columns=[LABEL_COL], errors="ignore")
            feat_df = _ensure_feature_order(feat_df, self.feature_cols)

            X_raw = feat_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
            missing = np.isnan(X_raw)

            if LABEL_COL in df.columns:
                y = df[LABEL_COL].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.int64)
            else:
                y = np.zeros((len(df),), dtype=np.int64)

            X, _ = _make_X_model(self.args, X_raw, missing, self.feature_cols)

            if self.add_missing_mask:
                X = np.concatenate([X, missing.astype(np.float32)], axis=1).astype(np.float32)

        # ✅ THIS WAS MISSING
        self._cache_file_idx = file_idx
        self._cache_X = X
        self._cache_y = y
        self._cache_missing = missing


    def __getitem__(self, idx: int):
        file_idx, t = self._locate(idx)
        self._load_patient_cached(file_idx)

        X = self._cache_X
        y = self._cache_y

        start = max(0, t - self.seq_len + 1)
        chunk = X[start:t + 1]
        L = chunk.shape[0]

        window = np.zeros((self.seq_len, X.shape[1]), dtype=np.float32)
        mask = np.zeros((self.seq_len,), dtype=np.float32)

        window[:L] = chunk
        mask[:L] = 1.0

        label = int(y[t])
        return (
            torch.from_numpy(window),
            torch.tensor([label], dtype=torch.long),
            torch.from_numpy(mask),
        )
