#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."   # -> Time-Series-Library/

export CUDA_VISIBLE_DEVICES=""  # caching is CPU-only

python - <<'PY'
from types import SimpleNamespace as S
from data_provider.sepsis_psv_loader import build_psv_cache_dir

args = S(
    root_path="./dataset",
    npz_cache_dir="NPZ_Patients_preprocessed",
    cache_psv=1,
    rebuild_cache=1,  # set to 0 if you want to reuse existing
    preproc_mode="NO_PREPROC_NO_FE",
    impute_method="zero",   # not strictly needed if preproc_mode is used
)

# Raw PSV folders (your new structure)
build_psv_cache_dir(args, "./dataset/raw_PSV_Patient_Files/PSV_Patients_TRAIN_FIT")
build_psv_cache_dir(args, "./dataset/raw_PSV_Patient_Files/PSV_Patients_TRAIN_THRESH")
build_psv_cache_dir(args, "./dataset/raw_PSV_Patient_Files/PSV_Patients_TEST")
print("DONE: NO_PREPROC_NO_FE")
PY
