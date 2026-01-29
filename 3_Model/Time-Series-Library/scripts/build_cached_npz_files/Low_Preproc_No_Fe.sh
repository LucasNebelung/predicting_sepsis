#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export CUDA_VISIBLE_DEVICES=""

python - <<'PY'
from types import SimpleNamespace as S
from data_provider.sepsis_psv_loader import build_psv_cache_dir

args = S(
    root_path="./dataset",
    npz_cache_dir="NPZ_Patients_preprocessed",
    cache_psv=1,
    rebuild_cache=1,
    preproc_mode="LOW_PREPROC_NO_FE",
    impute_method="ffill",
)

build_psv_cache_dir(args, "./dataset/raw_PSV_Patient_Files/PSV_Patients_TRAIN_FIT")
build_psv_cache_dir(args, "./dataset/raw_PSV_Patient_Files/PSV_Patients_TRAIN_THRESH")
build_psv_cache_dir(args, "./dataset/raw_PSV_Patient_Files/PSV_Patients_TEST")
print("DONE: LOW_PREPROC_NO_FE")
PY
