#!/usr/bin/env bash
set -e

export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --data SepsisCSV \
  --root_path /teamspace/studios/this_studio/detecting_Sepsis/data \
  --train_dir High_Preproc_NoFe_CSV/train_fit_HIGH_PREPROC_NO_FE.csv \
  --thresh_dir High_Preproc_NoFe_CSV/train_thresh_HIGH_PREPROC_NO_FE.csv \
  --test_dir High_Preproc_NoFe_CSV/test_HIGH_PREPROC_NO_FE.csv \
  --k_fold 5 \
  --fold 0 \
  --model TemporalFusionTransformer \
  --model_id Sepsis_TFT_HIGH_CSV \
  --seq_len 48 \
  --sample_step 1 \
  --batch_size 512 \
  --num_workers 6 \
  --learning_rate 0.0001 \
  --train_epochs 12 \
  --patience 2 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.3 \
  --itr 1 \
  --pos_weight 20 \
  --clip_grad 1.0 \
  --des TFT_on_CSV_Debugging