export CUDA_VISIBLE_DEVICES=0

model_name=Chronos2

# Expected GPU memory: ~20-24GB with batch_size=256 and seq_len=48.
# Fallback for smaller GPUs (e.g., 12-16GB): reduce batch_size to 128 or 64,
# set num_workers to 4, and keep clip_grad=1.0 for stable training.

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --data SepsisCSV \
  --root_path /teamspace/studios/this_studio/detecting_Sepsis/data \
  --train_dir High_Preproc_NoFe_CSV/train_fit_HIGH_PREPROC_NO_FE.csv \
  --thresh_dir High_Preproc_NoFe_CSV/train_thresh_HIGH_PREPROC_NO_FE.csv \
  --test_dir High_Preproc_NoFe_CSV/test_HIGH_PREPROC_NO_FE.csv \
  --k_fold 5 \
  --fold 0 \
  --model $model_name \
  --model_id Sepsis_Chronos2_HIGH_CSV \
  --seq_len 48 \
  --sample_step 1 \
  --batch_size 256 \
  --num_workers 8 \
  --learning_rate 0.0005 \
  --train_epochs 12 \
  --patience 2 \
  --d_model 256 \
  --d_ff 1024 \
  --e_layers 3 \
  --dropout 0.15 \
  --itr 1 \
  --pos_weight 25 \
  --clip_grad 1.0 \
  --des Chronos2_on_CSV_seq48_bs256_dm256_el3_lr5e-4
