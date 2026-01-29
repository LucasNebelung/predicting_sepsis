export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --data SepsisPSV \
  --root_path ./dataset \
  --train_dir raw_PSV_Patient_Files/PSV_Patients_TRAIN_FIT \
  --test_dir raw_PSV_Patient_Files/PSV_Patients_TEST \
  --val_ratio 0.2 \
  --model Crossformer \
  --model_id Sepsis_Crossformer_HIGH \
  --preproc_mode HIGH_PREPROC_NO_FE \
  --cache_psv 1 \
  --rebuild_cache 0 \
  --impute_method ffill \
  --add_missing_mask 0 \
  --seq_len 48 \
  --batch_size 64 \
  --num_workers 8 \
  --learning_rate 3e-4 \
  --train_epochs 20 \
  --patience 5 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 2 \
  --itr 1 \
  --des HIGH_PREPROC_NO_FE
