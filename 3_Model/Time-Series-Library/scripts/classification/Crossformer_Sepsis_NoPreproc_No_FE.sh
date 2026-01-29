export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name classification \
  --is_training 0 \
  --data SepsisPSV \
  --root_path ./dataset \
  --train_dir raw_PSV_Patient_Files/PSV_Patients_TRAIN_FIT \
  --test_dir raw_PSV_Patient_Files/PSV_Patients_TEST \
  --val_ratio 0.2 \
  --model Crossformer \
  --model_id DEBUG \
  --preproc_mode NO_PREPROC_NO_FE \
  --cache_psv 1 \
  --rebuild_cache 0 \
  --impute_method zero \
  --add_missing_mask 0 \
  --seq_len 48 \
  --batch_size 32 \
  --num_workers 0 \
  --d_model 16 \
  --d_ff 32 \
  --n_heads 2 \
  --e_layers 1 \
  --learning_rate 0.001 \
  --train_epochs 1 \
  --patience 1 \
  --itr 1
