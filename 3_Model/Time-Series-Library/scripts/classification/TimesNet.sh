export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --data SepsisPSV \
  --root_path ./dataset \
  --train_dir PSV_Patients_TRAIN_FIT \
  --test_dir PSV_Patients_TEST \
  --val_ratio 0.2 \
  --model TimesNet \
  --model_id Sepsis \
  --seq_len 48 \
  --batch_size 8 \
  --num_workers 0 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 10
