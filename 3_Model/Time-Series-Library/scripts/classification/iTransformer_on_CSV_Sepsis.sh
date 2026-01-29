export CUDA_VISIBLE_DEVICES=0

model_name=iTransformer

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
  --model_id Sepsis_iTransformer_HIGH_CSV \
  --seq_len 48 \
  --sample_step 1 \
  --batch_size 256 \
  --num_workers 4 \
  --learning_rate 0.0005 \
  --train_epochs 12 \
  --patience 2 \
  --d_model 256 \
  --d_ff 1024 \
  --n_heads 8 \
  --e_layers 3 \
  --dropout 0.15 \
  --top_k 3 \
  --pos_weight 25 \
  --clip_grad 1.0 \
  --itr 1 \
  --des iTransformer_SepsisCSV
  