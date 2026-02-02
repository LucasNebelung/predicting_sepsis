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
  --model_id Sepsis_iTransformer_HIGH_CSV_FINAL \
  --seq_len 48 \
  --sample_step 1 \
  --batch_size 256 \
  --num_workers 12 \
  --learning_rate 0.0001 \
  --train_epochs 12 \
  --patience 2 \
  --d_model 128 \
  --d_ff 512 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.30 \
  --top_k 3 \
  --pos_weight 15 \
  --clip_grad 0.5 \
  --itr 1 \
  --des iTransformer_SepsisCSV_FINAL_small_reg_lr1e-4_bs128_pw15_clip0p5
