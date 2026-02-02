
export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --data SepsisCSV \
  --root_path /teamspace/studios/this_studio/detecting_Sepsis/data \
  --train_dir High_Preproc_TrainMeanImpute_CSV/train_fit_high_preproc_Mean_Impute.csv \
  --thresh_dir High_Preproc_TrainMeanImpute_CSV/train_thresh_high_preproc_Mean_Impute.csv \
  --test_dir High_Preproc_TrainMeanImpute_CSV/test_high_preproc_Mean_Impute.csv \
  --k_fold 5 \
  --fold 0 \
  --model TemporalFusionTransformer \
  --model_id Sepsis_TFT_HIGH_CSV_MedImpute \
  --seq_len 48 \
  --sample_step 1 \
  --batch_size 256 \
  --num_workers 4 \
  --learning_rate 0.0002 \
  --train_epochs 12 \
  --patience 2 \
  --d_model 64 \
  --d_ff 256 \
  --n_heads 4 \
  --e_layers 2 \
  --dropout 0.3 \
  --itr 1 \
  --pos_weight 20 \
  --clip_grad 0.5 \
  --des TFT_on_CSV_batchsize256_clip_grad05_medimpute
