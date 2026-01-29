for f in 0 1 2 3 4; do
  export CUDA_VISIBLE_DEVICES=0
  python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ./dataset/Heartbeat/ \
    --model_id Heartbeat \
    --model TimesNet \
    --data UEA \
    --e_layers 3 \
    --batch_size 16 \
    --num_workers 0 \
    --d_model 16 \
    --d_ff 32 \
    --top_k 1 \
    --des "Exp_fold${f}" \
    --itr 1 \
    --learning_rate 0.001 \
    --train_epochs 30 \
    --patience 10 \
    --k_fold 5 \
    --fold $f
done
