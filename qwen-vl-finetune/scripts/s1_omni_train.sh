#!/bin/bash
SWANLAB_PROJECT="${SWANLAB_PROJECT:-qwen3-vl-multitask}"
RUN_NAME="${RUN_NAME:-stage2-$(date +%Y%m%d-%H%M%S)}"
export SWANLAB_API_KEY="vq9w39sUAuxOYTywrqhud"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT}"

export S1OMNI_DEBUG_LAST_TOKEN="1"
# export S1OMNI_DEBUG_DTYPE="1"

torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29503 \
  qwenvl/train/train_qwen.py \
  --deepspeed /data/home/zdhs0092/Code/S1-Omni-pro/qwen-vl-finetune/scripts/zero2.json \
  --model_architecture s1-omni \
  --model_name_or_path /data/home/zdhs0092/Code/ms-swift/train_bash/output/s1_vl_32b_rl_0515_and_0501_131k_sft/v2-20260521-114812/checkpoint-726 \
  --annotation_path /data/home/zdhs0092/Code/ms-swift/data/sft_messages_0515_and_0501_131k.jsonl \
  --bf16 \
  --output_dir /data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_ep150 \
  --num_train_epochs 150 \
  --save_only_model True \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --weight_decay 0 \
  --warmup_ratio 0.03 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 10000 \
  --save_total_limit 3 \
  --eval_strategy no \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 16 \
  --report_to "swanlab" \
  --run_name "${RUN_NAME}" \
  --pooled_hidden_cache_path "/data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_pooled_hidden.pt"
