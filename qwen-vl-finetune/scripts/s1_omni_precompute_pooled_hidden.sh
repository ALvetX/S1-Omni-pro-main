#!/bin/bash
set -euo pipefail

OUTPUT_PATH="${OUTPUT_PATH:-/data/home/zdhs0092/Code/S1-Omni-pro/output/s1_omni_pooled_hidden.pt}"

torchrun --nproc_per_node="${NPROC_PER_NODE:-8}" \
  --master_addr="${MASTER_ADDR:-127.0.0.1}" \
  --master_port="${MASTER_PORT:-29504}" \
  qwen-vl-finetune/scripts/precompute_s1_omni_pooled_hidden.py \
  --model_architecture s1-omni \
  --model_name_or_path /data/home/zdhs0092/Code/ms-swift/train_bash/output/s1_vl_32b_rl_0515_and_0501_131k_sft/v2-20260521-114812/checkpoint-726 \
  --annotation_path /data/home/zdhs0092/Code/ms-swift/data/sft_messages_0515_and_0501_131k.jsonl \
  --output_path "${OUTPUT_PATH}" \
  --per_device_batch_size "${PER_DEVICE_BATCH_SIZE:-1}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-4}" \
  --bf16 True \
  --cache_dtype bfloat16 \
  --overwrite "${OVERWRITE:-False}"
