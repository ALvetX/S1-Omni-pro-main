#!/bin/bash
set -euo pipefail

SWANLAB_PROJECT="${SWANLAB_PROJECT:-qwen3-vl-multitask}"
RUN_NAME="${RUN_NAME:-protein_RNA_ep6_4}"
ANNOTATION_PATH="${ANNOTATION_PATH:-/nfs-12/liujunyi/S1-Omni-pro-main/data/data_expansion/RNA_binding_site_data/train/protein_site_prediction-interaction_site-RNA_binding_site.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_RNA_ep6_4}"
NNODES="${NNODES:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-10.20.4.9}"
MASTER_PORT="${MASTER_PORT:-29503}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"
RDZV_CONF="${RDZV_CONF:-timeout=600}"
HOST_SHORT="$(hostname -s)"

if [[ -z "${NODE_RANK:-}" ]]; then
  case "${HOST_SHORT}" in
    wg-4-9)
      NODE_RANK="${NODE_RANK:-0}"
      ;;
    wg-4-14)
      NODE_RANK="${NODE_RANK:-1}"
      ;;
    *)
      echo "Unable to infer NODE_RANK from hostname ${HOST_SHORT}." >&2
      echo "Set NODE_RANK=0 on wg-4-9 or NODE_RANK=1 on wg-4-14." >&2
      exit 1
      ;;
  esac
fi

if [[ -z "${NCCL_IB_HCA:-}" ]]; then
  case "${HOST_SHORT}" in
    wg-4-9)
      NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"
      ;;
    wg-4-14)
      NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
      ;;
  esac
fi

export SWANLAB_API_KEY="XGdUa86OmsTjJew8ql8hn"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJECT}"
export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME
export NCCL_IB_HCA
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
export NCCL_IB_ECE_ENABLE="${NCCL_IB_ECE_ENABLE:-0}"
export NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-0}"
export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"
export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"

# export S1OMNI_DEBUG_LAST_TOKEN="1"
# export S1OMNI_DEBUG_DTYPE="1"

echo "Distributed environment:"
echo "  hostname=${HOST_SHORT}"
echo "  NODE_RANK=${NODE_RANK}"
echo "  NNODES=${NNODES}"
echo "  NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "  MASTER_ADDR=${MASTER_ADDR}"
echo "  MASTER_PORT=${MASTER_PORT}"
echo "  NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "  GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME}"
echo "  NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "  NCCL_IB_ECE_ENABLE=${NCCL_IB_ECE_ENABLE}"
echo "  NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS}"
echo "  RDZV_CONF=${RDZV_CONF}"

torchrun --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_conf="${RDZV_CONF}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  /nfs-12/liujunyi/S1-Omni-pro-main/qwen-vl-finetune/qwenvl/train/train_qwen.py \
  --deepspeed /nfs-12/liujunyi/S1-Omni-pro-main/qwen-vl-finetune/scripts/zero3.json \
  --model_architecture s1-protein \
  --model_name_or_path /data3/xlf_model/S1-VL-32B-RL \
  --use_esm2 True \
  --esm_model_name /nfs-12/liujunyi/S1-Omni-pro-main/model/nv-community/esm2_t36_3B_UR50D \
  --esm_fusion_dim 512 \
  --esm_num_attention_heads 8 \
  --esm_fusion_num_layers 16 \
  --esm_fusion_ffn_dim 2048 \
  --esm_unfreeze_last_n_layers 8 \
  --esm_unfreeze_final_layer_norm True \
  --esm_unfreeze_pooler False \
  --esm_lr_multiplier 0.1 \
  --positive_loss_weight 10.0 \
  --protein_loss_type bce \
  --asl_gamma_pos 0.0 \
  --asl_gamma_neg 4.0 \
  --asl_clip 0.05 \
  --annotation_path "${ANNOTATION_PATH}" \
  --bf16 \
  --output_dir "${OUTPUT_DIR}" \
  --num_train_epochs 6 \
  --save_only_model True \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.1 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 1 \
  --eval_strategy no \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 16 \
  --report_to "swanlab" \
  --run_name "${RUN_NAME}"
