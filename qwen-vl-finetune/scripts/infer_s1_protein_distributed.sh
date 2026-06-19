#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29513}"
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_allosteric_site_ep6/checkpoint-30}"
BATCH_FILE="${BATCH_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/protein_pre_data/allosteric_site/test/protein_site_prediction-regulatory_site-allosteric_site.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_allosteric.jsonl}"


# CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_DNA_binding_site_ep6/checkpoint-12}"
# BATCH_FILE="${BATCH_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/protein_pre_data/DNA_binding_site/test/protein_site_prediction-interaction_site-DNA_binding_site.jsonl}"
# OUTPUT_FILE="${OUTPUT_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_DNA.jsonl}"


# CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_epitope_ep6/checkpoint-48}"
# BATCH_FILE="${BATCH_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/protein_pre_data/epitope/test/RoBep_test_epitope.jsonl}"
# OUTPUT_FILE="${OUTPUT_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_epitope.jsonl}"

# CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_paratope_ep6/checkpoint-12}"
# BATCH_FILE="${BATCH_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/protein_pre_data/paratope/test/protein_site_prediction-interaction_site-paratope.jsonl}"
# OUTPUT_FILE="${OUTPUT_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_paratope.jsonl}"


# CHECKPOINT_DIR="${CHECKPOINT_DIR:-/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_RNA_binding_site_ep6/checkpoint-6}"
# BATCH_FILE="${BATCH_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/protein_pre_data/RNA_binding_site/test/protein_site_prediction-interaction_site-RNA_binding_site.jsonl}"
# OUTPUT_FILE="${OUTPUT_FILE:-/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_RNA.jsonl}"


DEVICE="${DEVICE:-cuda:0}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-1}"
DTYPE="${DTYPE:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
THRESHOLD="${THRESHOLD:-0.5}"
AUTO_THRESHOLD="${AUTO_THRESHOLD:-true}"
OPTIMIZE_THRESHOLD_METRIC="${OPTIMIZE_THRESHOLD_METRIC:-f1_then_mcc}"

export NCCL_SOCKET_IFNAME
export GLOO_SOCKET_IFNAME
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"
export NCCL_IB_ECE_ENABLE="${NCCL_IB_ECE_ENABLE:-0}"
export NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-0}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

echo "========================================"
echo "S1-Protein Distributed Inference"
echo "========================================"
echo "Checkpoint      : ${CHECKPOINT_DIR}"
echo "Batch file      : ${BATCH_FILE}"
echo "Output          : ${OUTPUT_FILE}"
echo "NPROC_PER_NODE  : ${NPROC_PER_NODE}"
echo "NNODES          : ${NNODES}"
echo "NODE_RANK       : ${NODE_RANK}"
echo "MASTER_ADDR     : ${MASTER_ADDR}"
echo "MASTER_PORT     : ${MASTER_PORT}"
echo "Batch size/rank : ${INFER_BATCH_SIZE}"
echo "Threshold       : ${THRESHOLD}"
echo "Auto threshold  : ${AUTO_THRESHOLD}"
echo "Optimize metric : ${OPTIMIZE_THRESHOLD_METRIC}"
echo "========================================"

AUTO_THRESHOLD_ARG="--auto_threshold"
case "${AUTO_THRESHOLD}" in
  0|false|False|FALSE|no|No|NO)
    AUTO_THRESHOLD_ARG="--no-auto_threshold"
    ;;
esac

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${PROJECT_DIR}/infer_s1_protein_checkpoint.py" \
  --checkpoint_dir "${CHECKPOINT_DIR}" \
  --batch_file "${BATCH_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --device "${DEVICE}" \
  --batch_size "${INFER_BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --threshold "${THRESHOLD}" \
  "${AUTO_THRESHOLD_ARG}" \
  --optimize_threshold_metric "${OPTIMIZE_THRESHOLD_METRIC}"
