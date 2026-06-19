#!/bin/bash
set -euo pipefail

REPO_ROOT="/nfs-12/liujunyi/S1-Omni-pro-main"
DATA_ROOT="${REPO_ROOT}/protein_pre_data"
OUTPUT_ROOT="${REPO_ROOT}/output"
TRAIN_SCRIPT="${REPO_ROOT}/qwen-vl-finetune/scripts/s1_protein.sh"

TASKS=(
  "RNA_binding_site"
)

usage() {
  echo "Usage: bash $0"
  echo
  echo "Run this same command on both nodes:"
  echo "  wg-4-9  NODE_RANK=0, master 10.20.4.9"
  echo "  wg-4-14 NODE_RANK=1"
  echo
  echo "tasks: ${TASKS[*]}"
}

case "${1:-}" in
  "")
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 1
    ;;
esac

for task in "${TASKS[@]}"; do
  train_dir="${DATA_ROOT}/${task}/train"
  mapfile -t train_files < <(find "${train_dir}" -maxdepth 1 -type f -name "*.jsonl" | sort)

  if [[ "${#train_files[@]}" -ne 1 ]]; then
    echo "Expected exactly one train jsonl in ${train_dir}, found ${#train_files[@]}" >&2
    exit 1
  fi

  RUN_NAME="protein_${task}_ep6"
  ANNOTATION_PATH="${train_files[0]}"
  OUTPUT_DIR="${OUTPUT_ROOT}/protein_${task}_ep6"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting ${task}"
  echo "RUN_NAME=${RUN_NAME}"
  echo "ANNOTATION_PATH=${ANNOTATION_PATH}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"

  NNODES="${NNODES:-2}" \
  NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
  MASTER_ADDR="${MASTER_ADDR:-10.20.4.9}" \
  MASTER_PORT="${MASTER_PORT:-29503}" \
  RUN_NAME="${RUN_NAME}" \
  ANNOTATION_PATH="${ANNOTATION_PATH}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
    bash "${TRAIN_SCRIPT}"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished ${task}"
done
