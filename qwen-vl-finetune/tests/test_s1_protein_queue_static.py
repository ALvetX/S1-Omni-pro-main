import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "qwen-vl-finetune" / "scripts"
QUEUE_SCRIPT = SCRIPT_DIR / "run_s1_protein_queue.sh"
TRAIN_SCRIPT = SCRIPT_DIR / "s1_protein.sh"
DATA_ROOT = REPO_ROOT / "protein_pre_data"
OUTPUT_ROOT = REPO_ROOT / "output"


def _script_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_array(text: str, name: str) -> list[str]:
    match = re.search(rf"{name}=\((.*?)\)", text, flags=re.DOTALL)
    assert match, f"{name} array missing"
    return re.findall(r'"([^"]+)"', match.group(1))


class S1ProteinQueueStaticTest(unittest.TestCase):
    def test_queue_runs_one_multinode_sequence_for_eight_non_enzyme_tasks(self):
        text = _script_text(QUEUE_SCRIPT)

        all_tasks = _extract_array(text, "TASKS")

        self.assertEqual(len(all_tasks), 8)
        self.assertNotIn("enzyme_active_site", all_tasks)
        self.assertEqual(
            all_tasks,
            [
                "paratope",
                "allosteric_site",
                "DNA_binding_site",
                "mol_binding_site",
                "epitope",
                "iron_binding_site",
                "RNA_binding_site",
                "PPI_binding_site",
            ],
        )
        self.assertNotIn("SERVER1_TASKS", text)
        self.assertNotIn("SERVER2_TASKS", text)
        self.assertNotIn("server1|server2", text)

    def test_queue_uses_existing_task_train_jsonl_and_expected_output_names(self):
        text = _script_text(QUEUE_SCRIPT)
        all_tasks = _extract_array(text, "TASKS")

        for task in all_tasks:
            train_files = list((DATA_ROOT / task / "train").glob("*.jsonl"))
            self.assertEqual(len(train_files), 1, f"{task} should have one train jsonl")

        self.assertIn('RUN_NAME="protein_${task}_ep6"', text)
        self.assertIn('OUTPUT_DIR="${OUTPUT_ROOT}/protein_${task}_ep6"', text)
        self.assertIn("ANNOTATION_PATH=", text)

    def test_training_script_accepts_queue_environment_overrides(self):
        text = _script_text(TRAIN_SCRIPT)

        self.assertIn('RUN_NAME="${RUN_NAME:-protein_allosteric_ep6}"', text)
        self.assertIn('ANNOTATION_PATH="${ANNOTATION_PATH:-', text)
        self.assertIn('OUTPUT_DIR="${OUTPUT_DIR:-', text)
        self.assertIn('--annotation_path "${ANNOTATION_PATH}"', text)
        self.assertIn('--output_dir "${OUTPUT_DIR}"', text)

    def test_training_script_defaults_to_two_nodes_and_auto_node_rank(self):
        text = _script_text(TRAIN_SCRIPT)

        self.assertIn('NNODES="${NNODES:-2}"', text)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', text)
        self.assertIn('MASTER_ADDR="${MASTER_ADDR:-10.20.4.9}"', text)
        self.assertIn('MASTER_PORT="${MASTER_PORT:-29503}"', text)
        self.assertIn('wg-4-9)', text)
        self.assertIn('NODE_RANK="${NODE_RANK:-0}"', text)
        self.assertIn('wg-4-14)', text)
        self.assertIn('NODE_RANK="${NODE_RANK:-1}"', text)
        self.assertIn('--nnodes="${NNODES}"', text)
        self.assertIn('--node_rank="${NODE_RANK}"', text)

    def test_training_script_checks_cuda_nccl_runtime_before_torchrun(self):
        text = _script_text(TRAIN_SCRIPT)

        self.assertIn("Checking PyTorch CUDA/NCCL runtime", text)
        self.assertIn("ctypes.CDLL", text)
        self.assertIn("ncclGetVersion", text)
        self.assertLess(text.index("Checking PyTorch CUDA/NCCL runtime"), text.index("torchrun"))

    def test_training_script_sets_nccl_network_and_failure_diagnostics(self):
        text = _script_text(TRAIN_SCRIPT)

        self.assertIn('NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond0}"', text)
        self.assertIn('GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME}}"', text)
        self.assertIn('export TORCH_NCCL_TRACE_BUFFER_SIZE="${TORCH_NCCL_TRACE_BUFFER_SIZE:-1048576}"', text)
        self.assertIn('export TORCH_NCCL_DUMP_ON_TIMEOUT="${TORCH_NCCL_DUMP_ON_TIMEOUT:-1}"', text)
        self.assertIn('export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"', text)
        self.assertIn('export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"', text)
        self.assertIn('export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"', text)
        self.assertIn('export NCCL_IB_ECE_ENABLE="${NCCL_IB_ECE_ENABLE:-0}"', text)
        self.assertIn('export NCCL_IB_MERGE_NICS="${NCCL_IB_MERGE_NICS:-0}"', text)
        self.assertIn('NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_2,mlx5_3}"', text)
        self.assertIn('NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"', text)
        self.assertIn("echo \"  NCCL_IB_HCA=${NCCL_IB_HCA}\"", text)
        self.assertIn('--rdzv_backend=c10d', text)
        self.assertIn('--rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"', text)
        self.assertIn('--rdzv_conf="${RDZV_CONF}"', text)


if __name__ == "__main__":
    unittest.main()
