import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = PROJECT_ROOT / "infer_s1_protein_checkpoint.py"
LAUNCH_SCRIPT = PROJECT_ROOT / "scripts" / "infer_s1_protein_distributed.sh"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _function_names(path: Path) -> set[str]:
    tree = ast.parse(_text(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


class S1ProteinDistributedInferenceStaticTest(unittest.TestCase):
    def test_inference_defaults_to_allosteric_checkpoint_and_test_file(self):
        text = _text(INFER_SCRIPT)

        self.assertIn(
            "/nfs-12/liujunyi/S1-Omni-pro-main/output/protein_allosteric_site_ep6/checkpoint-30",
            text,
        )
        self.assertIn(
            "/nfs-12/liujunyi/S1-Omni-pro-main/protein_pre_data/allosteric_site/test/protein_site_prediction-regulatory_site-allosteric_site.jsonl",
            text,
        )
        self.assertIn(
            "/nfs-12/liujunyi/S1-Omni-pro-main/output/predict_protein/predictions_allosteric_site_checkpoint-30.jsonl",
            text,
        )

    def test_inference_script_has_distributed_rank_sharding_and_merge_helpers(self):
        names = _function_names(INFER_SCRIPT)
        text = _text(INFER_SCRIPT)

        for name in (
            "setup_distributed",
            "cleanup_distributed",
            "distributed_barrier",
            "is_rank0",
            "batched",
            "indexed_records_for_rank",
            "rank_output_path",
            "write_indexed_results",
            "read_rank_outputs",
            "cleanup_rank_outputs",
            "write_distributed_batch_results",
            "run_batch_inference",
            "finalize_predictions",
            "compute_metrics",
            "print_metrics_summary",
        ):
            self.assertIn(name, names)

        self.assertIn('idx % args.world_size == args.rank', text)
        self.assertIn("torch.distributed.init_process_group", text)
        self.assertIn("torch.distributed.destroy_process_group", text)
        self.assertIn("torch.cuda.set_device(args.local_rank)", text)
        self.assertIn("model.esm_tokenizer = esm_tokenizer", text)
        self.assertIn("--auto_threshold", text)
        self.assertIn("--optimize_threshold_metric", text)
        for metric_name in ("Precision", "Recall", "F1", "MCC", "AUROC", "AUPR"):
            self.assertIn(metric_name, text)

    def test_distributed_launch_script_defaults_to_allosteric_inference(self):
        text = _text(LAUNCH_SCRIPT)

        self.assertIn("CHECKPOINT_DIR=", text)
        self.assertIn("BATCH_FILE=", text)
        self.assertIn("OUTPUT_FILE=", text)
        self.assertIn("--nproc_per_node", text)
        self.assertIn("infer_s1_protein_checkpoint.py", text)
        self.assertIn("--batch_size", text)
        self.assertIn("AUTO_THRESHOLD=", text)
        self.assertIn("OPTIMIZE_THRESHOLD_METRIC=", text)
        self.assertIn("--optimize_threshold_metric", text)


if __name__ == "__main__":
    unittest.main()
