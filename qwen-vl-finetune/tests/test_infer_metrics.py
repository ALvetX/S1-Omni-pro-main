import math
import sys
import types
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

torch_stub = types.ModuleType("torch")
torch_stub.bfloat16 = "bfloat16"
torch_stub.float16 = "float16"
torch_stub.float32 = "float32"
torch_stub.bool = "bool"
torch_stub.is_tensor = lambda value: False
torch_stub.inference_mode = lambda: None
sys.modules.setdefault("torch", torch_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoTokenizer = object
sys.modules.setdefault("transformers", transformers_stub)

data_processor_stub = types.ModuleType("qwenvl.data.data_processor")
data_processor_stub._extract_protein_sequence_and_qwen_text = lambda question: (question, "")
data_processor_stub._find_subsequence = lambda haystack, needle: -1
data_processor_stub._space_protein_sequence = lambda question: (question, "")
sys.modules.setdefault("qwenvl.data.data_processor", data_processor_stub)

modeling_stub = types.ModuleType("qwenvl.modeling_s1_protein")
modeling_stub.S1Protein = object
sys.modules.setdefault("qwenvl.modeling_s1_protein", modeling_stub)

from infer import finalize_predictions  # noqa: E402


class InferMetricsTest(unittest.TestCase):
    def test_finalize_predictions_selects_best_threshold_and_reports_metrics(self):
        preds = [
            {
                "question": "q1",
                "answer": "[1]",
                "probabilities": [0.9, 0.2],
                "positive_indices": [1, 2],
                "bit_string": "11",
                "threshold": 0.1,
            },
            {
                "question": "q2",
                "answer": [2],
                "probabilities": [0.4, 0.8],
                "positive_indices": [1, 2],
                "bit_string": "11",
                "threshold": 0.1,
            },
        ]

        summary = finalize_predictions(
            preds,
            threshold=0.1,
            auto_threshold=True,
            optimize_metric="f1_then_mcc",
        )

        self.assertAlmostEqual(summary["selected_threshold"], 0.8)
        self.assertEqual(preds[0]["positive_indices"], [1])
        self.assertEqual(preds[0]["bit_string"], "10")
        self.assertEqual(preds[1]["positive_indices"], [2])
        self.assertEqual(preds[1]["bit_string"], "01")
        self.assertAlmostEqual(summary["metrics"]["Precision"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["Recall"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["F1"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["MCC"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["AUROC"], 1.0)
        self.assertAlmostEqual(summary["metrics"]["AUPR"], 1.0)

    def test_finalize_predictions_keeps_fallback_threshold_without_labels(self):
        preds = [
            {
                "question": "q",
                "answer": None,
                "probabilities": [0.7, 0.3],
                "positive_indices": [],
                "bit_string": "00",
                "threshold": 0.5,
            }
        ]

        summary = finalize_predictions(preds, threshold=0.5, auto_threshold=True)

        self.assertAlmostEqual(summary["selected_threshold"], 0.5)
        self.assertEqual(preds[0]["positive_indices"], [1])
        self.assertEqual(preds[0]["bit_string"], "10")
        self.assertTrue(math.isnan(summary["metrics"]["AUROC"]))
        self.assertTrue(math.isnan(summary["metrics"]["AUPR"]))


if __name__ == "__main__":
    unittest.main()
