import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qwenvl.modeling_s1_protein import S1Protein  # noqa: E402


class S1ProteinEsmSourceResolutionTest(unittest.TestCase):
    def test_checkpoint_esm_dir_missing_remote_code_is_not_used_as_model_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            esm_dir = checkpoint_dir / "esm2"
            esm_dir.mkdir()
            save_file({"embeddings.word_embeddings.weight": torch.ones(1, 1)}, esm_dir / "model.safetensors")
            (esm_dir / "config.json").write_text(
                json.dumps({"auto_map": {"AutoModel": "esm_nv.NVEsmModel"}}),
                encoding="utf-8",
            )

            self.assertEqual(
                S1Protein._resolve_esm_model_source(str(checkpoint_dir)),
                str(checkpoint_dir),
            )

    def test_bundled_esm_loader_accepts_non_loadable_inv_freq_extra_key(self):
        class TinyEsm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1, 1))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            esm_dir = checkpoint_dir / "esm2"
            esm_dir.mkdir()
            save_file(
                {
                    "weight": torch.ones(1, 1),
                    "encoder.rotary_embeddings.inv_freq": torch.ones(1),
                },
                esm_dir / "model.safetensors",
            )

            model = object.__new__(S1Protein)
            torch.nn.Module.__init__(model)
            model.esm_model = TinyEsm()
            model.esm_model_name = "dummy-esm"

            self.assertTrue(
                S1Protein._load_esm2_from_pretrained(
                    model,
                    str(checkpoint_dir),
                    require_bundled=True,
                )
            )


class S1ProteinHeadLoadingTest(unittest.TestCase):
    def test_custom_head_loader_reports_applied_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            weights = {
                "protein_head.0.weight": torch.full((4,), 2.0),
                "protein_head.0.bias": torch.full((4,), 3.0),
                "protein_head.1.weight": torch.full((2, 4), 4.0),
                "protein_head.1.bias": torch.full((2,), 5.0),
                "protein_head.4.weight": torch.full((1, 2), 6.0),
                "protein_head.4.bias": torch.full((1,), 7.0),
            }
            save_file(weights, checkpoint_dir / "model.safetensors")

            model = object.__new__(S1Protein)
            torch.nn.Module.__init__(model)
            model.protein_head = torch.nn.Sequential(
                torch.nn.LayerNorm(4),
                torch.nn.Linear(4, 2),
                torch.nn.GELU(),
                torch.nn.Dropout(0.0),
                torch.nn.Linear(2, 1),
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                report = S1Protein._load_head_from_pretrained(model, str(checkpoint_dir))

            self.assertEqual(report["loaded_tensors"], 6)
            self.assertIn("[S1Protein] custom module LOAD REPORT", stdout.getvalue())
            self.assertIn("protein_head: loaded 6/6", stdout.getvalue())
            self.assertTrue(torch.equal(model.protein_head[4].bias, weights["protein_head.4.bias"]))


class S1ProteinDefaultsTest(unittest.TestCase):
    def test_protein_defaults_use_8_esm_layers_and_16_cross_attention_layers(self):
        text = (PROJECT_ROOT / "qwenvl" / "modeling_s1_protein.py").read_text(encoding="utf-8")
        self.assertIn("esm_fusion_num_layers: int = 16", text)
        self.assertIn("esm_unfreeze_last_n_layers: int = 8", text)


if __name__ == "__main__":
    unittest.main()
