import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = PROJECT_ROOT / "infer.py"


class InferDefaultsStaticTest(unittest.TestCase):
    def test_infer_trusts_local_esm_custom_code_without_prompting(self):
        text = INFER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("trust_remote_code=True", text)


if __name__ == "__main__":
    unittest.main()
