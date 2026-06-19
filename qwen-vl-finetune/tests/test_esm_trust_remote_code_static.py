import ast
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrustRemoteCodeVisitor(ast.NodeVisitor):
    def __init__(self, target_call: str):
        self.target_call = target_call
        self.matching_calls = []

    def visit_Call(self, node):
        call_name = self._call_name(node.func)
        if call_name == self.target_call:
            self.matching_calls.append(node)
        self.generic_visit(node)

    def _call_name(self, node):
        if isinstance(node, ast.Attribute):
            prefix = self._call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Name):
            return node.id
        return None


def calls_with_trust_remote_code(relative_path: str, target_call: str):
    source = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    visitor = TrustRemoteCodeVisitor(target_call)
    visitor.visit(tree)
    return [
        call
        for call in visitor.matching_calls
        if any(
            keyword.arg == "trust_remote_code"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in call.keywords
        )
    ]


class EsmTrustRemoteCodeStaticTest(unittest.TestCase):
    def test_s1_protein_loads_esm_model_with_trust_remote_code(self):
        calls = calls_with_trust_remote_code(
            "qwenvl/modeling_s1_protein.py",
            "AutoModel.from_pretrained",
        )
        self.assertGreaterEqual(len(calls), 1)

    def test_protein_collator_loads_esm_tokenizer_with_trust_remote_code(self):
        calls = calls_with_trust_remote_code(
            "qwenvl/data/data_processor.py",
            "transformers.AutoTokenizer.from_pretrained",
        )
        self.assertGreaterEqual(len(calls), 1)

    def test_protein_inference_loads_esm_tokenizer_with_trust_remote_code(self):
        calls = calls_with_trust_remote_code(
            "infer_s1_protein_checkpoint.py",
            "AutoTokenizer.from_pretrained",
        )
        self.assertGreaterEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
