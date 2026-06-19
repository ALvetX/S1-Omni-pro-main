# S1Protein 8/16-Layer Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Configure the S1Protein protein branch to use 8 ESM unfrozen layers and 16 cross-attention fusion layers.

**Architecture:** Keep the change narrow and configuration-driven. Update the training script to pass 8/16-layer values explicitly, align the protein model defaults and config fallback to ESM 8 plus cross-attention 16, and sync the CLI dataclass defaults so future runs do not drift. Preserve checkpoint compatibility by continuing to honor saved `s1_protein_config.json` values on load.

**Tech Stack:** Bash training script, Python dataclasses, PyTorch model definition, existing static pytest coverage.

---

### Task 1: Align protein defaults and CLI parameters

**Files:**
- Modify: `qwen-vl-finetune/qwenvl/modeling_s1_protein.py:97-109, 437-446`
- Modify: `qwen-vl-finetune/qwenvl/train/argument.py:11-23`

- [ ] **Step 1: Update the model constructor and load fallback defaults**

```python
# modeling_s1_protein.py
esm_fusion_num_layers: int = 16,
esm_unfreeze_last_n_layers: int = 8,
...
"esm_fusion_num_layers": saved_config.get(
    "esm_fusion_num_layers",
    16 if saved_config else 16,
),
"esm_unfreeze_last_n_layers": saved_config.get("esm_unfreeze_last_n_layers", 8),
```

- [ ] **Step 2: Update the CLI dataclass defaults to match**

```python
# argument.py
esm_fusion_num_layers: int = field(default=16)
esm_unfreeze_last_n_layers: int = field(default=8)
```

- [ ] **Step 3: Verify the parameter names remain unchanged**

Run:
```bash
rg -n "esm_fusion_num_layers|esm_unfreeze_last_n_layers" qwen-vl-finetune/qwenvl/modeling_s1_protein.py qwen-vl-finetune/qwenvl/train/argument.py
```
Expected: all existing references still use the same parameter names, only defaults changed.

### Task 2: Update the protein training script

**Files:**
- Modify: `qwen-vl-finetune/scripts/s1_protein.sh:86-95`

- [ ] **Step 1: Change the explicit runtime arguments to ESM 8 and cross-attention 16**

```bash
--esm_num_attention_heads 8 \
--esm_fusion_num_layers 16 \
--esm_fusion_ffn_dim 2048 \
--esm_unfreeze_last_n_layers 8 \
```

- [ ] **Step 2: Verify the script still passes the expected flags**

Run:
```bash
sed -n '86,95p' qwen-vl-finetune/scripts/s1_protein.sh
```
Expected: `--esm_fusion_num_layers 16` and `--esm_unfreeze_last_n_layers 8` are present.

### Task 3: Add a static regression check

**Files:**
- Modify: `qwen-vl-finetune/tests/test_s1_protein_esm_source_resolution.py`

- [ ] **Step 1: Add a config-default regression assertion**

```python
from pathlib import Path


def test_s1_protein_defaults_use_8_esm_layers_and_16_cross_attention_layers():
    text = Path("qwen-vl-finetune/qwenvl/modeling_s1_protein.py").read_text()
    assert 'esm_fusion_num_layers: int = 16' in text
    assert 'esm_unfreeze_last_n_layers: int = 8' in text
```

- [ ] **Step 2: Run the targeted test file**

Run:
```bash
pytest qwen-vl-finetune/tests/test_s1_protein_esm_source_resolution.py -q
```
Expected: pass.

- [ ] **Step 3: Run the repository's existing static protein tests**

Run:
```bash
pytest qwen-vl-finetune/tests/test_s1_protein_esm_source_resolution.py qwen-vl-finetune/tests/test_s1_protein_queue_static.py -q
```
Expected: pass.
