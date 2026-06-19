# S1Protein ESM 8层与Cross-Attention 16层设计

## 目标

将 `S1-Omni-pro-main/qwen-vl-finetune` 中的 protein 分支调整为：

- ESM 后续解冻层数保持为 8
- cross-attention fusion 层数从 8 改为 16

仅修改 `S1-Omni-pro-main`，不触碰 `S1-Omni-pro-main_2`。

## 影响范围

1. `qwen-vl-finetune/scripts/s1_protein.sh`
2. `qwen-vl-finetune/qwenvl/modeling_s1_protein.py`
3. `qwen-vl-finetune/qwenvl/train/argument.py`

## 设计

### 训练脚本

`s1_protein.sh` 显式传参改为：

- `--esm_fusion_num_layers 16`
- `--esm_unfreeze_last_n_layers 8`

这样当前训练任务会明确使用 cross-attention 16 层、ESM 解冻 8 层配置，不依赖代码默认值。

### 模型默认值

`modeling_s1_protein.py` 中与 protein 模型相关的默认值同步调整：

- `esm_fusion_num_layers` 默认值改为 16
- `esm_unfreeze_last_n_layers` 默认值改为 8
- `from_pretrained` 在没有保存配置时的 fallback 也改为 ESM 解冻 8 层、cross-attention 16 层

这样新建 protein 模型实例时，默认结构与训练脚本一致。

### CLI 参数默认值

`train/argument.py` 中的 dataclass 默认值同步为 `esm_fusion_num_layers=16`、`esm_unfreeze_last_n_layers=8`，避免命令行参数未显式覆盖时出现“脚本、模型默认值、训练参数”三处不一致。

## 兼容性

- 既有 checkpoint 仍然优先读取 `s1_protein_config.json` 中保存的层数配置。
- 这次修改不会改变 `esm_num_attention_heads`，保持 8。
- 这次修改不会改动 `S1-Omni-pro-main_2` 的脚本或模型。

## 验证

修改后检查三处是否一致：

- 训练脚本传参
- 模型构造默认值
- 保存/回读配置的默认 fallback
