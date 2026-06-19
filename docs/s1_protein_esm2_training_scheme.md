# S1Protein ESM2-Qwen Cross-Attention 训练方案

本文档记录当前 `S1Protein` 氨基酸结合位点预测方案的完整训练设计、数据流、模型结构、loss、训练参数、推理评估流程和已知注意事项。当前实现面向 residue-level 多标签二分类任务：输入问题文本中包含一条 `<PROT>...</PROT>` 氨基酸序列，模型输出与序列等长的二值标签或概率，表示每个 residue 是否为目标结合位点。

## 1. 任务定义

### 1.1 输入

每条样本包含一段自然语言问题，问题中必须有且仅有一个大写 protein 标签：

```text
... <PROT>ACDEFGHIK...</PROT> ...
```

当前实现只支持：

- `<PROT>...</PROT>`

当前实现不兼容：

- `<PORT>...</PORT>`
- `<prot>...</prot>`
- 多个 `<PROT>...</PROT>` span
- 空 protein span

`<PROT>...</PROT>` 内部允许存在换行或空白，数据处理阶段会删除所有空白后得到连续氨基酸字符串。例如：

```text
<PROT>A C
D</PROT>
```

会被解析为：

```text
ACD
```

### 1.2 输出

输出为与 protein 序列等长的 residue-level 标签：

```text
label[i] = 1 表示第 i 个 residue 是结合位点
label[i] = 0 表示第 i 个 residue 不是结合位点
```

训练时使用 0-based 张量对齐；推理输出中的 `positive_indices` 为 1-based，方便和原始 `answer` 字段对齐。

### 1.3 类别分布

该任务极度类别不平衡，正样本 residue 占比很低。此前观察中，负样本约占 95% 以上，甚至训练集正样本率只有约 1% 到 2%。因此：

- accuracy 不是主要指标；
- 固定阈值下的 F1 容易受阈值影响；
- 更推荐关注 AUPR、Top-K recall、Recall@num_true 和 positive rank。

## 2. 总体架构

当前路线是 ESM2 + Qwen3-VL 的 cross-attention residue 分类模型。

```text
原始 question
  |
  |-- 抽取 <PROT>...</PROT> 中的连续氨基酸序列
  |       |
  |       v
  |     ESM2 tokenizer
  |       |
  |       v
  |     frozen ESM2
  |       |
  |       v
  |     residue hidden: [B, L, E]  ------+
  |                                      |
  |                                      | as Query
  v                                      v
Qwen 输入文本: 原 span 替换为 <PROT></PROT>   Cross-Attention
  |                                      ^
  v                                      | as Key / Value
Qwen tokenizer                           |
  |                                      |
  v                                      |
Qwen3-VL language_model forward          |
  |                                      |
  v                                      |
token hidden: [B, T, H] -----------------+
  |
  v
融合后 residue hidden: [B, L, fusion_dim]
  |
  v
MLP protein_head
  |
  v
logits: [B, L, 1]
  |
  v
weighted BCEWithLogitsLoss
```

核心思想：

- ESM2 专门负责 residue-level 蛋白序列表征；
- Qwen3-VL 负责理解问题文本、任务类型、SMILES 或其他上下文；
- ESM2 residue hidden 作为 Query，主动从 Qwen token hidden 中读取任务上下文；
- 最终对每个 residue 输出一个 logit，用 sigmoid 得到正类概率。

## 3. 数据处理流程

相关实现：

- `qwen-vl-finetune/qwenvl/data/data_processor.py`
- `_extract_protein_sequence_and_qwen_text`
- `_normalize_protein_messages_and_sequence`
- `ProteinSupervisedDataset`
- `ProteinDataCollator`

### 3.1 protein span 抽取

对于每条样本：

1. 从用户问题文本中查找 `<PROT>...</PROT>`。
2. 要求匹配数量必须等于 1。
3. 抽取标签内部内容。
4. 删除所有空白字符。
5. 转成大写。
6. 返回两部分：

```python
qwen_text, protein_sequence
```

示例：

```text
原始文本:
Predict iron binding site for <PROT>A C D</PROT> with <SMILES>[Fe+2]</SMILES>.

protein_sequence:
ACD

qwen_text:
Predict iron binding site for <PROT></PROT> with <SMILES>[Fe+2]</SMILES>.
```

注意：ESM2 接收到的是连续字符串 `ACD`，不会被插入空格。

### 3.2 Qwen 输入

当前 `use_esm2=True` 时，Qwen 不再直接看到完整氨基酸序列，而是看到空 protein tag：

```text
<PROT></PROT>
```

这样做的目的：

- 避免 Qwen tokenizer 对长 protein 序列产生大量文本 token；
- 避免 Qwen token 和 residue 之间复杂、脆弱的对齐；
- 让 Qwen 专注于问题语义、任务类型、SMILES、约束信息和上下文；
- residue-level 信息由 ESM2 提供。

随后使用 Qwen processor/tokenizer 的 `apply_chat_template` 得到：

```python
input_ids: [1, T]
attention_mask: [1, T]
```

batch collator 中 padding 后得到：

```python
input_ids: [B, T_max]
attention_mask: [B, T_max]
```

### 3.3 ESM2 输入

`ProteinDataCollator` 初始化 ESM tokenizer：

```python
transformers.AutoTokenizer.from_pretrained(esm_model_name)
```

对 batch 内的连续氨基酸序列做 padding：

```python
esm_inputs = esm_tokenizer(
    protein_sequences,
    padding=True,
    return_tensors="pt",
)
```

得到：

```python
esm_input_ids: [B, S_max]
esm_attention_mask: [B, S_max]
```

其中 `S = L + 2`，通常包含：

- 一个 CLS/BOS special token；
- `L` 个 residue token；
- 一个 EOS special token。

模型内部会取：

```python
esm_hidden[:, 1 : 1 + L]
```

作为 residue hidden。

### 3.4 label 处理

标签来源：

- 优先读取 `label`
- 若没有，则读取 `ground_truth`

支持格式：

- 纯 0/1 字符串，例如 `"001000100"`
- 0/1 list，例如 `[0, 0, 1, 0]`

要求：

```python
len(protein_labels) == len(protein_sequence)
```

否则直接抛错，避免训练时 residue 和 label 错位。

batch padding 后：

```python
protein_labels: [B, L_max]
protein_label_mask: [B, L_max]
```

`protein_label_mask` 中真实 residue 位置为 `True`，padding 位置为 `False`。

### 3.5 Trainer 字段裁剪注意事项

Hugging Face Trainer 会根据 `model.forward(...)` 的参数签名移除“不需要”的字段。因此当前 `S1Protein.forward` 中保留了：

```python
protein_sequence=None
```

这个字段主要用于保证 Trainer 不会在进入 collator 前删掉 `protein_sequence`。进入 forward 后会直接 `del protein_sequence`，不会传给 Qwen，也不参与计算。

## 4. 模型结构

相关实现：

- `qwen-vl-finetune/qwenvl/modeling_s1_protein.py`
- `S1Protein`

### 4.1 Backbone

当前训练脚本使用：

```bash
--model_name_or_path /data/group/wenge/xlf_model/S1-VL-32B-RL
```

模型类：

```python
Qwen3VLForConditionalGeneration
```

在 `S1Protein` 中实际 forward 调用的是：

```python
self.backbone.model(...)
```

并取：

```python
hidden_states = outputs[0]
```

形状：

```python
qwen_hidden: [B, T, H]
```

其中 `H = backbone.config.text_config.hidden_size`。

### 4.2 ESM2 encoder

默认配置：

```python
esm_model_name = "facebook/esm2_t33_650M_UR50D"
```

当前训练脚本使用本地路径：

```bash
--esm_model_name /data/home/zdhs0092/Models/esm2_t33_650M_UR50D
```

ESM2 加载方式：

```python
AutoModel.from_pretrained(esm_model_name)
```

默认全冻结：

```python
esm_model.requires_grad_(False)
esm_model.eval()
```

训练时 ESM2 前向路径已不再包在 `torch.no_grad()` 中。冻结子模块的参数 `requires_grad=False`，
反向不会流过；解冻的子模块（见 4.2.1）会正常接收梯度。

### 4.2.1 ESM2 末几层解冻策略

可通过以下参数控制：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--esm_unfreeze_last_n_layers` | `0`（保持全冻结） | 解冻 ESM2 倒数 N 个 transformer layer 的全部参数 |
| `--esm_unfreeze_final_layer_norm` | `True` | 是否同时解冻 `encoder.emb_layer_norm_after` / `encoder.LayerNorm` |
| `--esm_unfreeze_pooler` | `False` | 是否解冻 `esm_model.pooler`（仅当 ESM2 forward 使用 pooler 时生效） |
| `--esm_lr_multiplier` | `0.1` | 被解冻的 ESM2 参数相对 base LR 的倍率 |

实现要点：

- 解冻粒度为**整层解冻**：被选中的 `encoder.layer[i]` 内部 `self_attn` / FFN / LayerNorm 全部 `requires_grad=True`。
- ESM2 始终保持 `eval()`，即 ESM2 内部 Dropout 不会激活。
- `S1Protein.unfreeze_llm_backbone()` 在 `unfreeze Qwen` 之后会重新调用 `_apply_esm_unfreeze_config()`，确保 ESM2 解冻策略不会被覆盖。
- ESM2 forward 不再 `torch.no_grad`：解冻部分走 autograd，冻结部分 `requires_grad=False` 自然不参与梯度。
- 优化器分组：被解冻的 ESM2 参数走 `base_lr × esm_lr_multiplier`，其他可训练参数走 `base_lr`。
- 训练 checkpoint 会把完整 ESM2 权重保存到 `<output_dir>/esm2/model.safetensors` 和 `esm2_config.json`，方便从 checkpoint 恢复时继续训练。
- 从 checkpoint 加载时，`from_pretrained` 会优先尝试读取 bundled ESM2 权重；缺失时回退到 `esm_model_name` 指定的外部预训练权重。

ESM2 输出：

```python
esm_outputs.last_hidden_state: [B, S, E]
```

去掉 special tokens 后：

```python
esm_hidden: [B, L, E]
```

对于 `esm2_t33_650M_UR50D`，通常：

```python
E = 1280
```

### 4.3 ESM2 超长序列处理

ESM2 有位置上限：

```python
max_position_embeddings = 1026
```

由于输入包含 CLS/EOS，单次 ESM2 forward 最多可处理：

```python
max_residues_per_chunk = 1026 - 2 = 1024
```

如果 `esm_input_ids.size(1) <= 1026`：

- 直接整条序列前向。

如果 `esm_input_ids.size(1) > 1026`：

- 模型内部自动做滑窗式 chunk 编码；
- 每个 chunk 重新加 CLS/EOS；
- 每个 chunk 长度不超过 ESM2 位置上限；
- 最后一个窗口会向前回退，避免只有少量 residue 的尾块；
- 重叠 residue 的 hidden 求平均；
- 最后拼回原始长度：

```python
esm_hidden: [B, L, E]
residue_mask: [B, L]
```

示例：

```text
L = 1028
max_residues_per_chunk = 1024

窗口 1: residue 0    ~ 1023
窗口 2: residue 4    ~ 1027
重叠: residue 4      ~ 1023
融合: 重叠位置 hidden 取平均
```

这样可以解决 `1030 > 1026` 一类错误，同时尽量避免尾部 residue 缺少上下文。

### 4.4 Cross-Attention Fusion

核心融合模块由输入投影和可堆叠的 cross-attention block 组成。当前默认堆叠 2 层，每层包含：

- pre-LN cross-attention；
- residual connection；
- pre-LN FFN；
- residual connection；
- residue mask。

```python
esm_query_projection = nn.Linear(esm_hidden_size, esm_fusion_dim)
qwen_key_projection = nn.Linear(qwen_hidden_size, esm_fusion_dim)
qwen_value_projection = nn.Linear(qwen_hidden_size, esm_fusion_dim)

fusion_layers = nn.ModuleList(
    [
        ProteinCrossAttentionBlock(
            fusion_dim=esm_fusion_dim,
            num_attention_heads=esm_num_attention_heads,
            ffn_dim=esm_fusion_ffn_dim,
            dropout=head_dropout,
        )
        for _ in range(esm_fusion_num_layers)
    ]
)
fusion_norm = nn.LayerNorm(esm_fusion_dim)
```

当前训练配置：

```bash
--esm_fusion_dim 512
--esm_num_attention_heads 8
--esm_fusion_num_layers 2
--esm_fusion_ffn_dim 2048
```

张量形状：

```python
esm_hidden:  [B, L, E]
qwen_hidden: [B, T, H]

query = Linear(E -> 512)(esm_hidden)     # [B, L, 512]
key   = Linear(H -> 512)(qwen_hidden)    # [B, T, 512]
value = Linear(H -> 512)(qwen_hidden)    # [B, T, 512]
```

每一层 cross-attention block：

```python
attn_input = attention_norm(residue_hidden)
attn_output, _ = cross_attention(
    query=attn_input,
    key=key,
    value=value,
    key_padding_mask=~qwen_attention_mask.bool(),
    need_weights=False,
)
residue_hidden = residue_hidden + attention_dropout(attn_output)

ffn_input = ffn_norm(residue_hidden)
residue_hidden = residue_hidden + ffn_dropout(ffn(ffn_input))
residue_hidden = residue_hidden * residue_mask[..., None]
```

多层堆叠：

```python
fused_hidden = query * residue_mask[..., None]
for fusion_layer in fusion_layers:
    fused_hidden = fusion_layer(
        residue_hidden=fused_hidden,
        qwen_key=key,
        qwen_value=value,
        key_padding_mask=key_padding_mask,
        residue_mask=residue_mask,
    )
fused_hidden = fusion_norm(fused_hidden)
fused_hidden = fused_hidden * residue_mask[..., None]
```

输出：

```python
fused_hidden: [B, L, 512]
```

### 4.5 Protein Head

当前 head 是 MLP，不使用 residue embedding，也不使用 1D conv：

```python
protein_head = nn.Sequential(
    nn.LayerNorm(head_input_size),
    nn.Linear(head_input_size, head_hidden_size),
    nn.GELU(),
    nn.Dropout(head_dropout),
    nn.Linear(head_hidden_size, 1),
)
```

当 `use_esm2=True`：

```python
head_input_size = esm_fusion_dim = 512
```

默认：

```python
head_hidden_size = max(256, hidden_size // 4)
head_dropout = 0.1
output_size = 1
```

输出：

```python
logits: [B, L, 1]
probabilities = sigmoid(logits)
pred_bits = probabilities >= threshold
```

注意：训练时 threshold 不参与 loss，只影响二值化输出。

## 5. Loss 设计

当前支持两种 masked residue loss：

相关函数：

```python
_weighted_residue_bce
_asymmetric_residue_loss
_protein_residue_loss
```

### 5.1 Masked weighted BCE

```python
per_residue_loss = binary_cross_entropy_with_logits(
    logits,
    targets,
    pos_weight=positive_loss_weight,
    reduction="none",
)

loss = sum(per_residue_loss * valid_mask) / sum(valid_mask)
```

其中：

- `valid_mask = protein_label_mask & residue_mask`
- padding residue 不参与 loss；
- ESM2 padding 不参与 loss；
- 正样本 loss 被 `positive_loss_weight` 放大，用于缓解类别不平衡。

### 5.2 Masked ASL

ASL 通过对正负样本使用不同 focal gamma，更强地抑制大量 easy negative residue。

```python
prob = sigmoid(logits)
pos_prob = clamp(prob)
neg_prob = clamp(1 - prob + asl_clip)

pos_loss = target * log(pos_prob) * positive_loss_weight
neg_loss = (1 - target) * log(neg_prob)
base_loss = -(pos_loss + neg_loss)

pt = target * pos_prob + (1 - target) * neg_prob
gamma = target * asl_gamma_pos + (1 - target) * asl_gamma_neg
per_residue_loss = base_loss * (1 - pt) ** gamma

loss = sum(per_residue_loss * valid_mask) / sum(valid_mask)
```

当前 ASL 起步配置：

```bash
--protein_loss_type asl
--positive_loss_weight 10.0
--asl_gamma_pos 0.0
--asl_gamma_neg 4.0
--asl_clip 0.05
```

说明：

- `asl_gamma_pos=0.0` 保留正样本梯度；
- `asl_gamma_neg=4.0` 大幅降低 easy negative 的影响；
- `asl_clip=0.05` 对负样本概率做 clipping，进一步减轻极易负样本主导 loss；
- ASL 已经会聚焦 hard examples，因此 `positive_loss_weight` 建议比纯 BCE 小，先从 10 左右试。

为什么不用 softmax dim=2：

- 每个 residue 是独立二分类；
- 1-logit sigmoid 更直接；
- BCE/ASL 都可以直接作用在每个 residue 的 1-logit 上；
- 推理时概率解释为每个 residue 独立为正类的概率。

## 6. 训练参数

当前训练入口：

```bash
qwen-vl-finetune/scripts/s1_protein.sh
```

核心命令：

```bash
torchrun --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29503 \
  qwenvl/train/train_qwen.py \
  --deepspeed /data/home/zdhs0092/Code/S1-Omni-pro/qwen-vl-finetune/scripts/zero3.json \
  --model_architecture s1-protein \
  --model_name_or_path /data/group/wenge/xlf_model/S1-VL-32B-RL \
  --use_esm2 True \
  --esm_model_name /data/home/zdhs0092/Models/esm2_t33_650M_UR50D \
  --esm_fusion_dim 512 \
  --esm_num_attention_heads 8 \
  --esm_fusion_num_layers 2 \
  --esm_fusion_ffn_dim 2048 \
  --positive_loss_weight 10.0 \
  --protein_loss_type asl \
  --asl_gamma_pos 0.0 \
  --asl_gamma_neg 4.0 \
  --asl_clip 0.05 \
  --annotation_path /data/home/zdhs0092/Code/S1-Omni-pro/all_data/protein_pre_data/iron_binding_site/train/protein_site_prediction-interaction_site-iron_binding_site.jsonl \
  --bf16 \
  --output_dir /data/home/zdhs0092/Code/S1-Omni-pro/output/s1_protein_iron_binding_site_esm2 \
  --num_train_epochs 3 \
  --save_only_model True \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --warmup_ratio 0.1 \
  --max_grad_norm 1 \
  --lr_scheduler_type cosine \
  --logging_steps 1 \
  --save_strategy epoch \
  --save_total_limit 5 \
  --eval_strategy no \
  --model_max_length 4096 \
  --gradient_checkpointing True \
  --dataloader_num_workers 16 \
  --report_to "swanlab"
```

### 6.1 有效 batch size

当前配置：

```text
GPU 数量: 8
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
```

因此理论 global batch size：

```text
8 * 8 * 2 = 128 samples / optimizer step
```

如果 DeepSpeed 配置中也设置了 gradient accumulation，需要确认它和 Trainer 参数一致。日志中如果出现：

```text
Gradient accumulation steps mismatch
```

则实际值以 DeepSpeed 使用的配置为准。

### 6.2 学习率

当前可训练参数学习率分组：

| 分组 | LR | 说明 |
|---|---|---|
| Qwen language_model + lm_head | `1e-5` | 基础学习率 |
| ESM2 解冻层（末 N 层 + final LayerNorm） | `1e-6` | `base_lr × 0.1` |
| ESM query / Qwen key / Qwen value projection | `1e-5` | 新增 fusion 模块 |
| Cross-attention fusion layers | `1e-5` | 新增 fusion 模块 |
| Fusion LayerNorm | `1e-5` | 新增 fusion 模块 |
| Protein MLP head | `1e-5` | residue-level classifier |

未解冻的 ESM2 参数不参与训练，因此不进入优化器。

### 6.3 可训练与冻结模块

`set_model(...)` 中对 `S1Protein` 调用：

```python
model.unfreeze_llm_backbone()
for p in model.protein_head.parameters():
    p.requires_grad = True
```

`unfreeze_llm_backbone()` 内部会在 Qwen 解冻之后重新调用 `_apply_esm_unfreeze_config()`，因此当
`esm_unfreeze_last_n_layers > 0` 时 ESM2 末 N 层会被正确置为可训练。

实际训练状态（默认配置，`--esm_unfreeze_last_n_layers 4`）：

| 模块 | 状态 | 说明 |
|---|---|---|
| Qwen language_model | 训练 | 从训练开始解冻 |
| Qwen lm_head | 训练 | 若存在则解冻 |
| Qwen visual tower | 冻结 | 当前任务无图像输入 |
| ESM2 末 N 层 | 训练 | `esm_unfreeze_last_n_layers=4` 时解冻最后 4 个 encoder.layer |
| ESM2 `emb_layer_norm_after` | 训练 | 默认解冻，避免统计量漂移 |
| ESM2 其余 layer | 冻结 | `requires_grad=False` |
| ESM2 pooler | 冻结 | 默认不解冻 |
| ESM query projection | 训练 | 新增 fusion 模块 |
| Qwen key/value projection | 训练 | 新增 fusion 模块 |
| Cross-attention | 训练 | 新增 fusion 模块 |
| Fusion LayerNorm | 训练 | 新增 fusion 模块 |
| Protein MLP head | 训练 | residue-level classifier |

ESM2 解冻层使用 `lr = base_lr × esm_lr_multiplier = 1e-5 × 0.1 = 1e-6`，由
`S1OmniTrainer.create_optimizer` 显式分组。其余可训练参数共用 `base_lr = 1e-5`。

### 6.4 精度和显存策略

当前使用：

```bash
--bf16
--gradient_checkpointing True
--deepspeed zero3.json
```

设计目的：

- bf16 降低显存并保持数值稳定性；
- gradient checkpointing 降低 Qwen 反向显存；
- ZeRO-3 切分大模型参数、梯度和优化器状态；
- ESM2 冻结且 `no_grad()`，不会产生反向图。

## 7. 保存与加载

训练保存目录示例：

```text
output/s1_protein_iron_binding_site_esm2
```

`S1Protein.save_pretrained(...)` 会保存：

- Qwen backbone 权重；
- `protein_head.*`
- `esm_query_projection.*`
- `qwen_key_projection.*`
- `qwen_value_projection.*`
- `fusion_layers.*`
- `fusion_norm.*`
- tokenizer / processor；
- `s1_protein_config.json`

不会保存：

- ESM2 权重

原因是 ESM2 使用外部 pretrained checkpoint，推理时根据 `s1_protein_config.json` 中的 `esm_model_name` 重新加载。

`s1_protein_config.json` 记录：

```json
{
  "use_esm2": true,
  "esm_model_name": "...",
  "esm_fusion_dim": 512,
  "esm_num_attention_heads": 8,
  "esm_fusion_num_layers": 2,
  "esm_fusion_ffn_dim": 2048,
  "positive_loss_weight": 10.0,
  "protein_loss_type": "asl",
  "asl_gamma_pos": 0.0,
  "asl_gamma_neg": 4.0,
  "asl_clip": 0.05,
  "asl_eps": 1e-8
}
```

## 8. 推理流程

相关脚本：

```bash
qwen-vl-finetune/infer_s1_protein_checkpoint.py
```

默认参数：

```bash
--checkpoint_dir /data/home/zdhs0092/Code/S1-Omni-pro/output/s1_protein_iron_binding_site_esm2
--batch_file /data/home/zdhs0092/Code/S1-Omni-pro/all_data/protein_pre_data/iron_binding_site/test/protein_site_prediction-interaction_site-iron_binding_site.jsonl
--output_file /data/home/zdhs0092/Code/S1-Omni-pro/output_protein_iron_binding_site_mlp_test_esm2.jsonl
--device cuda:0
--dtype bf16
--threshold 0.5
```

推理时复用训练阶段相同的抽取逻辑：

1. 从原始 question 中抽取 `<PROT>...</PROT>`；
2. Qwen 文本替换为 `<PROT></PROT>`；
3. ESM2 输入为连续氨基酸序列；
4. 模型输出：

```python
probabilities: [B, L, 1]
```

5. 按原始 protein length 截断；
6. 根据 threshold 生成 bit string。

输出 jsonl 每行包含：

```json
{
  "positive_indices": [12, 34],
  "answer": "...",
  "bit_string": "000100...",
  "question": "...<PROT>...</PROT>...",
  "threshold": 0.5,
  "probabilities": [0.01, 0.03, 0.72]
}
```

注意：当前推理脚本中 batch 文件模式暂时限制：

```python
questions = questions[:100]
```

如果需要完整测试集推理，需要移除或修改该限制。

## 9. 评估流程

相关脚本：

```bash
evaluation_s1_protein.py
```

默认读取：

```python
jsonl_file = "/data/home/zdhs0092/Code/S1-Omni-pro/output_protein_iron_binding_site_mlp_test_esm2.jsonl"
```

默认评估：

```python
pred_source="probabilities"
threshold=0.9
use_record_threshold=False
```

### 9.1 label 构造

评估脚本从输出 jsonl 的 `answer` 字段解析真实正类位点：

- `answer` 是 1-based residue index；
- 内部构造成 0-based 的 `y_true`；
- 长度优先从 `question` 中 `<PROT>...</PROT>` 的真实序列长度解析。

### 9.2 指标

当前评估输出包括：

Global / Micro：

- precision
- recall
- f1
- mcc
- auroc
- aupr

Macro per sample：

- precision
- recall
- f1
- mcc
- auroc
- aupr

Top-K / Macro per sample：

- precision@num_true
- recall@num_true
- precision@5
- recall@5
- precision@10
- recall@10
- precision@20
- recall@20

Threshold sweep / Global：

- best_f1_threshold
- best_f1
- best_f1_precision
- best_f1_recall

### 9.3 推荐主指标

由于类别极不平衡，建议优先看：

1. Global AUPR
2. Macro AUPR
3. Recall@num_true
4. Recall@5 / Recall@10 / Recall@20
5. positive rank 中位数或分位数

固定 threshold 下的 precision/recall/F1 可用于最终部署阈值选择，但不建议作为唯一选模标准。

## 10. 常见问题与排查

### 10.1 `KeyError: protein_length`

原因：

- 旧 collator 依赖 `protein_length` 字段；
- Trainer 可能根据 forward 签名裁剪字段；
- 当前 ESM2 路径不再需要显式 `protein_length`。

当前处理：

- batch 内不再传 `protein_lengths`；
- 长度由 `protein_label_mask` 和 ESM attention mask 表达；
- 推理阶段用抽取出的 `protein_sequence` 长度截断输出。

### 10.2 `ESM2 input length exceeds max_position_embeddings`

原因：

- ESM2 位置上限包含 special tokens；
- `esm2_t33_650M_UR50D` 的 `max_position_embeddings=1026`；
- residue 长度超过 1024 时直接 forward 会报错。

当前处理：

- 模型内部自动滑窗 chunk；
- chunk 后拼回原始 residue hidden；
- 不截断 label；
- 不跳过样本。

### 10.3 `pooler.dense.weight/bias MISSING`

日志中可能出现：

```text
EsmModel LOAD REPORT
pooler.dense.weight MISSING
pooler.dense.bias MISSING
```

通常可以忽略。当前模型只使用：

```python
esm_outputs.last_hidden_state
```

不依赖 pooler 输出。

### 10.4 训练输出全 0

可能原因：

- 正样本极少，模型倾向负类；
- `positive_loss_weight` 仍不足；
- 学习率太小或训练 epoch 不够；
- threshold 过高；
- 训练集标注噪声或任务上下文不充分；
- Qwen/ESM fusion 尚未学会有效对齐。

排查建议：

1. 先做 32 条样本 overfit sanity check；
2. 观察 loss 是否稳定下降；
3. 用 probabilities 评估 AUPR 和 Top-K，不要只看 `threshold=0.5` 的 bit string；
4. sweep threshold；
5. 检查真实 label 和 protein length 是否严格一致；
6. 检查推理输出 probability 是否有区分度。

## 11. 推荐实验流程

### 11.1 数据检查

训练前建议检查：

- 每条样本有且仅有一个 `<PROT>...</PROT>`；
- 没有 `<PORT>`；
- `<PROT>` 内序列非空；
- 删除空白后的序列长度等于 label 长度；
- label 只包含 0/1；
- 正样本率统计正常。

### 11.2 Overfit sanity check

建议先构造 32 条训练样本：

- 训练若干 epoch；
- 观察 loss 是否明显下降；
- 评估训练集输出 probabilities；
- 检查 Recall@num_true 是否明显高于随机；
- 检查正样本 residue 的概率排名是否前移。

如果 32 条都无法 overfit，优先排查：

- label 对齐；
- ESM2 输入序列；
- Qwen 文本是否保留了任务信息；
- loss mask；
- trainable params；
- LR 和梯度。

### 11.3 完整训练

当前主配置：

```text
learning_rate: 1e-5
warmup_ratio: 0.1
lr_scheduler_type: cosine
max_grad_norm: 1.0
positive_loss_weight: 50.0
bf16: true
ZeRO-3: true
gradient_checkpointing: true
```

初始 epoch：

```text
num_train_epochs: 3
```

如果 loss 仍在下降且验证 AUPR 没有过拟合，可以尝试提高到：

```text
num_train_epochs: 5
```

### 11.4 复评

完整训练后建议分别评估：

- train 输出 jsonl；
- eval/test 输出 jsonl；

重点比较：

- AUPR 是否提升；
- Recall@num_true 是否提升；
- Recall@5/10/20 是否提升；
- positive rank 中位数是否下降；
- threshold sweep 的 best F1 threshold 是否合理；
- 预测正样本数量是否明显偏全 0 或全 1。

## 12. 当前方案的优点

1. residue-level 对齐更可靠  
   不再依赖 Qwen tokenizer 对蛋白序列的 tokenization 对齐。

2. 利用了蛋白预训练知识  
   ESM2 对氨基酸序列和局部/远程 residue 依赖更有先验。

3. Qwen 仍能理解任务上下文  
   问题、SMILES、结合类型等信息仍由 Qwen 编码。

4. 支持长序列  
   超过 ESM2 单窗限制时自动滑窗。

5. loss 适配不平衡  
   weighted BCE 明确提高正样本梯度权重。

6. 训练策略相对简单  
   所有可训练参数统一 LR，便于先建立稳定 baseline。

## 13. 当前方案的局限和后续方向

### 13.1 ESM2 在线前向速度较慢

当前 ESM2 是在线 frozen forward。虽然没有反向图，但每一步仍需要运行 ESM2。

后续可考虑：

- 离线缓存 ESM2 residue embeddings；
- 按 protein sequence hash 做 cache；
- 对训练集预计算 `.pt` 或 memmap 特征；
- 对不同任务共享同一份 ESM2 cache。

### 13.1.1 ESM2 末几层解冻（新增）

`--esm_unfreeze_last_n_layers` > 0 时：

- ESM2 末 N 个 transformer layer 转为可训练，使用更小的学习率（默认 0.1× base_lr）。
- ESM2 forward 不再 `no_grad`，解冻层正常反向，激活显存/算力开销会增加。
- 训练 checkpoint 会包含完整 ESM2 权重（`output_dir/esm2/model.safetensors`），推理时优先使用 bundled 权重。
- 解冻层数过多（>= 12）会让 ESM2 的预训练知识被快速冲刷，建议从 2~4 层起步。

解冻效果高度依赖任务：

- 任务上下文丰富（SMILES + 离子类型 + 任务描述）时解冻往往带来提升。
- 任务上下文稀薄时 ESM2 学不到"如何配合 Qwen 任务语义"，收益可能不明显。

### 13.2 Fusion 深度仍然较浅

当前 fusion 已经从单层 cross-attention 升级为 2 层 cross-attention block。相比单层结构，它能让 residue 表征多次读取 Qwen 任务上下文，并通过 FFN 做非线性重整。但整体深度仍然较浅，属于稳健 baseline。

后续可尝试：

- 3 到 4 层 lightweight cross-attention block；
- 加 gated fusion；
- 加 task-conditioned bias；
- 加 residue pair/relative position encoding。

### 13.3 Qwen 看到空 protein tag

当前设计让 Qwen 不直接看序列，这减少了 token 对齐问题，但也意味着 Qwen 无法直接基于序列文本推理。

可选改进：

- Qwen 输入保留 protein 长度、物种、结构信息等摘要；
- Qwen 输入保留 `<PROT len=1024></PROT>` 这类显式长度提示；
- 加入任务类型 token。

### 13.4 Loss 仍可能不够

weighted BCE 能缓解不平衡，但不一定最优。

可尝试：

- focal loss；
- BCE + ranking loss；
- AUPR surrogate loss；
- hard negative mining；
- 按样本正样本数动态调整 `pos_weight`；
- 对没有正样本或正样本极少的样本单独采样策略。

### 13.5 评估应更关注 rank

结合位点任务常常更接近“候选位点排序”问题，而不是固定阈值分类。

后续建议增加：

- positive rank median；
- positive rank mean；
- MRR；
- hit@K；
- per-protein AUPR 分布；
- 按序列长度分桶的指标。

## 14. 关键文件索引

训练脚本：

```text
qwen-vl-finetune/scripts/s1_protein.sh
```

训练入口：

```text
qwen-vl-finetune/qwenvl/train/train_qwen.py
```

训练参数：

```text
qwen-vl-finetune/qwenvl/train/argument.py
```

数据处理：

```text
qwen-vl-finetune/qwenvl/data/data_processor.py
```

模型：

```text
qwen-vl-finetune/qwenvl/modeling_s1_protein.py
```

推理：

```text
qwen-vl-finetune/infer_s1_protein_checkpoint.py
```

评估：

```text
evaluation_s1_protein.py
```

checkpoint 配置：

```text
output/s1_protein_iron_binding_site_esm2/s1_protein_config.json
```

## 15. 当前默认配置速查

| 项目 | 当前值 |
|---|---|
| 架构 | ESM2 Query + Qwen hidden Key/Value cross-attention |
| protein tag | 只支持 `<PROT>...</PROT>` |
| ESM2 | `/data/home/zdhs0092/Models/esm2_t33_650M_UR50D` |
| ESM2 状态 | 末 4 层 + final LayerNorm 可训练，其余冻结 |
| ESM2 解冻层 LR 倍率 | 0.1（实际 LR = 1e-6） |
| Qwen | `/data/group/wenge/xlf_model/S1-VL-32B-RL` |
| Qwen language_model | trainable |
| Qwen visual tower | frozen |
| fusion dim | 512 |
| attention heads | 8 |
| fusion layers | 2 |
| fusion FFN dim | 2048 |
| head | MLP |
| output | 1-logit sigmoid |
| loss | masked ASL |
| pos_weight | 10.0 |
| ASL gamma_pos/gamma_neg | 0.0 / 4.0 |
| ASL clip | 0.05 |
| LR | 1e-5 |
| LR 分组 | ESM2 解冻层 1e-6，其余 1e-5 |
| epoch | 3 |
| bf16 | true |
| ZeRO | stage 3 |
| gradient checkpointing | true |
| max grad norm | 1.0 |
| warmup ratio | 0.1 |
| scheduler | cosine |
| global batch size | 约 128 |
| 主评估指标 | AUPR、Top-K recall、Recall@num_true |
