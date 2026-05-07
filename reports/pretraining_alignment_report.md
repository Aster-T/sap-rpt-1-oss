# 预训练代码与论文对齐审计报告

**审计对象**：`sap-rpt-1-oss`（即 ConTextTab） 仓库中的预训练相关代码
**对照论文**：`papers/ContextTab.pdf` —— *ConTextTab: A Semantics-Aware Tabular In-Context Learner*（NeurIPS 2025）
**审计日期**：2026-05-07

---

## 1. 审计范围

预训练相关源码：

- `sap_rpt_oss/pretrain_run.py` —— 训练主循环
- `sap_rpt_oss/configs.py` —— 训练超参与表过滤规则
- `sap_rpt_oss/constants.py` —— 模型尺寸、嵌入模型注册
- `sap_rpt_oss/data/ds.py` —— `RPTParquetDataset` / `RPTTableDataset`
- `sap_rpt_oss/data/rules.py` —— 列过滤、目标列候选选择
- `sap_rpt_oss/data/tokenizer.py` —— 文本/数值/日期编码
- `sap_rpt_oss/data/sentence_embedder.py` —— Sentence-Transformers 包装
- `sap_rpt_oss/model/torch_model.py` —— `RPT` 主干
- `sap_rpt_oss/model/embeddings.py` —— `CellEmbeddings`（多模态嵌入合成）
- `sap_rpt_oss/model/attention.py` —— `TwoDimensionalAttentionLayer`（行/列交替注意力）

论文相关章节：第 3 节（Method，含 Encoding / Backbone / Decoding / Alternative architectures）、第 4.1 节（Training and inference）。

---

## 2. 整体结论

代码总体高度还原了论文 Section 3 与 Section 4.1 中描述的预训练流程：表过滤规则、行/列采样、目标列采样的所有阈值、优化器配置、批量累积、二维注意力主干结构、L2 回归头、交叉熵分类头与可选的 clustering 头都能直接对应。

但仍存在 **若干明显偏离**：默认句向量模型不是论文使用的 `all-MiniLM-L6-v2`、数值列裁剪分位数与论文 Section 3.1 的描述不一致、以及最关键的 —— **代码并未实现论文声称的"层间权重共享（weight sharing）"作为默认训练配置**。如果以仓库中 `pretrain_run.py` 直接训练，得到的是"non-shared weights"消融配置（Table 2 中的 `non-shared weights` 行），而非论文主表所用的 base 配置。

详细比对见下文。

---

## 3. 数据预处理与表过滤

### 3.1 表级过滤规则

| 维度 | 论文 (Section 4.1) | 代码（`configs.py::TableRulesConfig` + `data/ds.py` + `data/rules.py`） | 对齐情况 |
| --- | --- | --- | --- |
| 最小行数 | 丢弃行数 < 150 的表 | `min_num_rows: 150` | ✅ 一致 |
| 数值列 NaN 阈值 | 排除 NaN 比例 > 50% 的数值候选目标 | `numeric_nan_ratio_threshold: 0.5` + `rules.get_target_candidates` | ✅ 一致 |
| 类别列唯一比 | 排除唯一值比例 > 20% 的非数值候选目标 | `categorical_unique_ratio_threshold: 0.2` | ✅ 一致 |
| 日期列作为目标 | 论文明确"excluding all date columns" | `rules._is_date_like_column` 在生成候选时跳过 | ✅ 一致 |
| 常数列 | 论文未明确提到 | `drop_constant_columns: True`（默认） | ➕ 合理扩展 |
| 最大列数 | **未提及具体上限**（T4 中位数 9 列） | `max_num_columns: 50` ⇒ 最多 49 个特征 | ⚠️ **代码额外限制**：T4 中虽然中位数仅 9 列，但仍存在大量更宽的表。这一上限在论文 Table 2 未被消融，可能影响"longer/wider tables"那条 future-work 讨论 |

### 3.2 行采样

论文："randomly select 1000 rows, then between 50 and 900 rows as query, and use the rest as context."

代码：
- `FinetuneConfig.stage1_max_num_rows = 1000` → `RPTTableDataset` 在 `_prepare_table` 中按此 `sample(n=max_num_rows)`
- `query_size_range: (50, 900)` → `_resolve_fit_rows` 中按 `rng.integers(min, max+1)` 随机抽 query 行数；其余作为 fit/context

✅ 完全一致。

### 3.3 目标列采样与回归/分类平衡

论文："we up-sample non-numeric columns to have roughly the same proportion of regression and classification tasks."

代码 `RPTParquetDataset._build_auto_target_specs`：
- 对每个 parquet 文件分别构造 regression_specs（数值候选）和 classification_specs（非数值候选）
- 当 `balance_classification_tasks=True` 且 `len(classification_specs) < len(regression_specs)` 时，从 `classification_specs` 有放回采样补齐到与回归数相同
- 最终 `np.shuffle` 全部 spec

✅ 思想对齐，但实现是"以表为单位"上采样分类，而非论文笼统所说的"非数值列上采样"。考虑到一个表只采样一个目标列，效果等价。

---

## 4. 多模态编码（Section 3.1）

### 4.1 文本嵌入模型

| 项 | 论文 | 代码 | 对齐情况 |
| --- | --- | --- | --- |
| 默认句向量模型 | `sentence-transformers/all-MiniLM-L6-v2` | `Tokenizer.sentence_embedding_model_name = "sentence-transformers/all-MiniLM-L12-v2"` | ❌ **不一致** |
| 嵌入维度 | 384 | 384（两者均为 384） | ✅ 一致 |
| 池化方式 | 论文未明说 | `mean` 池化（`constants.py` 注册） | — |

⚠️ **偏离**：论文正文与 Table 2 ablation 行 `multilingual-e5-small`/`gte-multilingual-base` 都将 `all-MiniLM-L6-v2` 作为 base 比较基线。代码实际默认使用 L12-v2。两者均为 6 层 / 12 层结构，输出维度同为 384，因此线性投影层 shape 不变，但模型权重不同，**复现论文数值时需手动改回 L6-v2**。

### 4.2 文本单元（自由文本 + 类别）

代码 `tokenizer.process_features` 对 `dtype == "object"` 列：直接用 `texts_to_tensor` 调用 `SentenceEmbedder.embed`，再经 `CellEmbeddings.content_remapping = nn.Linear(384, hidden)` 投到目标维度。
论文 Section 3.1：自由文本与类别列共用同一文本嵌入器，再过一个可学习线性层映射。✅ 一致。

### 4.3 列名（column header）

论文："We embed column headers with the same model used for text cells. The result is passed through a separate learnable linear layer to map to the correct target dimension and summed with the cell embedding."

代码：`Tokenizer.__call__` 中 `data["column_embeddings"] = texts_to_tensor([col_names + target_name])`；`CellEmbeddings.column_remapping = nn.Linear(384, hidden)` 与 `content_remapping` 是两个独立的线性层。最后所有 embedding 求和后再 LayerNorm。

✅ 一致（确实是独立的线性层，并求和+LayerNorm）。

### 4.4 日期编码

论文："we embed each of the numbers representing day, month, and year separately and sum the three resulting vectors."

代码 `CellEmbeddings.DateEmbeddings`：
```python
self.year_embeddings    = nn.Embedding(52, hidden_size)   # 2000–2050 + NaN sentinel 0
self.month_embeddings   = nn.Embedding(13, hidden_size)
self.day_embeddings     = nn.Embedding(32, hidden_size)
self.weekday_embeddings = nn.Embedding(8, hidden_size)
return year + month + day + weekday
```

⚠️ **小幅偏离**：
1. 代码额外引入 `weekday` 维度（论文未提）。
2. 年份硬编码裁剪到 `[2000, 2050]`（`column_values.dt.year.clip(2000, 2050) - 2000`）；2000 之前与 2050 之后的年份会被映射到同一 embedding。论文未约束，但对历史时间序列表会丢失精度。

### 4.5 数值编码

论文 Section 3.1（默认 1-D 编码）：
> "first, we clip columns between the 2% and 98% quantiles of the distribution. Second, we scale them to have zero mean and unit variance ... Finally, the resulting number is multiplied by a learnable vector and a bias is added. If the original value was NaN, 0 is used instead, so the bias works as an 'is-NaN' flag."

代码：
- `Tokenizer.standard_scale_column`：
  ```python
  if self.is_valid:
      vmin, vmax = np.nanquantile(train_data, [0.005, 0.995])  # 0.5% / 99.5%
  else:
      vmin, vmax = np.nanpercentile(train_data, [2, 98])       # 2% / 98%
  ```
- `pretrain_run.py::build_model_and_tokenizer` 中显式传入 `is_valid=True`：
  ```python
  tokenizer = Tokenizer(..., is_valid=True, ...)
  ```
- `CellEmbeddings.number_embeddings = nn.Linear(1, hidden_size)`（即 "乘可学习向量 + bias"） ✅
- NaN 处理：在 `process_features` 中先用上下文均值填补，再 standard-scale。`number_normalized` 默认值为 `-100`，`CellEmbeddings.forward` 对 `<= -99` 的位置用零向量替换。

⚠️ **不一致 1（裁剪分位数）**：论文 Section 3.1 明确写 `2% / 98%` 用于训练稳定性。但代码预训练入口走 `is_valid=True` 分支，使用 `0.5% / 99.5%`。
- 这与 Table 2 中 `0.5% clipping` 行相符（该行相对 base 几乎无变化），所以并不会显著改变结果，但与论文正文文字描述不符。
- 若严格按 Section 3.1 复现，应将 `pretrain_run.py` 中 `is_valid=True` 改为 `False`，或在 `Tokenizer.standard_scale_column` 中调换两个分支。

⚠️ **不一致 2（NaN 处理）**：论文设计意图是 "NaN→0 让 bias 充当 is-NaN flag"。代码先用 **上下文均值** 填补再标准化（标准化后均值为 0，整体上仍能让 NaN ≈ 0），最终走的是 `numbers_normalized <= -99 → zero embedding` 的 sentinel 路径，仅在"整列被 drop"或 query/test 行的 sentinel 路径上严格符合论文描述。普通 NaN 单元在 forward 计算路径上并没有走 sentinel，而是走 `Linear(0) = bias`，**功能等价于论文描述的 is-NaN flag**，但实现路径不同。

### 4.6 目标列特殊编码

论文 Section 3.1 末尾："this breaks equivariance under permutation, we retain it as we found it to be effective in the most common scenario of few-classes classification."

代码：
- 分类（cross-entropy）：通过 `CellEmbeddings.target_embedding_layer_classif = nn.Embedding(QUANTILE_DIMENSION=64, hidden)` 给 context 行的目标列一个特殊的类别 id 嵌入；query 行的 target 被设为 -100，对应零向量。
- 回归（L2）：通过 `target_embedding_layer_reg = nn.Linear(1, hidden)` 嵌入归一化后的目标值，query 行用 sentinel 屏蔽。
- Clustering 分支：使用 `target_content_remapping` 对目标 cell 的文本嵌入做映射，保留语义。

✅ 与 Section 3.1 / 3.4 一致。

### 4.7 最终融合

`CellEmbeddings.forward` 末尾：`column_embeds + content_embeds + number_embeds + date_embeds + padded_target_embeds`，然后 `LayerNorm + Dropout(0.1)`。

✅ 与论文"After summation, the embeddings are normalized via layer normalization"一致；但论文未提及 0.1 dropout（来自复用的 `RobertaConfig`）。

---

## 5. 主干（Section 3.2）

### 5.1 二维注意力

论文："alternating 'horizontal' (cross-column) and 'vertical' (cross-row) self-attention transformer layers ... cross-column attention has no masking, while cross-row attention is masked so that each row can only attend to the provided context."

代码 `TwoDimensionalAttentionLayer`：
- `cross_column_layer`（不传 mask） + `cross_row_layer`（传 `attention_mask`）
- `RPT.build_context_attention_mask`：构造 (num_rows, num_rows) 的 0 / -inf mask，使得 query 行只能 attend 到自己 + 所有 context 行 ✅

论文："the feedforward MLP block of the transformer encoder is repeated after each self-attention block so that 'horizontal' and 'vertical' blocks have the same structure."

代码：每个 `TorchRobertaLayer` = `Attention + Intermediate(FFN) + Output(FFN)`，cross_column 与 cross_row 分别封装一个完整 RobertaLayer，因此每个二维块包含 **两个 FFN** 与两次 LayerNorm，对应论文设计 ✅。

### 5.2 模型尺寸

论文 base 配置："n=12, d=768, d_ff=3072 → 172M params (16M trainable with weight sharing)"

代码：
```python
class ModelSize(Enum):
    base = (12, 768)   # ✅
    ...
self.config = RobertaConfig(
    num_hidden_layers=num_layers,                # 12 ✅
    hidden_size=hidden_size,                     # 768 ✅
    intermediate_size=hidden_size * 4,           # 3072 ✅
    num_attention_heads=hidden_size // 64,       # 12 heads ✅
)
```

✅ 主表 base 尺寸完全对齐。

### 5.3 ⚠️ 权重共享（关键偏离）

论文："Model weights can be optionally shared between each instance of the transformer block ... we observed that sharing weights did not affect model performance and thus we use weight sharing as the default option for our model in the following."

代码 `RPT.__init__`：
```python
self.in_context_encoder = nn.ModuleList(
    [TwoDimensionalAttentionLayer(self.config) for _ in range(self.config.num_hidden_layers)]
)
```

每层都是独立的 `TwoDimensionalAttentionLayer` 实例，**没有任何参数 tying**。仓库中也没有任何 `weight_sharing` 选项、`tie_weights` 调用，或控制流共享单层 forward 多次的代码。

`RPT.copy_last_layer_weights_to_all` 仅在 **加载 checkpoint** 时被调用：当发布的 checkpoint 中只保存了"最后一层"权重（说明它原本是共享的）时，将其复制到全部 12 个位置以适配新的 unshared 模型结构。它 **不影响训练时的参数 tying**。

❌ **不一致**：按 `pretrain_run.py` 直接训练得到的是 Table 2 中 `non-shared weights` 那一行（`All Rank 4.36`，相对 base 全面变差）的配置，而非论文主表的 `2.96 Rank` 配置。172 M 参数全部独立训练，显存与时间需求也是论文宣称的 ~10 倍。

如需对齐，需在 `RPT.__init__` 中支持例如：

```python
shared_layer = TwoDimensionalAttentionLayer(self.config)
self.in_context_encoder = nn.ModuleList([shared_layer] * num_layers)
```

或在 forward 中循环复用同一层 N 次。

### 5.4 ISAB（Section 3.4）

论文 Section 3.4 描述了 Induced Set Attention Block 作为可替换 cross-row attention 的备选架构。
代码无 ISAB 实现（`grep ISAB|InducedSet|inducing` 无结果）。

ℹ️ 这是论文描述的 alternative architecture，未实现属于合理省略，主表 ConTextTab 也未使用 ISAB。

---

## 6. 解码 / 损失（Section 3.3）

### 6.1 分类头

论文：MLP → 至少与类数等大的输出维度 → cross-entropy。

代码：`self.dense_classif = Linear(h, h); output_head_classif = Linear(h, QUANTILE_DIMENSION=64)`，`gelu` 激活 + cross-entropy。`Tokenizer.QUANTILE_DIMENSION = 64` 充当类数上限（与论文"the number of classes seen in pretraining cannot be exceeded at inference"对应）。

✅ 一致。

### 6.2 回归头

论文：直接预测浮点 + L2 loss + 标准化 / 反标准化。

代码 `compute_regression_output_loss_and_metric`（l2 分支）：
```python
loss_reg = F.mse_loss(masked_logits, masked_labels)
loss_reg = torch.clip(loss_reg, 0, 10)
```

✅ 一致；额外加了 `clip(0, 10)` 防止极端值，论文未提及，属于实现稳健化。

### 6.3 Clustering 头（Section 3.4 备选）

代码同时支持 `classification_type ∈ {cross-entropy, clustering, clustering-cosine}`，并实现了 Eq.(1) 的 `binary_cross_entropy(clip(cosine_similarity), adjacency)` 形式。

✅ 与 Section 3.4 公式一致；另支持 dot-product 变体 `clustering`。

---

## 7. 训练循环（Section 4.1）

| 维度 | 论文 | 代码 | 对齐情况 |
| --- | --- | --- | --- |
| 训练步数 | 4M – 10M | `max_steps: 8_000_000`（默认） | ✅ 在范围内 |
| micro batch size | 1 | `micro_batch_size: 1` 且显式校验 ≠ 1 时报错 | ✅ |
| 梯度累积模拟 batch | 256（mini 用 128） | `resolved_accumulate_grad_batches → 128 if mini else 256` | ✅ |
| 优化器 | AdamW | `torch.optim.AdamW` | ✅ |
| 学习率上限 | 1e-4 | `learning_rate: 1e-4` | ✅ |
| Warmup | linear warmup 1000 步 | `warmup_steps: 1000`, `LambdaLR(min(step/warmup, 1))` | ✅ |
| 梯度裁剪 | "we employ gradient clipping" | `torch.nn.utils.clip_grad_norm_(..., gradient_clip_val=1.0)` | ✅ 一致；阈值 1.0 论文未明确 |
| 混合精度 | 论文未明说 | bf16（H100）/ fp16（≤A100），通过 `infer_autocast_dtype` 与 `GradScaler` | ➕ 实用补充 |

✅ 主体训练循环逻辑与论文完全一致。

### 7.1 课程学习 stage 2

论文 4.1 末段："we further added in a second step using the same training data as Ma et al. [25] ... we increased the number of rows used for training to 4000."

代码：
- `FinetuneConfig.curriculum_stage2_data_root_path: Path | None = None`（默认关闭）
- `curriculum_stage2_max_num_rows: 4000` ✅
- `curriculum_stage2_max_steps: 4_000_000`
- `pretrain_run.main` 在 stage1 结束后若 `use_curriculum_stage2=True` 则连续跑 stage2

✅ 与论文一致；但代码未实现论文所述"以 80% / 20% 概率混合 T4 与 Ma et al. 数据采样"（Appendix A.4）。代码 stage2 是 **完全切换** 到 stage2 数据集，不是论文 80/20 混合。这是一个实现细节上的偏离。

---

## 8. 推理时配置（与预训练相关项）

仅作参考（不在用户问的预训练范围，但与 Section 4.1 末段相关）：

- `MAX_NUM_COLUMNS = 500`（`rpt.py`）：与论文"in each bag, we sample up to 500 columns"一致 ✅
- 8-fold bagging 默认 ✅
- 推理 context size 默认 8192 ✅（README + 默认参数）

---

## 9. 关键偏离汇总

按影响程度排序（✅ = 2026-05-07 三组实验改动已修复，详见 `three_experiment_changelog.md`）：

| # | 偏离 | 影响 | 修复建议 / 状态 |
| --- | --- | --- | --- |
| 1 | ~~**`RPT` 没实现 weight sharing**，与论文默认配置不符~~ | 训练参数从 16 M → 172 M，显存/时间 ~10×；性能对应 Table 2 的 `non-shared weights` 行（全面下滑） | ✅ Task 1：`RPT.__init__(weight_sharing=True)` 默认开启；`FinetuneConfig.weight_sharing=True` |
| 2 | ~~默认句向量模型 L12-v2 ≠ 论文 L6-v2~~ | 不影响维度，但权重不同；Table 2 ablation 显示不同 embedder 影响 < 1% | ✅ Task 2：默认改回 `all-MiniLM-L6-v2`；`Tokenizer` 接受 `sentence_embedding_model_name` 参数 |
| 3 | ~~数值列裁剪走 `is_valid=True` 分支（0.5%/99.5%）~~ | 与 Section 3.1 文字描述（2%/98%）不一致；与 Table 2 `0.5% clipping` 行结果差异极小 | ✅ Task 3：`Tokenizer(clip_quantile=0.02)` 默认 2/98；`is_valid` 仅作为 deprecated 别名；推理侧保留 0.005 |
| 4 | ~~`max_num_columns: 50` 限制（论文未提）~~ | T4 中位数 9 列，多数表不受影响；但更宽的表会被丢弃，可能解释论文中"模型受限于 wider tables"的观察 | ✅ Task 4：`TableRulesConfig.max_num_columns = 500`，与推理侧 `MAX_NUM_COLUMNS` 一致 |
| 5 | 日期编码额外加 `weekday`，年份硬裁到 [2000,2050] | 论文未述；2000 前数据丢失分辨率 | 🟡 Task 5：`use_weekday` 可控（默认 False）；年份范围因破坏 ckpt 兼容性留作 TODO |
| 6 | NaN→0 的实现路径与论文文字略不同（先用上下文均值填补再 standardize） | 数学上等价（标准化后均值即 0），训练时仍是常向量 + bias | 不需要修复 |
| 7 | ~~Stage 2 课程学习使用单数据源 vs 论文 80/20 混合~~ | 仅在启用 stage 2 时影响；论文 Table 2 显示 stage 2 提升不显著 | ✅ Task 6：`MixedRPTDataset` + `curriculum_stage2_mixing_ratio=0.8`；默认仍关闭 stage 2 |
| 8 | ISAB 块未实现 | 论文为可选架构，base 模型本来不用 | 不影响主结果对齐 |

---

## 10. 论文未明确但代码合理的工程补充

- LRU cache（`utils/lru_cache.py`，默认 100 万条）缓存重复文本嵌入 —— 显著加速 T4 训练
- bf16/fp16 自动选择
- gradient checkpointing 选项 (`checkpointing_segments`)
- Parquet streaming read，避免一次性把整个 T4 加载进内存
- 类别值在 `process_target` 中如果出现"嵌入向量碰撞"（不同类被嵌成相同向量），自动加前缀重新嵌入
- 损失裁剪 `clip(loss_reg, 0, 10)` 防止训练爆炸

---

## 11. 验证

为减少臆断，本次审计除阅读源码外做了以下交叉验证：
- `grep weight.sharing|share.weights|tie.weights` 全仓 → 0 命中（佐证第 9.1 条结论）
- `grep ISAB|InducedSet|inducing` 全仓 → 0 命中（佐证 5.4）
- `grep all-MiniLM-L6-v2` → 仅出现在 `constants.py` 的注册表（即支持但未使用），`Tokenizer.sentence_embedding_model_name` 实际指向 L12-v2
- 比对 `RobertaConfig(intermediate_size=hidden*4)` → 768 × 4 = 3072，与论文 d_ff 一致

---

## 12. 结论

代码在 **数据流水线、行/列采样、目标选择、优化器与训练循环、二维注意力主干、多模态嵌入合成、L2/CE/clustering 头**等方面忠实再现了论文。

但若用户的目的是 **从零复现论文 base 模型的数值**，需至少：

1. 在 `RPT` 中实现层间 weight sharing 并默认开启；
2. 将默认句向量模型切回 `all-MiniLM-L6-v2`；
3. 将预训练数值裁剪改为 `2% / 98%`（或确认论文实际使用 0.5%/99.5% —— Table 2 的存在使这一点存疑）；
4. （可选）放宽 `max_num_columns`、去掉日期 weekday 与年份硬裁；
5. （可选）若启用 stage 2，实现 80/20 混采样。

仅使用仓库发布的预训练 **checkpoint** 做推理时，上述偏离 1、2、4、5、7 不影响结果，因为 checkpoint 与代码一同发布、实际训练时使用了相应配置（特别是 weight-shared 的 16 M 权重，已通过 `copy_last_layer_weights_to_all` 在加载时复制到 12 层）。这也解释了为何 `load_weights` 必须 fallback 到这一函数：**published checkpoint 与 `pretrain_run.py` 训练得到的 state_dict 结构本质不同**。
