# 三组消融实验改动 Changelog

**审计日期**：2026-05-07
**对应 prompt**：`reports/claude_code_prompts.md`
**对应 diff**：`reports/code_alignment_diff.patch`

---

## 摘要

本次改动覆盖 Task 1–11，目标是支持三组 ablation：

| Exp | 句向量 | 融合 | 训练策略 | 启动命令 |
| --- | --- | --- | --- | --- |
| 1 | MiniLM-L6-v2 | FiLM | 从官方 HF ckpt post-training | `python -m sap_rpt_oss.run_experiment --exp 1` |
| 2 | Qwen3-Embedding-0.6B | sum | T4 从零预训练 | `LRU_CACHE_SIZE=10000000 python -m sap_rpt_oss.run_experiment --exp 2` |
| 3 | Qwen3-Embedding-0.6B | FiLM | T4 从零预训练 | `LRU_CACHE_SIZE=10000000 python -m sap_rpt_oss.run_experiment --exp 3` |

**不变性约束已满足**：默认 `FinetuneConfig()` 与 `SAP_RPT_OSS_Classifier/Regressor()`
无新增必填参数，调用方式不变。新功能全部通过显式参数 / config 工厂函数启用。

---

## 各 Task 改动详情

### Task 1 — RPT 层间权重共享（默认开启）

**文件**：`sap_rpt_oss/model/torch_model.py`、`sap_rpt_oss/configs.py`、`sap_rpt_oss/pretrain_run.py`

**关键 diff**（`torch_model.py::RPT.__init__`）：

```python
# 新增参数
weight_sharing: bool = True,
combination_type: Literal["sum", "film"] = "sum",
use_weekday: bool = False,
sentence_embedding_dim: int = 384,
verbose: bool = False,

# 关键：用 [shared_layer] * num_layers 重复同一对象
if weight_sharing:
    shared_layer = TwoDimensionalAttentionLayer(self.config)
    self.in_context_encoder = nn.ModuleList(
        [shared_layer] * self.config.num_hidden_layers
    )
else:
    self.in_context_encoder = nn.ModuleList(
        [TwoDimensionalAttentionLayer(self.config)
         for _ in range(self.config.num_hidden_layers)]
    )
```

`FinetuneConfig.weight_sharing: bool = True` 透传到 `pretrain_run.py::build_model_and_tokenizer`。

`load_weights` 的 `copy_last_layer_weights_to_all` fallback 路径**保留**，确保从官方 HF ckpt
（只保存 1 层权重）加载时仍能复制到 12 层共享对象。

**验证（静态）**：
- `tests/test_film_ablation.py::test_weight_sharing_active_by_default` — 对 base 模型构造后
  检查 `{id(layer) for layer in m.in_context_encoder}` 长度为 1，且 trainable params < 60 M。
- `test_weight_sharing_disabled_yields_independent_layers` — 检查 `weight_sharing=False`
  时各层独立。

---

### Task 2 — 默认句向量回退到 `all-MiniLM-L6-v2`

**文件**：`sap_rpt_oss/data/tokenizer.py`、`sap_rpt_oss/configs.py`、`sap_rpt_oss/pretrain_run.py`

**关键 diff**：
- `Tokenizer.sentence_embedding_model_name` 类属性从 `L12-v2` → `L6-v2`（与官方 HF ckpt 对齐）。
- 新增 `__init__` 参数 `sentence_embedding_model_name`，默认 L6-v2，未知模型抛错。
- `embedding_dim` 改为 `__init__` 中根据模型名动态查表（来自
  `embedding_model_to_dimension_and_pooling`）；类属性保留作为兼容默认值。
- `FinetuneConfig.sentence_embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"`。
- `pretrain_run.py::build_model_and_tokenizer` 显式传入 config 中的模型名。

---

### Task 3 — 数值列裁剪分位数 2/98

**文件**：`sap_rpt_oss/data/tokenizer.py`、`sap_rpt_oss/pretrain_run.py`、`sap_rpt_oss/rpt.py`

**关键 diff**：
- `Tokenizer.__init__` 新增 `clip_quantile: float = 0.02`；`is_valid` 保留为 deprecated 别名
  （传入会触发一次 `DeprecationWarning`，并按 `is_valid → 0.005 / 0.02` 等价转换）。
- `standard_scale_column` 现在用 `q_lo, q_hi = self.clip_quantile, 1 - self.clip_quantile`
  （`np.nanquantile`，原 `is_valid=False` 分支用的 `np.nanpercentile` 在数值上等价但
  `np.nanquantile` 是更现代的 API）。
- `pretrain_run.py::build_model_and_tokenizer` 改用 `clip_quantile=0.02`（与论文 Section 3.1 文字描述一致）。
- `rpt.py::SAP_RPT_OSS_Estimator.__init__` 改用 `clip_quantile=0.005 if is_valid else 0.02`，
  保留 `is_valid` 公开参数兼容已有调用方，与 ckpt 推理行为一致。

**验证（grep）**：
- 仓库中 `is_valid=True` 仅在 `rpt.py` 的内部分支存在；外部 API 默认值未变。
- `clip_quantile=0.02` 出现在 `pretrain_run.py::build_model_and_tokenizer`。

---

### Task 4 — 放宽 `max_num_columns` 到 500

**文件**：`sap_rpt_oss/configs.py`

`TableRulesConfig.max_num_columns` 默认 `50 → 500`，与推理侧 `MAX_NUM_COLUMNS = 500` 对齐。

---

### Task 5 — 日期编码可配置（去掉 weekday）

**文件**：`sap_rpt_oss/model/embeddings.py`、`sap_rpt_oss/model/torch_model.py`、
`sap_rpt_oss/configs.py`

**关键 diff**：
- `DateEmbeddings.__init__` 新增 `use_weekday: bool = False`；`weekday_embeddings` Embedding 仍
  构造（保 ckpt 兼容），但 forward 中按 `use_weekday` 决定是否相加。
- 透传到 `CellEmbeddings`、`RPT`、`FinetuneConfig`，全部默认 `False`。
- 年份范围扩展属于 ckpt-incompatible 改动，按 prompt 要求**未实现**，留作 TODO。

---

### Task 6 — Stage 2 课程学习 80/20 混采样

**文件**：`sap_rpt_oss/data/ds.py`（新增 `MixedRPTDataset`）、`sap_rpt_oss/configs.py`、
`sap_rpt_oss/pretrain_run.py`

**关键 diff**：
- `MixedRPTDataset(IterableDataset)` 在两个 `RPTParquetDataset` 上做 Bernoulli 抽样
  （`p=mixing_ratio` 选 primary，否则 secondary）；某条流耗尽时回退到另一条流。
- `FinetuneConfig` 新增：
  - `curriculum_stage2_t4_data_root_path: Path | None = None`（默认复用 stage1 的 `data_root_path`）
  - `curriculum_stage2_mixing_ratio: float = 0.8`
- `pretrain_run.py::main` stage 2 入口：当 `use_curriculum_stage2=True` 时，primary 用 T4 路径
  （默认复用 stage1），secondary 用 `curriculum_stage2_data_root_path`（Ma et al. 数据），
  传入 `mixing_ratio=0.8`。

**默认行为不变**：`curriculum_stage2_data_root_path is None` 时 `use_curriculum_stage2=False`，
完全不进入 stage 2。

---

### Task 8 — 三个实验的 config 工厂函数

**文件**：`sap_rpt_oss/configs.py`

新增 `get_paper_aligned_pretrain_config / get_exp1_minilm_film_config /
get_exp2_qwen3_sum_config / get_exp3_qwen3_film_config`。

**实现方案**：去掉 `@dataclass(slots=True)`，全部用 `dataclasses.replace` 构造（slots=True 与
`replace` + 嵌套 `default_factory` 在某些边界场景下会抛错，去掉是最稳健的）。

---

### Task 9 — FiLM 嵌入融合（Exp 1 / Exp 3 共用）

**文件**：`sap_rpt_oss/model/embeddings.py`、`sap_rpt_oss/model/torch_model.py`、
`sap_rpt_oss/configs.py`

**关键 diff**：
- 新增 `FiLMGenerator(nn.Module)`：`proj1 = Linear(h, h) → GELU → proj2 = Linear(h, 2h)
  → split → (1+γ, β)`；**`proj2` zero-init**（γ=1, β=0），保证起点等价 sum 模式。
- `CellEmbeddings.__init__` 新增 `combination_type: Literal["sum", "film"] = "sum"`；
  `combination_type="film"` 时构造 `self.film = FiLMGenerator(hidden)`。
- `CellEmbeddings.forward` 实现 **方案 A**（推荐）：
  ```python
  if self.combination_type == "film":
      cell_sum = content_embeds + number_embeds + date_embeds
      gamma, beta = self.film(column_embeds)
      input_embeds = gamma * cell_sum + beta + column_embeds
  else:
      input_embeds = column_embeds + content_embeds + number_embeds + date_embeds
  ```
  zero-init 起点 `gamma=1, beta=0`，再加回 `column_embeds`，与 sum 模式严格等价。
- `RPT.__init__` 透传 `combination_type`；`load_weights` 在 film 模式下用 `strict=False`
  并校验 missing keys 全部以 `embeddings.film.` 开头，打印
  "FiLM parameters initialized from scratch (not in checkpoint)."。

**验证**：`tests/test_film_ablation.py::test_film_init_equivalent_to_sum` —
分别构造 sum / film 两个 RPT，用同种子初始化，再把 sum 的 state_dict load 进 film
（strict=False），对比 dummy batch 的 forward 输出 max abs diff < 1e-5。

---

### Task 10 — Qwen3-Embedding-0.6B 句向量支持

**文件**：`sap_rpt_oss/constants.py`、`sap_rpt_oss/data/sentence_embedder.py`、
`sap_rpt_oss/data/tokenizer.py`、`sap_rpt_oss/model/embeddings.py`、
`sap_rpt_oss/model/torch_model.py`

**关键 diff**：
- `constants.embedding_model_to_dimension_and_pooling` 新增
  `"Qwen/Qwen3-Embedding-0.6B": (1024, "last_token")`。
- `SentenceEmbedder.pooling` 新增 `last_token` 分支：当 `tokenizer.padding_side == "left"`
  时直接取 `[:, -1]`，否则用 `attention_mask.sum(1) - 1` 索引（兼容 right-padding）。
- `SentenceEmbedder.__init__` 在 `pooling_method == "last_token"` 时强制
  `tokenizer.padding_side = "left"`（Qwen3-Embedding 官方推荐 left-padding）。
- `Tokenizer.embedding_dim` 改为 `__init__` 中动态查表赋值。
- `CellEmbeddings.__init__` 新增 `sentence_embedding_dim: Optional[int] = None`，
  替代直接读 `Tokenizer.embedding_dim`；`column_remapping / content_remapping /
  target_content_remapping` 三个 Linear 都用此参数。
- `RPT.__init__` 新增 `sentence_embedding_dim: int = 384`，透传给 `CellEmbeddings`。
- `pretrain_run.py::build_model_and_tokenizer` 先构造 Tokenizer，再用
  `tokenizer.embedding_dim` 构造 RPT，确保 adapter 形状随句向量模型动态调整。
- LRU cache 默认值未改（`int(os.getenv("LRU_CACHE_SIZE", 1_000_000))`），通过
  `LRU_CACHE_SIZE=10000000` 环境变量在启动脚本里上调（README 已写明）。

**验证**：`tests/test_qwen3_embedder.py`（默认 skip，需 `RUN_QWEN3_TEST=1`，避免在 CI 下载 1.2 GB）：
- `test_qwen3_embed_shape` — 检查输出 (2, 1024)。
- `test_qwen3_rpt_construction` — 检查 `column_remapping.weight.shape == (256, 1024)`。

预计算脚本（Task 10.6）作为 Phase 2 优化未实现，按 prompt 同意。

---

### Task 11 — 统一启动入口 `sap_rpt_oss/run_experiment.py`

**关键内容**：
- argparse: `--exp {1,2,3}`、`--max-steps N`（dry-run 用）、`--data-root PATH`（覆盖数据路径）、`--no-resume`（关闭自动续训）。
- 字典 `EXPERIMENTS` 把 1/2/3 映射到对应 config 工厂函数。
- `download_official_ckpt()` 用 `huggingface_hub.hf_hub_download` 下载官方 ckpt：
  - `repo_id = "SAP/sap-rpt-1-oss"`
  - `filename = "2025-11-04_sap-rpt-one-oss.pt"`（与 `rpt.py` 默认 ckpt 名一致）
- 仅 Exp 1 使用 `download_official_ckpt`；Exp 2/3 走 `pretrain_from_scratch=True` 分支。
- README 末尾追加 "Reproducing the ablation experiments" 章节，给出 3 条命令。

### Resume from latest checkpoint（默认开启）

**文件**：`sap_rpt_oss/pretrain_run.py`、`sap_rpt_oss/run_experiment.py`

**关键 diff**：
- `pretrain_run.py` 新增 `find_latest_checkpoint(checkpoint_dir, stage_name)`：扫描目录里
  形如 `{stage_name}-{N}-step.pt` 的文件，按 N 取最大；按 stage_name 过滤避免 stage1 / stage2
  ckpt 串台。
- `run_stage` 新增 `start_step: int = 0` 参数；构造 LR scheduler 后用循环 `scheduler.step()`
  把 warmup 状态推进到对应步；`tqdm(total=max_steps, initial=start_step)` 让进度条从断点
  显示；`global_step / last_saved_step` 都用 `start_step` 初始化。
- `run_experiment.py::main` 在构造 model 之前先 `find_latest_checkpoint(checkpoint_dir,
  stage_name)`：找到则覆盖 `pretrain_from_scratch=False`、把本地 ckpt 路径喂给
  `build_model_and_tokenizer`、把 N 喂给 `run_stage(start_step=N)`；否则保持原行为
  （Exp 1 走 HF download，Exp 2/3 走 from-scratch init）。
- `--no-resume` flag 关闭自动续训，强制冷启动。
- 边界处理：若 `start_step >= max_steps`，直接 print 提示并 return（避免空进度条死循环或
  抛"no batches yielded"）。

**步数预算**（来自工厂函数）：

| Exp | `max_steps` | `warmup_steps` | 说明 |
| --- | --- | --- | --- |
| 1 | 200,000 | 2,000 | post-training，明确缩短 |
| 2 | 8,000,000 | 1,000 | 论文 base 下限（论文范围 4M–10M） |
| 3 | 8,000,000 | 1,000 | 同 Exp 2 |

---

## 不变性 / 兼容性回归点

1. **README 中的 breast_cancer / openml-531 demo**：未触及代码路径，但 RPT 默认参数变化
   （`weight_sharing=True`，原默认行为是 unshared）会改变内部结构。
   - 推理路径 `SAP_RPT_OSS_Estimator.__init__` 构造 RPT 不传 `weight_sharing`，使用新默认 `True`。
   - `load_weights` 的 `copy_last_layer_weights_to_all` fallback 与 weight-shared 模型兼容
     （ModuleList 含 12 个共享引用，最终所有 key 写入同一物理 tensor）。
   - **风险点**：若用户手中有过去 unshared 训练的自定义 ckpt，加载到默认 weight-shared 模型
     会被静默退化为只保留最后一层权重。该风险已在审计报告中列出。

2. **公开 API 不变**：`Tokenizer / RPT / SAP_RPT_OSS_Classifier / SAP_RPT_OSS_Regressor`
   构造时**不传新参数**仍能正常工作，行为与论文/官方 ckpt 对齐（即比修改前更正确）。
   `is_valid` 保留为 Tokenizer 的公开兼容参数，仅触发 DeprecationWarning。

3. **examples/ 目录未改**：`sample_classification.py / sample_regression.py` 仍能直接 `python` 运行。

---

## 三组 dry-run 数据（需在服务器上运行）

> 本机环境暂未跑通，按用户要求"放在服务器上运行"。下面是建议在服务器上跑的命令清单：

```bash
# pytest 单元测试
pytest tests/test_film_ablation.py -v
RUN_QWEN3_TEST=1 pytest tests/test_qwen3_embedder.py -v   # 会下载 Qwen3 ~1.2 GB

# breast_cancer 推理 demo（保护已发布 ckpt 的加载兼容性）
python -c "
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sap_rpt_oss import SAP_RPT_OSS_Classifier
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=42)
clf = SAP_RPT_OSS_Classifier(max_context_size=1024, bagging=1)
clf.fit(X_tr, y_tr)
print('accuracy:', clf.score(X_te, y_te))
"

# 三组 dry-run（10 步）
python -m sap_rpt_oss.run_experiment --exp 1 --max-steps 10
LRU_CACHE_SIZE=10000000 python -m sap_rpt_oss.run_experiment --exp 2 --max-steps 10
LRU_CACHE_SIZE=10000000 python -m sap_rpt_oss.run_experiment --exp 3 --max-steps 10
```

期望（按 prompt）：
- Exp 1 base + weight_sharing=True：`trainable_params` 在 ~16-30 M 量级（不应是 172 M）。
- Exp 2 base + Qwen3 1024 dim adapter：略高于 Exp 1（每个 (768,1024) Linear 多 ~0.4M，三层共 ~1.2M）。
- Exp 3 = Exp 2 + FiLM 投影 (768→2*768) ~1.2 M：再加 ~1 M。
- 三组都应能跑 10 步、loss 不 NaN、`outputs/exp{1,2,3}_*` 下生成 ckpt。

---

## 论文对齐覆盖率（Task 1–8）

参考 `reports/pretraining_alignment_report.md` 第 9 节"关键偏离汇总"：

| # | 偏离 | 状态 |
| --- | --- | --- |
| 1 | RPT 没实现 weight sharing | ✅ Task 1 已修复（默认开启） |
| 2 | 默认句向量 L12-v2 ≠ 论文 L6-v2 | ✅ Task 2 已修复 |
| 3 | 预训练裁剪走 0.5%/99.5% 分支 | ✅ Task 3 已修复（pretrain 用 0.02，inference 保留 0.005） |
| 4 | `max_num_columns: 50` 限制 | ✅ Task 4 已修复（500） |
| 5 | 日期编码 weekday + 年份硬裁 | 🟡 Task 5 部分修复（weekday 关闭可控；年份范围保留为 TODO 以保 ckpt 兼容） |
| 6 | NaN 处理路径与论文文字略不同 | ⏭️ 数学等价，按 prompt 跳过 |
| 7 | Stage 2 单数据源 vs 论文 80/20 混合 | ✅ Task 6 已修复（默认仍关闭 stage 2） |
| 8 | ISAB 块未实现 | ⏭️ 论文为可选架构，base 不用 |

---

## 已知 Caveat / 待办

1. **Qwen3 训练成本**：单次嵌入 ~30× MiniLM。无大 LRU cache（建议 1000 万条）几乎无法 from-scratch
   在合理时间内跑完 Exp 2/3。已通过 README + Task 10.5 文档化 `LRU_CACHE_SIZE=10000000` 必须 export。
   预计算脚本（Task 10.6）作为 Phase 2 优化保留。
2. **官方 HF ckpt 文件名**：`run_experiment.py` 中 `OFFICIAL_CHECKPOINT_FILENAME =
   "2025-11-04_sap-rpt-one-oss.pt"` 取自 `rpt.py` 的默认值；如 SAP 后续重命名 ckpt 文件，需要
   同步更新此常量。
3. **weight_sharing 默认变更**：用户手中如有 unshared 训练的自定义 ckpt，加载到新默认会
   silent-degrade。文档中已说明，迁移路径是显式传 `RPT(..., weight_sharing=False)`。
4. **未本地跑通三组 dry-run**：按用户要求在服务器跑，本 changelog 的"dry-run 数据"
   章节实际未填充 loss/throughput 数据，留待服务器结果回填。
