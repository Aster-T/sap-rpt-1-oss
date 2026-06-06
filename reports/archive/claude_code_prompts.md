# sap-rpt-1-oss 三组消融实验任务清单（Claude Code）

> 把下面整段（从 `# 任务背景` 起）原样交给 Claude Code 执行。

---

# 任务背景

仓库 `D:\Projects\sap-rpt-1-oss`（容器内 `/sessions/.../mnt/sap-rpt-1-oss`）是 ConTextTab 论文（`papers/ContextTab.pdf`）的 OSS 实现。预训练对齐审计在 `reports/pretraining_alignment_report.md`，**请先读这两份文件再开工**。

我要训练 **3 个模型** 做 ablation：

| 实验 | 句向量模型 | embedding 融合方式 | 训练策略 | 目的 |
| --- | --- | --- | --- | --- |
| **Exp 1** | `sentence-transformers/all-MiniLM-L6-v2` | FiLM | 从官方 HF ckpt `sap/sap-rpt-1-oss` post-training | 单独验证 FiLM 是否优于 sum |
| **Exp 2** | `Qwen/Qwen3-Embedding-0.6B` | sum（论文默认） | 从零在 T4 上预训练 | 单独验证更强的句向量模型是否优于 MiniLM |
| **Exp 3** | `Qwen/Qwen3-Embedding-0.6B` | FiLM | 从零在 T4 上预训练 | 组合效果 |

**重要事实纠正**：当前代码里 `Tokenizer.sentence_embedding_model_name = "all-MiniLM-L12-v2"` 是历史本地修改，**官方 HF checkpoint 实际用的是 `all-MiniLM-L6-v2`**。所以"换回 L6-v2"既对齐论文也对齐官方 ckpt，没有 trade-off。

**实验设计原则**：

- Exp 1 与 Exp 2 是干净的两组对照（一个动 FiLM，一个动 embedder），便于分别归因。
- Exp 2 与 Exp 3 之间只差 FiLM 一项。
- 每组实验各跑各的 checkpoint，**不要混用**，否则结果归因不清。

请按下面 Tasks 顺序执行。每完成一个 Task：
1. 给出 `git diff`；
2. 跑该 Task 自带的 `验证`（如有）；
3. 任何破坏 `examples/` 中两个 demo 的改动 **必须默认关闭**。

> **不变性约束**：默认配置（不传任何新参数）下，仓库行为必须与改动前完全等价 —— 即 README 里的分类/回归 demo 用 `sap/sap-rpt-1-oss` ckpt 仍能跑通。所有新功能都通过显式参数 / 环境变量启用。

---

# Foundation Tasks（三组实验都需要）

## Task 1 — 在 `RPT` 中实现层间权重共享并默认开启

**论文依据**：Section 3.2，"we use weight sharing as the default option"。Table 2 中 `non-shared weights` 行相对 base 全面下滑 → 当前代码就是 unshared 配置。官方 HF ckpt 实际是 weight-shared 的（这是 `RPT.copy_last_layer_weights_to_all` 存在的原因）。

**改动位置**：`sap_rpt_oss/model/torch_model.py`、`sap_rpt_oss/configs.py`。

**要求**：

1. `RPT.__init__` 增加 `weight_sharing: bool = True`。
2. `weight_sharing=True` 时：
   - 只构造 **一个** `TwoDimensionalAttentionLayer` 实例；
   - `self.in_context_encoder = nn.ModuleList([shared_layer] * num_layers)`（必须用 `* num_layers`，不要 list-comprehension，否则会构造多次）；
   - forward 路径不需要改 —— `nn.ModuleList` 中重复对象自动共享参数。
3. `weight_sharing=False` 保持现有行为（每层独立）。
4. `FinetuneConfig` 增加 `weight_sharing: bool = True`，在 `pretrain_run.py::build_model_and_tokenizer` 传入。
5. 保留 `copy_last_layer_weights_to_all` 与 `load_weights` 的 fallback 路径 —— 必须能加载 HF ckpt。
6. `weight_sharing=True` 时打印一次 trainable 参数量。base 应在 ~16-30M 量级，不应是 172M。

**验证**：

```python
from sap_rpt_oss.model.torch_model import RPT
from sap_rpt_oss.constants import ModelSize

m = RPT(model_size=ModelSize.base, regression_type="l2", classification_type="cross-entropy")
ids = {id(layer) for layer in m.in_context_encoder}
assert len(ids) == 1, "weight sharing not active"
print("trainable params:", sum(p.numel() for p in m.parameters() if p.requires_grad))
```

跑 README 中的 breast_cancer 分类 demo 验证 HF ckpt 能加载。

---

## Task 2 — 句向量模型默认改回 `all-MiniLM-L6-v2`

**论文依据 + 兼容性**：论文 Section 3.1 用 L6-v2；**官方 HF checkpoint 也是用 L6-v2 训的**。当前代码里写的 L12-v2 是历史本地改动，应当撤回。

**改动位置**：`sap_rpt_oss/data/tokenizer.py`。

**要求**：

1. `Tokenizer.sentence_embedding_model_name` 改回 `"sentence-transformers/all-MiniLM-L6-v2"`。
2. 把 `sentence_embedding_model_name` 提升为 `Tokenizer.__init__` 的参数（默认 L6-v2），方便 ablation 切换。
3. `Tokenizer.embedding_dim` 不再做类属性写死，改成 `__init__` 中根据 `sentence_embedding_model_name` 动态查表赋值（来自 `constants.embedding_model_to_dimension_and_pooling`）—— 这是 Task 10 的前置改造。
4. `pretrain_run.py::build_model_and_tokenizer` 显式传入模型名（默认 L6-v2，但允许 config 覆盖）。

**验证**：

```python
from sap_rpt_oss.data.tokenizer import Tokenizer
tok = Tokenizer(sentence_embedder_device="cpu")
assert tok.sentence_embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
assert tok.embedding_dim == 384
```

跑分类 demo 确认 ckpt 仍正常加载（这次应该比之前 L12-v2 表现还好一点，因为 ckpt 本来就是 L6 训的）。

---

## Task 3 — 修复数值列裁剪分位数

**论文依据**：Section 3.1，"clip columns between the 2% and 98% quantiles"（预训练用）。
**当前 bug**：`pretrain_run.py` 显式传 `is_valid=True` → tokenizer 走 0.5/99.5 分支，**与 Section 3.1 文字描述不符**。

**改动位置**：`sap_rpt_oss/pretrain_run.py`、`sap_rpt_oss/data/tokenizer.py`。

**要求**：

1. `Tokenizer.__init__` 把 `is_valid: bool` 替换为 `clip_quantile: float = 0.02`（即默认 2% / 98%）。
2. `Tokenizer.standard_scale_column` 改为：
   ```python
   q_lo, q_hi = self.clip_quantile, 1.0 - self.clip_quantile
   vmin, vmax = np.nanquantile(train_data, [q_lo, q_hi])
   ```
3. 保留 `is_valid` 作为 deprecated 别名：传 `is_valid=True` → 等价 `clip_quantile=0.005`，并 `warnings.warn` 一次。
4. `pretrain_run.py::build_model_and_tokenizer` **去掉** `is_valid=True`，改用默认 `clip_quantile=0.02`。
5. 推理侧（`rpt.py::SAP_RPT_OSS_Estimator`）中调用 `Tokenizer` 的地方**保留** `clip_quantile=0.005`（与论文 Section 4.1 末段"在下游任务上根据 outlier 分析选择"一致），不破坏 ckpt 推理行为。

**验证**：grep `is_valid=True` 应只剩推理侧；`grep clip_quantile=0.02` 出现在 `pretrain_run.py`。

---

## Task 4 — 放宽预训练 `max_num_columns`

**改动位置**：`sap_rpt_oss/configs.py`。

`TableRulesConfig.max_num_columns` 默认从 `50` 改为 `500`，与推理侧 `MAX_NUM_COLUMNS` 一致。

**验证**：`from sap_rpt_oss.configs import FINETUNE_CONFIG; assert FINETUNE_CONFIG.table_rules.max_num_columns == 500`。

---

## Task 5 — 日期编码可配置（去掉 weekday、扩展年份范围）

**改动位置**：`sap_rpt_oss/model/embeddings.py`。

按 checkpoint 兼容优先：

1. `DateEmbeddings.__init__` 增加 `use_weekday: bool = False`：仍构造 `weekday_embeddings` Embedding（保兼容），但 forward 中按 `use_weekday` 决定是否相加。
2. `CellEmbeddings` 透传该参数。`RPT.__init__` 也加 `use_weekday: bool = False`。
3. **年份范围扩展**（`[1900, 2099]`）作为 TODO 标记，本轮不实现 —— 它会改 Embedding shape 破坏 ckpt 兼容。

**验证**：

```python
from sap_rpt_oss.model.embeddings import DateEmbeddings
import torch
de = DateEmbeddings(768, use_weekday=False)
out = de(torch.zeros(2, 3, 4, dtype=torch.long))
assert out.shape == (2, 3, 768)
```

---

## Task 6 — Stage 2 课程学习改 80/20 混采样

**论文依据**：Appendix A.4，"either T4 (80%) or [Ma et al. data] (20%)"。

**改动**：

1. `FinetuneConfig` 增加：
   - `curriculum_stage2_t4_data_root_path: Path | None = None`（默认复用 `data_root_path`）
   - `curriculum_stage2_mixing_ratio: float = 0.8`
2. 写 `MixedRPTDataset(IterableDataset)`：内部并行迭代两个 `RPTParquetDataset`，每次按伯努利 `p=mixing_ratio` 选数据源。
3. stage2 入口改用 `MixedRPTDataset`。`use_curriculum_stage2` 仍由 `curriculum_stage2_data_root_path is not None` 控制（默认关闭）。

---

## Task 7 — （可选低优先级）NaN 处理路径与论文文字对齐

按 `reports/pretraining_alignment_report.md` 第 9 节第 6 项描述。如时间紧可跳过，因为数学上等价。

---

## Task 8 — 三个实验的 config 工厂函数

**改动位置**：`sap_rpt_oss/configs.py`。

实现三个工厂函数：

```python
def get_paper_aligned_pretrain_config() -> FinetuneConfig:
    """ConTextTab 论文 base 配置（MiniLM-L6 + sum + 从头训）"""
    return FinetuneConfig(
        pretrain_from_scratch=True,
        model_size=ModelSize.base,
        weight_sharing=True,
        sentence_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        combination_type="sum",
        learning_rate=1e-4,
        warmup_steps=1000,
        gradient_clip_val=1.0,
        max_steps=8_000_000,
        accumulate_grad_batches=256,
        stage1_max_num_rows=1000,
        query_size_range=(50, 900),
        balance_classification_tasks=True,
        table_rules=TableRulesConfig(
            min_num_rows=150, max_num_columns=500,
            numeric_nan_ratio_threshold=0.5,
            categorical_unique_ratio_threshold=0.2,
            drop_constant_columns=True,
        ),
    )


def get_exp1_minilm_film_config() -> FinetuneConfig:
    """Exp 1: MiniLM-L6 + FiLM，从官方 HF ckpt post-training"""
    cfg = get_paper_aligned_pretrain_config()
    return dataclasses.replace(
        cfg,
        pretrain_from_scratch=False,                    # post-training
        resume_checkpoint_path=None,                    # 由 main 函数从 HF 下载，见 Task 11
        combination_type="film",
        max_steps=200_000,                              # post-training 步数远少于从头训
        warmup_steps=2_000,                             # 适当延长 warmup 让新 FiLM 参数稳定
        learning_rate=1e-4,                             # FiLM 参数从 zero-init 起步，全局同 LR 即可
        output_root_path=Path("outputs/exp1_minilm_film"),
        checkpoint_root_path=Path("checkpoints/exp1_minilm_film"),
    )


def get_exp2_qwen3_sum_config() -> FinetuneConfig:
    """Exp 2: Qwen3-Embedding-0.6B + sum，T4 从头预训练"""
    cfg = get_paper_aligned_pretrain_config()
    return dataclasses.replace(
        cfg,
        sentence_embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        combination_type="sum",
        # Qwen3 嵌入计算更慢，根据实际算力调整 max_steps
        output_root_path=Path("outputs/exp2_qwen3_sum"),
        checkpoint_root_path=Path("checkpoints/exp2_qwen3_sum"),
    )


def get_exp3_qwen3_film_config() -> FinetuneConfig:
    """Exp 3: Qwen3-Embedding-0.6B + FiLM，T4 从头预训练"""
    return dataclasses.replace(
        get_exp2_qwen3_sum_config(),
        combination_type="film",
        output_root_path=Path("outputs/exp3_qwen3_film"),
        checkpoint_root_path=Path("checkpoints/exp3_qwen3_film"),
    )
```

注意 `FinetuneConfig` 现在是 `slots=True`，可能需要去掉 `slots=True` 才能用 `dataclasses.replace`，**或者**直接构造新实例。请评估后选一种方案。

---

# Experiment 1 专属

## Task 9 — FiLM 嵌入融合模块（Exp 1 与 Exp 3 共用）

**论文依据**：论文 Section 3.1 默认 `column + content + number + date` 直接相加。本项是**对论文方法的扩展 ablation**。

**设计**：让 column header embedding 充当 conditioning 信号，对 cell 内容（content + number + date 之和）做 FiLM 调制：

```
column_embeds : (1, num_cols, hidden)                 # 已有
cell_sum      = content + number + date               # (num_rows, num_cols, hidden)
γ, β          = FiLMGenerator(column_embeds)          # 各 (1, num_cols, hidden)
out           = γ ⊙ cell_sum + β
out[:, -1]   += target_embeds                          # target 仍按原方式叠加
out           = LayerNorm(Dropout(out))
```

target embedding 不参与 FiLM —— 它是行级别的 sentinel/标签，不是列特征。

**改动位置**：`sap_rpt_oss/model/embeddings.py`、`sap_rpt_oss/model/torch_model.py`、`sap_rpt_oss/configs.py`。

**要求**：

1. `embeddings.py` 新增：

```python
class FiLMGenerator(nn.Module):
    """从 column header embedding 生成 per-column 的 (γ, β)。"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj1 = nn.Linear(hidden_size, hidden_size)
        self.proj2 = nn.Linear(hidden_size, 2 * hidden_size)
        # 关键：proj2 zero-init → 初始时 γ ≈ 1, β ≈ 0 → forward ≈ cell_sum
        # 这样 Exp 1 从官方 ckpt 起步时，FiLM 起点等价于"sum 但少了 column_embeds"
        # 见下方"⚠️ 起步 gap"
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

    def forward(self, column_embeds):
        h = torch.nn.functional.gelu(self.proj1(column_embeds))
        gamma_beta = self.proj2(h)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return 1.0 + gamma, beta
```

2. `CellEmbeddings.__init__` 增加 `combination_type: Literal["sum", "film"] = "sum"`；`"film"` 时构造 `self.film = FiLMGenerator(config.hidden_size)`，否则不创建（`hasattr` 即可）。

3. `CellEmbeddings.forward` 把：
   ```python
   input_embeds = column_embeds + content_embeds + number_embeds + date_embeds
   ```
   替换为：
   ```python
   if self.combination_type == "film":
       cell_sum = content_embeds + number_embeds + date_embeds
       gamma, beta = self.film(column_embeds)
       input_embeds = gamma * cell_sum + beta
   else:
       input_embeds = column_embeds + content_embeds + number_embeds + date_embeds
   ```

4. `RPT.__init__` 透传 `combination_type`；`FinetuneConfig` 增加同名字段，默认 `"sum"`。

5. `RPT.load_weights` 在 `combination_type="film"` 时 `load_state_dict(strict=False)`，并在 missing keys 中确认只有 `embeddings.film.*` 缺失。打印一次："FiLM parameters initialized from scratch (not in checkpoint)."

**⚠️ 起步 gap（重要）**：FiLM 公式没有 `column_embeds` 这一项。这意味着即使 γ=1, β=0，FiLM 模式的 forward 输出也比 sum 模式**少了 `column_embeds`**。从官方 ckpt 做 Exp 1 时这是个真实的 gap，不是 bug。处理方式两种，请实现 **方案 A**：

- **方案 A（推荐，简单干净）**：把 `column_embeds` 也加到 FiLM 输出上 —— 即 `out = γ ⊙ cell_sum + β + column_embeds`。这样 zero-init 时严格等价于 sum 模式，post-training 起点平滑。这与"FiLM 调制 cell 内容、列名作为偏置"的解读完全一致。
- **方案 B（更"纯"FiLM）**：让 FiLM 同时调制 column —— `γ ⊙ (column + cell_sum) + β`。物理意义略弱化，但 zero-init 起点也等价于 sum。

**实现方案 A**。在 `forward` 中：
```python
input_embeds = gamma * cell_sum + beta + column_embeds
```

**验证（新建 `tests/test_film_ablation.py`）**：

```python
import torch
from sap_rpt_oss.model.torch_model import RPT
from sap_rpt_oss.constants import ModelSize


def _make_dummy_batch(num_rows=8, num_cols=4, embed_dim=384):
    return {
        "column_embeddings": torch.randn(num_cols, embed_dim, dtype=torch.float16),
        "text_embeddings": torch.zeros(num_rows, num_cols, embed_dim, dtype=torch.float16),
        "date_year_month_day_weekday": torch.zeros(num_rows, num_cols, 4, dtype=torch.long),
        "number_normalized": torch.randn(num_rows, num_cols),
        "target": torch.zeros(num_rows),
        "target_delta": torch.zeros(num_rows),
    }


def test_sum_default_no_film_module():
    m = RPT(model_size=ModelSize.mini, regression_type="l2",
            classification_type="cross-entropy")
    assert not hasattr(m.embeddings, "film") or m.embeddings.film is None


def test_film_init_equivalent_to_sum():
    """FiLM zero-init + 加回 column_embeds → 与 sum 模式 forward 输出严格相等"""
    torch.manual_seed(42)
    m_sum = RPT(model_size=ModelSize.mini, regression_type="l2",
                classification_type="cross-entropy", combination_type="sum")
    m_sum.eval()

    torch.manual_seed(42)
    m_film = RPT(model_size=ModelSize.mini, regression_type="l2",
                 classification_type="cross-entropy", combination_type="film")
    m_film.eval()
    m_film.load_state_dict(m_sum.state_dict(), strict=False)

    batch = _make_dummy_batch()
    with torch.no_grad():
        out_sum = m_sum.embeddings(batch, is_regression=True)
        out_film = m_film.embeddings(batch, is_regression=True)
    diff = (out_sum - out_film).abs().max().item()
    print(f"sum vs film init max abs diff: {diff:.2e}")
    assert diff < 1e-5, f"FiLM 初始化未严格等价 sum (max diff={diff})"
```

---

# Experiment 2 / 3 专属

## Task 10 — Qwen3-Embedding-0.6B 句向量模型支持

**目标**：让代码能用 `Qwen/Qwen3-Embedding-0.6B`（输出 1024 维，last-token pooling）作为句向量模型，支撑 Exp 2 / 3 从零预训练。

**关键事实**：

- Qwen3-Embedding-0.6B 输出 **1024 维**（MiniLM 是 384，dim 不同 → adapter 必须根据模型动态构造）
- pooling 方式 **last_token**（当前代码只支持 `mean`、`cls`）
- 单次推理成本约 MiniLM-L6 的 **30×**
- 模型支持 MRL 截断 dim，但本任务**不使用 MRL** —— 直接用原生 1024 维，让 adapter 自适应
- 不使用 instruction prompt（直接 embed 字符串），保持与 MiniLM 等价的接口

**改动位置**：`sap_rpt_oss/constants.py`、`sap_rpt_oss/data/sentence_embedder.py`、`sap_rpt_oss/data/tokenizer.py`、`sap_rpt_oss/model/embeddings.py`。

**要求**：

### 10.1 注册 Qwen3 到模型表

`constants.py`:
```python
embedding_model_to_dimension_and_pooling = {
    "sentence-transformers/all-MiniLM-L6-v2": (384, "mean"),
    "sentence-transformers/all-MiniLM-L12-v2": (384, "mean"),
    "intfloat/multilingual-e5-small": (384, "mean"),
    "Alibaba-NLP/gte-multilingual-base": (768, "cls"),
    "Qwen/Qwen3-Embedding-0.6B": (1024, "last_token"),
}
```

### 10.2 扩展 `SentenceEmbedder` 池化

`sentence_embedder.py::SentenceEmbedder.pooling` 增加 `last_token` 分支：

```python
elif self.pooling_method == "last_token":
    # 取每条序列最后一个非 pad token 的 hidden state
    # attention_mask shape: (batch, seq_len)
    seq_lengths = attention_mask.sum(dim=1) - 1  # 每条序列最后一个有效位置
    batch_size = token_embeddings.size(0)
    last_token_emb = token_embeddings[torch.arange(batch_size), seq_lengths]
    return last_token_emb.type(token_embeddings.dtype)
```

注意 Qwen3 用 left-padding 还是 right-padding，看 `AutoTokenizer.from_pretrained(...).padding_side`。Qwen3-Embedding 官方推荐 **left-padding**，对应 last token 永远是序列倒数第一个位置。请用以下更稳健的实现（与 padding side 无关）：

```python
elif self.pooling_method == "last_token":
    # 兼容 left/right padding
    seq_lengths = attention_mask.sum(dim=1) - 1
    if self.tokenizer.padding_side == "left":
        return token_embeddings[:, -1].type(token_embeddings.dtype)
    else:
        batch_size = token_embeddings.size(0)
        return token_embeddings[torch.arange(batch_size), seq_lengths].type(token_embeddings.dtype)
```

且在 `SentenceEmbedder.__init__` 中显式设置：
```python
if self.pooling_method == "last_token":
    self.tokenizer.padding_side = "left"
```

### 10.3 `Tokenizer` 适配动态 embedding_dim

Task 2 中已经把 `embedding_dim` 改成 `__init__` 中动态计算。这里只需保证 Qwen3 时它正确读到 1024。
另外 `texts_to_tensor` 中的 `torch.zeros((0, self.embedding_dim), dtype=torch.float16)` 需要确认走的是 `self.embedding_dim` 而不是类属性。

### 10.4 `CellEmbeddings` 接受动态输入维度

当前 `embeddings.py`:
```python
self.column_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)
self.content_remapping = nn.Linear(Tokenizer.embedding_dim, config.hidden_size)
```
直接读类属性 —— 这在 Qwen3 时会读到 1024 还是 384 取决于 import 顺序，**不可靠**。改为：

```python
def __init__(self, config, ..., sentence_embedding_dim: int = 384):
    ...
    self.column_remapping = nn.Linear(sentence_embedding_dim, config.hidden_size)
    self.content_remapping = nn.Linear(sentence_embedding_dim, config.hidden_size)
    if self.is_target_content_mapping:
        self.target_content_remapping = nn.Linear(sentence_embedding_dim, config.hidden_size)
```

`RPT.__init__` 透传：从 tokenizer 拿到 `embedding_dim`，传给 `CellEmbeddings`。但是 `RPT` 当前不持有 tokenizer 引用 —— 改为接受 `sentence_embedding_dim` 参数（默认 384，向后兼容）：

```python
def __init__(self, model_size, regression_type=..., classification_type=...,
             weight_sharing=True, combination_type="sum",
             sentence_embedding_dim=384, ...):
```

`pretrain_run.py::build_model_and_tokenizer` 改为：
```python
tokenizer = Tokenizer(sentence_embedding_model_name=config.sentence_embedding_model_name, ...)
model = RPT(..., sentence_embedding_dim=tokenizer.embedding_dim, ...)
```

### 10.5 增大 LRU cache（Qwen3 时）

Qwen3 单次嵌入贵 30×，但 T4 中列名重复极高、类别值重复也高。增大 cache 上限是性价比最高的优化：

`tokenizer.py`:
```python
self.cache = LRU_Cache(max_size=int(os.getenv("LRU_CACHE_SIZE", 1_000_000)))
```

不需要改代码（已经支持环境变量）。**只需在 Exp 2/3 启动脚本里 export `LRU_CACHE_SIZE=10000000`**（1000 万条，按 1024×2 bytes 算约 20 GB 主机内存，单机 H100 节点足够）。

### 10.6 （可选）Qwen3 embedding 预计算脚本

为进一步降低 Exp 2/3 的训练成本，提供一个独立脚本 `scripts/precompute_qwen3_embeddings.py`：

- 扫描 T4 所有 parquet
- 抽出 unique 字符串（列名 + 所有 string-typed cell 值）
- 用 Qwen3 离线嵌入
- 存为 `<dataset>/qwen3_emb.parquet`，结构 `(text: str, embedding: list[float])`
- `Tokenizer` 启动时若发现该文件存在 → 预加载到 LRU cache

这一项**作为 Phase 2 优化**，可以先不做，等 Exp 2 跑起来后再决定是否需要。如果直接在线嵌入，配合 Task 10.5 的大 cache，应该也能跑得动 —— 只是冷启动几小时较慢。

### 10.7 验证

新建 `tests/test_qwen3_embedder.py`：

```python
import os
import pytest
import torch


@pytest.mark.skipif(not os.environ.get("RUN_QWEN3_TEST"),
                    reason="set RUN_QWEN3_TEST=1 to actually download Qwen3")
def test_qwen3_embed_shape():
    from sap_rpt_oss.data.sentence_embedder import SentenceEmbedder
    se = SentenceEmbedder("Qwen/Qwen3-Embedding-0.6B", device="cpu")
    emb = se.embed(["hello world", "tabular learning"])
    assert emb.shape == (2, 1024)


@pytest.mark.skipif(not os.environ.get("RUN_QWEN3_TEST"),
                    reason="set RUN_QWEN3_TEST=1 to actually download Qwen3")
def test_qwen3_rpt_construction():
    from sap_rpt_oss.data.tokenizer import Tokenizer
    from sap_rpt_oss.model.torch_model import RPT
    from sap_rpt_oss.constants import ModelSize

    tok = Tokenizer(sentence_embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
                    sentence_embedder_device="cpu")
    assert tok.embedding_dim == 1024

    m = RPT(model_size=ModelSize.mini, sentence_embedding_dim=tok.embedding_dim)
    # adapter 形状应为 (hidden=256, 1024)
    assert m.embeddings.column_remapping.weight.shape == (256, 1024)
    assert m.embeddings.content_remapping.weight.shape == (256, 1024)
```

下载 Qwen3 较大（~1.2 GB），所以测试默认跳过 —— 在 Claude Code 环境里跑一次确认即可，commit 时 skip。

---

# 实验编排

## Task 11 — 三个实验的统一启动入口

**目标**：让我能用一行命令启动任意一个实验，并自动从 HF Hub 下载官方 ckpt（仅 Exp 1）。

**改动位置**：新建 `sap_rpt_oss/run_experiment.py`，并修改 `pretrain_run.py::main`。

**要求**：

1. 新建 `run_experiment.py`：

```python
"""
统一的实验启动器。

用法：
    python -m sap_rpt_oss.run_experiment --exp 1   # MiniLM + FiLM (post-training)
    python -m sap_rpt_oss.run_experiment --exp 2   # Qwen3 + sum (from scratch)
    python -m sap_rpt_oss.run_experiment --exp 3   # Qwen3 + FiLM (from scratch)
"""
import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from sap_rpt_oss.configs import (
    get_exp1_minilm_film_config,
    get_exp2_qwen3_sum_config,
    get_exp3_qwen3_film_config,
)
from sap_rpt_oss.pretrain_run import (
    seed_everything,
    build_model_and_tokenizer,
    run_stage,
)


EXPERIMENTS = {
    1: ("Exp 1: MiniLM-L6 + FiLM (post-training)", get_exp1_minilm_film_config),
    2: ("Exp 2: Qwen3 + sum (from scratch)", get_exp2_qwen3_sum_config),
    3: ("Exp 3: Qwen3 + FiLM (from scratch)", get_exp3_qwen3_film_config),
}


def download_official_ckpt() -> Path:
    """下载 sap/sap-rpt-1-oss 官方 weight-shared 权重到本地缓存"""
    return Path(hf_hub_download(
        repo_id="sap/sap-rpt-1-oss",
        filename="model.pt",   # 实际文件名以 HF 仓库为准，请 Claude Code 验证
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="覆盖默认 max_steps，便于 dry-run")
    args = parser.parse_args()

    name, config_fn = EXPERIMENTS[args.exp]
    print(f"[run_experiment] Starting {name}")
    config = config_fn()
    if args.max_steps is not None:
        config = dataclasses.replace(config, max_steps=args.max_steps)

    seed_everything(config.random_seed)

    initial_checkpoint = None
    if not config.pretrain_from_scratch:
        # Exp 1: 从官方 ckpt post-training
        initial_checkpoint = download_official_ckpt()
        print(f"[run_experiment] Loading official ckpt from {initial_checkpoint}")

    model, tokenizer = build_model_and_tokenizer(config, checkpoint=initial_checkpoint)
    run_stage(
        config=config, model=model, tokenizer=tokenizer,
        data_root=config.data_root_path,
        output_root=config.output_root_path,
        max_num_rows=config.stage1_max_num_rows,
        max_steps=config.max_steps,
    )


if __name__ == "__main__":
    main()
```

2. **HF ckpt 文件名**：请 Claude Code 用 `huggingface_hub` 列一下 `sap/sap-rpt-1-oss` 仓库内的文件，确认要加载的是哪个 `.pt` / `.ckpt`，并改 `download_official_ckpt`。

3. `pretrain_run.py::main` 保留不动（兼容现有 `python -m sap_rpt_oss.pretrain_run` 用法），只是不再是主推荐入口。

4. 在 `README.md` 末尾追加 "Reproducing the ablation experiments" 一节，给出 3 条命令：

   ```bash
   # Exp 1: MiniLM-L6 + FiLM, post-training from official HF checkpoint
   python -m sap_rpt_oss.run_experiment --exp 1

   # Exp 2: Qwen3-Embedding-0.6B + sum, from scratch on T4
   LRU_CACHE_SIZE=10000000 python -m sap_rpt_oss.run_experiment --exp 2

   # Exp 3: Qwen3-Embedding-0.6B + FiLM, from scratch on T4
   LRU_CACHE_SIZE=10000000 python -m sap_rpt_oss.run_experiment --exp 3
   ```

**验证**：

```bash
# 不真跑训练，只确认 config 加载、模型构造、数据加载都没问题
python -m sap_rpt_oss.run_experiment --exp 1 --max-steps 10
python -m sap_rpt_oss.run_experiment --exp 2 --max-steps 10
python -m sap_rpt_oss.run_experiment --exp 3 --max-steps 10
```

每次都应：
- 构造好模型（Exp 2/3 应能看到 Qwen3 下载或缓存命中日志）
- 跑 10 步，loss 不 NaN
- 在 `outputs/exp{1,2,3}_*` 下生成 checkpoint

---

# 验证与交付

## Task 12 — 端到端验证

按 Task 11 的方法跑三组 dry-run（10 步）。再额外跑：

```bash
# 推理 demo 不挂（保护已发布 ckpt 加载兼容性）
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

# pytest
pytest tests/test_film_ablation.py -v
RUN_QWEN3_TEST=1 pytest tests/test_qwen3_embedder.py -v
```

参数量与 dim 检查：

```python
# 在 weight_sharing=True 默认下：
# Exp 1 base: ~16-30 M trainable
# Exp 2 base + Qwen3 1024 dim adapter: 略增（adapter 三层 (768, 1024) 各 ~0.8M → +2-3M）
# Exp 3 = Exp 2 + FiLM 一个 hidden→2*hidden 投影 ~1.2M
```

## Task 13 — 交付

完成所有 task 后：

1. `git diff` 存到 `reports/code_alignment_diff.patch`。
2. 新建 `reports/three_experiment_changelog.md`：
   - 每个 task 改了哪些文件、关键 diff 摘要、验证结果
   - 三个实验 config 的最终 diff
   - 三组 dry-run（`--max-steps 10`）的实际 loss/throughput 数据
3. 简短总结：
   - 论文对齐覆盖率（Task 1–8）
   - 三个 ablation 实验的训练入口是否就绪
   - 已知 caveat（特别是 Qwen3 训练成本估算、是否需要预计算优化）

---

# 重要约束（再次强调）

- ⚠️ **不动公共 API**：`RPT`、`Tokenizer`、`SAP_RPT_OSS_Classifier`、`SAP_RPT_OSS_Regressor` 的 `__init__` 默认参数若变化，须保证默认行为完全等价旧版。
- ⚠️ **不动推理 demo**：`examples/` 下、README 里的 demo 必须保持原命令可跑通（Exp 1/2/3 都不能破坏 ckpt 推理路径）。
- ⚠️ **Qwen3 ckpt 缓存**：第一次跑 Qwen3 测试时会下载 ~1.2 GB；请确认 `huggingface_hub` 缓存路径有空间；commit 时不带这些权重。
- ⚠️ **Exp 2/3 启动前**：`LRU_CACHE_SIZE=10000000` 必须 export，否则 cache miss 会让 Qwen3 嵌入成为瓶颈。
- ⚠️ **Task 6 的混采样默认关闭**，三个实验默认都不启用 stage2。
- 📄 完成后请把 `reports/pretraining_alignment_report.md` 第 9 节"关键偏离汇总"中已修复的项目划掉或标 ✅。
