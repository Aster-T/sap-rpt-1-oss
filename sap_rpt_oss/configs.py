import dataclasses
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sap_rpt_oss.constants import (
    ModelSize,
    embedding_model_to_dimension_and_pooling,
)


# Task 8 uses dataclasses.replace on FinetuneConfig; slots=True can interact
# poorly with replace + nested default_factory in some configurations, so we
# keep the dataclasses without slots for forward-compatibility.
@dataclass
class TableRulesConfig:
    """Rules for table filtering, feature pruning, and target selection."""

    # 去掉唯一值列
    drop_constant_columns: bool = True
    # 最大特征数（与推理侧 MAX_NUM_COLUMNS 对齐，论文 Section 4.1 末段）
    max_num_columns: int = 500
    # 最小行数
    min_num_rows: int = 150
    # 缺失值超过这个比例的数值列跳过
    numeric_nan_ratio_threshold: float = 0.5
    # 非数值列中唯一值超过这个比例的列跳过
    categorical_unique_ratio_threshold: float = 0.2

    @property
    def max_num_features(self) -> int:
        return max(0, self.max_num_columns - 1)


@dataclass
class FinetuneConfig:
    data_root_path: Path = Path("datasets/t4/datas")
    output_root_path: Path = Path("outputs/finetune")
    # True: initialize model weights from scratch. False: load from resume_checkpoint_path.
    pretrain_from_scratch: bool = False
    # Checkpoint path used only when pretrain_from_scratch is False.
    resume_checkpoint_path: Path | None = None
    checkpoint_root_path: Path | None = None
    checkpoint_save_every_n_train_steps: int = 100

    model_size: ModelSize = ModelSize.base
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    training: bool = True

    # Paper-style pretraining typically runs for 4M to 10M updates.
    # This default uses the lower bound; raise it if you want the longer schedule.
    max_steps: int = 8_000_000
    max_epochs: int = 5
    micro_batch_size: int = 1
    num_workers: int = 0
    accumulate_grad_batches: int | None = None
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50

    # Architecture toggles (paper-aligned defaults).
    weight_sharing: bool = True
    combination_type: str = "sum"  # "sum" (paper) or "film" (Exp 1/3 ablation)
    use_weekday: bool = False  # paper does not use weekday in date encoding
    sentence_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Table filtering and target-selection rules aligned with the pretraining setup.
    table_rules: TableRulesConfig = field(
        default_factory=lambda: TableRulesConfig(
            min_num_rows=150,
            max_num_columns=500,
        )
    )

    # Table sampling and loading.
    query_size_range: tuple[int, int] = (50, 900)
    target_column: str | None = None
    predict_chunk_size: int | None = None
    streaming_read_batch_size: int | None = None
    shuffle_table: bool = True
    regression_keyword: str = "regression"
    random_seed: int = 42
    auto_select_target: bool = True
    skip_ineligible_target: bool = True
    balance_classification_tasks: bool = True

    # Stage 1.
    stage1_max_num_rows: int = 1000

    # Optional curriculum stage 2.
    curriculum_stage2_data_root_path: Path | None = None
    # If set, stage 2 mixes T4 (curriculum_stage2_t4_data_root_path) and the Ma et al.
    # data via Bernoulli sampling with p=curriculum_stage2_mixing_ratio (T4 share).
    # When None, stage 2 falls back to a single source (curriculum_stage2_data_root_path).
    curriculum_stage2_t4_data_root_path: Path | None = None
    curriculum_stage2_mixing_ratio: float = 0.8
    curriculum_stage2_output_root_path: Path = Path(
        "outputs/finetune/curriculum_stage2"
    )
    curriculum_stage2_max_num_rows: int = 4000
    curriculum_stage2_max_steps: int = 4_000_000
    curriculum_stage2_max_epochs: int = 5

    @property
    def resolved_accumulate_grad_batches(self) -> int:
        if self.accumulate_grad_batches is not None:
            if self.accumulate_grad_batches <= 0:
                raise ValueError("accumulate_grad_batches must be a positive integer")
            return self.accumulate_grad_batches
        return 128 if self.model_size == ModelSize.mini else 256

    @property
    def resolved_checkpoint_dir(self) -> Path:
        if self.checkpoint_root_path is not None:
            return Path(self.checkpoint_root_path)
        return Path("checkpoints") / date.today().isoformat()

    @property
    def resolved_resume_checkpoint_path(self) -> Path | None:
        if self.resume_checkpoint_path is None:
            return None

        checkpoint_path = Path(self.resume_checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path.resolve()

    @property
    def use_curriculum_stage2(self) -> bool:
        return self.curriculum_stage2_data_root_path is not None

    @property
    def sentence_embedding_dim(self) -> int:
        return embedding_model_to_dimension_and_pooling[
            self.sentence_embedding_model_name
        ][0]


@dataclass
class InferenceConfigs:
    checkpoints_path: Path = Path("checkpoints")
    input_root_path: Path = Path("datasets")
    output_root_path: Path = Path("results")


FINETUNE_CONFIG = FinetuneConfig()
INFERENCE_CONFIG = InferenceConfigs()


# ----------------------------------------------------------------------------
# Task 8 — three-experiment ablation factories
# ----------------------------------------------------------------------------
def get_paper_aligned_pretrain_config() -> FinetuneConfig:
    """ConTextTab paper base configuration: MiniLM-L6 + sum + from-scratch."""
    return FinetuneConfig(
        pretrain_from_scratch=True,
        model_size=ModelSize.base,
        weight_sharing=True,
        sentence_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        combination_type="sum",
        use_weekday=False,
        learning_rate=1e-4,
        warmup_steps=1000,
        gradient_clip_val=1.0,
        max_steps=8_000_000,
        accumulate_grad_batches=256,
        stage1_max_num_rows=1000,
        query_size_range=(50, 900),
        balance_classification_tasks=True,
        table_rules=TableRulesConfig(
            min_num_rows=150,
            max_num_columns=500,
            numeric_nan_ratio_threshold=0.5,
            categorical_unique_ratio_threshold=0.2,
            drop_constant_columns=True,
        ),
    )


def get_exp1_minilm_film_config() -> FinetuneConfig:
    """Exp 1: MiniLM-L6 + FiLM, post-training from official HF checkpoint."""
    cfg = get_paper_aligned_pretrain_config()
    return dataclasses.replace(
        cfg,
        pretrain_from_scratch=False,
        resume_checkpoint_path=None,  # downloaded inside run_experiment.py
        combination_type="film",
        max_steps=200_000,
        warmup_steps=2_000,
        learning_rate=1e-4,
        output_root_path=Path("outputs/exp1_minilm_film"),
        checkpoint_root_path=Path("checkpoints/exp1_minilm_film"),
        stage1_max_num_rows=1000,
        query_size_range=(50, 900),
        table_rules=TableRulesConfig(
            min_num_rows=150,
            max_num_columns=50,
        ),
    )


def get_exp2_qwen3_sum_config() -> FinetuneConfig:
    """Exp 2: Qwen3-Embedding-0.6B + sum, T4 from-scratch pretraining."""
    cfg = get_paper_aligned_pretrain_config()
    return dataclasses.replace(
        cfg,
        sentence_embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        combination_type="sum",
        output_root_path=Path("outputs/exp2_qwen3_sum"),
        checkpoint_root_path=Path("checkpoints/exp2_qwen3_sum"),
    )


def get_exp3_qwen3_film_config() -> FinetuneConfig:
    """Exp 3: Qwen3-Embedding-0.6B + FiLM, T4 from-scratch pretraining."""
    return dataclasses.replace(
        get_exp2_qwen3_sum_config(),
        combination_type="film",
        output_root_path=Path("outputs/exp3_qwen3_film"),
        checkpoint_root_path=Path("checkpoints/exp3_qwen3_film"),
    )
