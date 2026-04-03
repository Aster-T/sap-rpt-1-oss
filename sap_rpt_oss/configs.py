from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from sap_rpt_oss.constants import ModelSize


@dataclass(slots=True)
class TableRulesConfig:
    """Rules for table filtering, feature pruning, and target selection."""

    # 去掉唯一值列
    drop_constant_columns: bool = True
    # 最大特征数
    max_num_columns: int = 50
    # 最小行数
    min_num_rows: int = 150
    # 缺失值超过这个比例的数值列跳过
    numeric_nan_ratio_threshold: float = 0.5
    # 非数值列中唯一值超过这个比例的列跳过
    categorical_unique_ratio_threshold: float = 0.2

    @property
    def max_num_features(self) -> int:
        return max(0, self.max_num_columns - 1)


@dataclass(slots=True)
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

    # Table filtering and target-selection rules aligned with the pretraining setup.
    table_rules: TableRulesConfig = field(
        default_factory=lambda: TableRulesConfig(
            min_num_rows=150,
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


@dataclass(slots=True)
class InferenceConfigs:
    checkpoints_path: Path = Path("checkpoints")
    input_root_path: Path = Path("datasets")
    output_root_path: Path = Path("results")


FINETUNE_CONFIG = FinetuneConfig()
INFERENCE_CONFIG = InferenceConfigs()
