from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sap_rpt_oss.constants import ModelSize


@dataclass(slots=True)
class FinetuneConfig:
    data_root_path: Path = Path("datasets")
    output_root_path: Path = Path("outputs/finetune")
    pretrained_checkpoint: str = "2025-11-04_sap-rpt-one-oss.pt"
    checkpoint_root_path: Path | None = None
    checkpoint_save_every_n_train_steps: int = 100

    model_size: ModelSize = ModelSize.base
    learning_rate: float = 1e-4
    warmup_steps: int = 1000

    # 论文中给出的训练规模是 400 万到 1000 万步，大致对应 2 到 5 轮训练。
    max_steps: int = 4_000_000
    max_epochs: int = 5
    micro_batch_size: int = 1
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50

    # 按论文描述整理的数据采样参数。
    min_num_rows: int = 150
    query_size_range: tuple[int, int] = (50, 900)
    target_column: str | None = None
    predict_chunk_size: int | None = None
    shuffle_table: bool = True
    regression_keyword: str = "regression"
    random_seed: int = 42
    auto_select_target: bool = False
    skip_ineligible_target: bool = True
    numeric_nan_ratio_threshold: float = 0.5
    categorical_unique_ratio_threshold: float = 0.2
    balance_classification_tasks: bool = True

    # 第一阶段：默认的 T4 预训练设置，单表最多采样 1000 行。
    stage1_max_num_rows: int = 1000

    # 可选的课程学习第二阶段配置。
    curriculum_stage2_data_root_path: Path | None = None
    curriculum_stage2_output_root_path: Path = Path(
        "outputs/finetune/curriculum_stage2"
    )
    curriculum_stage2_max_num_rows: int = 4000
    curriculum_stage2_max_steps: int = 4_000_000
    curriculum_stage2_max_epochs: int = 5

    @property
    def accumulate_grad_batches(self) -> int:
        return 128 if self.model_size == ModelSize.mini else 256

    @property
    def resolved_checkpoint_dir(self) -> Path:
        if self.checkpoint_root_path is not None:
            return Path(self.checkpoint_root_path)
        return Path("checkpoints") / date.today().isoformat()

    @property
    def use_curriculum_stage2(self) -> bool:
        return self.curriculum_stage2_data_root_path is not None


FINETUNE_CONFIG = FinetuneConfig()
