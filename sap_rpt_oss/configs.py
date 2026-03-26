from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sap_rpt_oss.constants import ModelSize


@dataclass(slots=True)
class FinetuneConfig:
    data_root_path: Path = Path("datasets/t4/data_d")
    output_root_path: Path = Path("outputs/finetune")
    pretrained_checkpoint: str = "2025-11-04_sap-rpt-one-oss.pt"
    resume_checkpoint_path: Path | None = None
    checkpoint_root_path: Path | None = None
    checkpoint_save_every_n_train_steps: int = 100

    model_size: ModelSize = ModelSize.base
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    training: bool = True

    # 手写 PyTorch 训练循环按 step 停止；max_epochs 保留仅为兼容旧配置，不参与停止条件。
    max_steps: int = 4_000_000
    max_epochs: int = 5
    micro_batch_size: int = 1
    # DataLoader worker 数；这个项目的数据预处理包含文本 embedding，默认用 0 更稳。
    num_workers: int = 0
    # 有效 batch size = micro_batch_size * accumulate_grad_batches。
    # 训练入口当前固定要求 micro_batch_size == 1，因此这里就是直接控制有效 batch size。
    accumulate_grad_batches: int | None = None
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 50

    # 按论文描述整理的数据采样参数。
    min_num_rows: int = 150
    query_size_range: tuple[int, int] = (50, 900)
    # 跳过特征列数量超过该阈值的表；不包含 target 列。
    max_num_features: int | None = 50
    target_column: str | None = None
    predict_chunk_size: int | None = None
    # 流式读取 parquet 时单次最多拉取的行数；为空时按采样规模自动推断。
    streaming_read_batch_size: int | None = None
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
