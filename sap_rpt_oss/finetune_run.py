from datetime import date
from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sap_rpt_oss.constants import ModelSize
from sap_rpt_oss.model.lightning_model import LightningModel

DATA_ROOT_PATH = Path("datasets")
OUTPUT_ROOT_PATH = Path("outputs/finetune")
PRETRAINED_CHECKPOINT = "2025-11-04_sap-rpt-one-oss.pt"
CHECKPOINT_ROOT_PATH = None
CHECKPOINT_SAVE_EVERY_N_TRAIN_STEPS = 100

MODEL_SIZE = ModelSize.base
LEARNING_RATE = 1e-4
WARMUP_STEPS = 1000

# The paper reports 4M to 10M steps, roughly 2 to 5 epochs.
MAX_STEPS = 4_000_000
MAX_EPOCHS = 5

# Each optimizer step consumes one table/micro-batch. Gradient accumulation
# simulates the larger effective batch size described in the paper.
MICRO_BATCH_SIZE = 1
ACCUMULATE_GRAD_BATCHES = 128 if MODEL_SIZE == ModelSize.mini else 256

# The paper mentions gradient clipping but does not specify the exact norm.
GRADIENT_CLIP_VAL = 1.0

# Data sampling configuration adapted from the paper.
MIN_NUM_ROWS = 150
QUERY_SIZE_RANGE = (50, 900)
TARGET_COLUMN = None
PREDICT_CHUNK_SIZE = None
SHUFFLE_TABLE = True
REGRESSION_KEYWORD = "regression"
RANDOM_SEED = 42
# 控制自动选择目标列
AUTO_SELECT_TARGET = False
# 控制是否跳过不合格的目标列（对齐论文）
SKIP_INELIGIBLE_TARGET = True
NUMERIC_NAN_RATIO_THRESHOLD = 0.5
CATEGORICAL_UNIQUE_RATIO_THRESHOLD = 0.2
BALANCE_CLASSIFICATION_TASKS = True

# Stage 1: default T4-style pretraining setup with up to 1000 rows.
STAGE1_MAX_NUM_ROWS = 1000

# Optional curriculum stage: point to another dataset root to enable it.
CURRICULUM_STAGE2_DATA_ROOT_PATH = None
CURRICULUM_STAGE2_OUTPUT_ROOT_PATH = OUTPUT_ROOT_PATH / "curriculum_stage2"
CURRICULUM_STAGE2_MAX_NUM_ROWS = 4000
CURRICULUM_STAGE2_MAX_STEPS = MAX_STEPS
CURRICULUM_STAGE2_MAX_EPOCHS = MAX_EPOCHS


def infer_precision() -> str:
    if not torch.cuda.is_available():
        return "32-true"
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16-mixed" if major >= 8 else "16-mixed"


def build_trainer(
    output_root: Path,
    max_steps: int,
    max_epochs: int,
    checkpoint_root: Path | None = None,
) -> Trainer:
    checkpoint_dir = (
        Path(checkpoint_root)
        if checkpoint_root is not None
        else Path("checkpoints") / date.today().isoformat()
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}-step",
        auto_insert_metric_name=False,
        save_last=False,
        save_top_k=-1,
        every_n_train_steps=CHECKPOINT_SAVE_EVERY_N_TRAIN_STEPS,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(save_dir=str(output_root), name="logs")

    return Trainer(
        default_root_dir=str(output_root),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=infer_precision(),
        max_steps=max_steps,
        max_epochs=max_epochs,
        accumulate_grad_batches=ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        gradient_clip_algorithm="norm",
        log_every_n_steps=50,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )


def run_stage(
    model: LightningModel,
    data_root: Path,
    output_root: Path,
    max_num_rows: int,
    max_steps: int,
    max_epochs: int,
    checkpoint_root: Path | None = None,
):
    output_root.mkdir(parents=True, exist_ok=True)
    model.set_training_data_from_root(
        root_dir=data_root,
        fit_size=None,
        target_column=TARGET_COLUMN,
        predict_chunk_size=PREDICT_CHUNK_SIZE,
        shuffle_table=SHUFFLE_TABLE,
        regression_keyword=REGRESSION_KEYWORD,
        min_num_rows=MIN_NUM_ROWS,
        max_num_rows=max_num_rows,
        query_size_range=QUERY_SIZE_RANGE,
        auto_select_target=AUTO_SELECT_TARGET,
        skip_ineligible_target=SKIP_INELIGIBLE_TARGET,
        numeric_nan_ratio_threshold=NUMERIC_NAN_RATIO_THRESHOLD,
        categorical_unique_ratio_threshold=CATEGORICAL_UNIQUE_RATIO_THRESHOLD,
        balance_classification_tasks=BALANCE_CLASSIFICATION_TASKS,
    )
    trainer = build_trainer(
        output_root=output_root,
        max_steps=max_steps,
        max_epochs=max_epochs,
        checkpoint_root=checkpoint_root,
    )
    trainer.fit(model)


def main():
    seed_everything(RANDOM_SEED, workers=True)

    if MICRO_BATCH_SIZE != 1:
        raise ValueError("This training pipeline assumes a micro batch size of 1")

    model = LightningModel(
        model_size=MODEL_SIZE,
        checkpoint=PRETRAINED_CHECKPOINT,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        random_seed=RANDOM_SEED,
    )

    run_stage(
        model=model,
        data_root=DATA_ROOT_PATH,
        output_root=OUTPUT_ROOT_PATH,
        max_num_rows=STAGE1_MAX_NUM_ROWS,
        max_steps=MAX_STEPS,
        max_epochs=MAX_EPOCHS,
        checkpoint_root=CHECKPOINT_ROOT_PATH,
    )

    if CURRICULUM_STAGE2_DATA_ROOT_PATH is not None:
        run_stage(
            model=model,
            data_root=Path(CURRICULUM_STAGE2_DATA_ROOT_PATH),
            output_root=CURRICULUM_STAGE2_OUTPUT_ROOT_PATH,
            max_num_rows=CURRICULUM_STAGE2_MAX_NUM_ROWS,
            max_steps=CURRICULUM_STAGE2_MAX_STEPS,
            max_epochs=CURRICULUM_STAGE2_MAX_EPOCHS,
            checkpoint_root=CHECKPOINT_ROOT_PATH,
        )


if __name__ == "__main__":
    main()
