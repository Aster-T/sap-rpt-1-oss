from pathlib import Path

import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from sap_rpt_oss.configs import FINETUNE_CONFIG, FinetuneConfig
from sap_rpt_oss.model.lightning_model import LightningModelRPT


def infer_precision() -> str:
    if not torch.cuda.is_available():
        return "32-true"
    major, _ = torch.cuda.get_device_capability(0)
    return "bf16-mixed" if major >= 8 else "16-mixed"


def build_trainer(
    config: FinetuneConfig,
    output_root: Path,
    max_steps: int,
    max_epochs: int,
) -> Trainer:
    checkpoint_dir = config.resolved_checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="{step}-step",
        auto_insert_metric_name=False,
        save_last=False,
        save_top_k=-1,
        every_n_train_steps=config.checkpoint_save_every_n_train_steps,
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
        accumulate_grad_batches=config.accumulate_grad_batches,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
    )


def run_stage(
    config: FinetuneConfig,
    model: LightningModelRPT,
    data_root: Path,
    output_root: Path,
    max_num_rows: int,
    max_steps: int,
    max_epochs: int,
):
    output_root.mkdir(parents=True, exist_ok=True)
    model.set_training_data_from_root(
        root_dir=data_root,
        fit_size=None,
        target_column=config.target_column,
        predict_chunk_size=config.predict_chunk_size,
        shuffle_table=config.shuffle_table,
        regression_keyword=config.regression_keyword,
        min_num_rows=config.min_num_rows,
        max_num_rows=max_num_rows,
        query_size_range=config.query_size_range,
        auto_select_target=config.auto_select_target,
        skip_ineligible_target=config.skip_ineligible_target,
        numeric_nan_ratio_threshold=config.numeric_nan_ratio_threshold,
        categorical_unique_ratio_threshold=config.categorical_unique_ratio_threshold,
        balance_classification_tasks=config.balance_classification_tasks,
    )
    trainer = build_trainer(
        config=config,
        output_root=output_root,
        max_steps=max_steps,
        max_epochs=max_epochs,
    )
    trainer.fit(model)


def main():
    config = FINETUNE_CONFIG
    seed_everything(config.random_seed, workers=True)

    if config.micro_batch_size != 1:
        raise ValueError("This training pipeline assumes a micro batch size of 1")

    model = LightningModelRPT(
        model_size=config.model_size,
        checkpoint=config.pretrained_checkpoint,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        random_seed=config.random_seed,
    )

    run_stage(
        config=config,
        model=model,
        data_root=config.data_root_path,
        output_root=config.output_root_path,
        max_num_rows=config.stage1_max_num_rows,
        max_steps=config.max_steps,
        max_epochs=config.max_epochs,
    )

    if config.use_curriculum_stage2:
        run_stage(
            config=config,
            model=model,
            data_root=Path(config.curriculum_stage2_data_root_path),
            output_root=config.curriculum_stage2_output_root_path,
            max_num_rows=config.curriculum_stage2_max_num_rows,
            max_steps=config.curriculum_stage2_max_steps,
            max_epochs=config.curriculum_stage2_max_epochs,
        )


if __name__ == "__main__":
    main()
