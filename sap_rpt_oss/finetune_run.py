import random
from contextlib import nullcontext
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sap_rpt_oss.configs import FINETUNE_CONFIG, FinetuneConfig
from sap_rpt_oss.data.ds import RPTParquetDataset
from sap_rpt_oss.data.tokenizer import Tokenizer
from sap_rpt_oss.model.torch_model import RPT


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_autocast_dtype() -> Optional[torch.dtype]:
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability(0)
    return torch.bfloat16 if major >= 8 else torch.float16


def resolve_checkpoint(
    checkpoint: Optional[Union[str, Path]],
) -> Optional[Path]:
    if checkpoint is None:
        return None

    checkpoint_path = Path(checkpoint).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path.resolve()

    return Path(
        hf_hub_download(repo_id="SAP/sap-rpt-1-oss", filename=str(checkpoint))
    )


def move_to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def build_dataloader(
    config: FinetuneConfig,
    tokenizer: Tokenizer,
    data_root: Path,
    max_num_rows: int,
) -> DataLoader:
    dataset = RPTParquetDataset(
        root_dir=data_root,
        fit_size=None,
        tokenizer=tokenizer,
        target_column=config.target_column,
        predict_chunk_size=config.predict_chunk_size,
        shuffle_table=config.shuffle_table,
        drop_constant_columns=True,
        max_num_columns=500,
        max_num_features=config.max_num_features,
        min_num_rows=config.min_num_rows,
        max_num_rows=max_num_rows,
        query_size_range=config.query_size_range,
        auto_select_target=config.auto_select_target,
        skip_ineligible_target=config.skip_ineligible_target,
        numeric_nan_ratio_threshold=config.numeric_nan_ratio_threshold,
        categorical_unique_ratio_threshold=config.categorical_unique_ratio_threshold,
        balance_classification_tasks=config.balance_classification_tasks,
        random_seed=config.random_seed,
        regression_keyword=config.regression_keyword,
        streaming_read_batch_size=config.streaming_read_batch_size,
    )

    dataloader_kwargs = {
        "batch_size": None,
        "num_workers": config.num_workers,
    }
    if config.num_workers > 0:
        dataloader_kwargs["persistent_workers"] = True
    return DataLoader(dataset, **dataloader_kwargs)


def build_model_and_tokenizer(
    config: FinetuneConfig,
    checkpoint: Optional[Path],
):
    model = RPT(
        model_size=config.model_size,
        regression_type="l2",
        classification_type="cross-entropy",
    )
    if checkpoint is not None:
        model.load_weights(checkpoint, device=torch.device("cpu"))

    tokenizer_device = (
        "cpu"
        if config.num_workers > 0
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = Tokenizer(
        regression_type="l2",
        classification_type="cross-entropy",
        random_seed=config.random_seed,
        num_regression_bins=16,
        is_valid=True,
        sentence_embedder_device=tokenizer_device,
        verbose=False,
    )
    return model, tokenizer


def build_optimizer_and_scheduler(
    model: RPT,
    config: FinetuneConfig,
):
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    if config.warmup_steps <= 0:
        return optimizer, None

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(float(step + 1) / float(config.warmup_steps), 1.0),
    )
    return optimizer, scheduler


def save_model_weights(
    model: RPT,
    checkpoint_dir: Path,
    stage_name: str,
    step: int,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{stage_name}-{step}-step.pt"
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, checkpoint_path)
    return checkpoint_path


def run_stage(
    config: FinetuneConfig,
    model: RPT,
    tokenizer: Tokenizer,
    data_root: Path,
    output_root: Path,
    max_num_rows: int,
    max_steps: int,
    max_epochs: int,
):
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = infer_autocast_dtype()
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and autocast_dtype == torch.float16
    )

    dataloader = build_dataloader(
        config=config,
        tokenizer=tokenizer,
        data_root=data_root,
        max_num_rows=max_num_rows,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    checkpoint_dir = config.resolved_checkpoint_dir
    stage_name = output_root.name.replace("/", "_")
    progress = tqdm(total=max_steps, desc=stage_name, unit="step")

    global_step = 0
    pending_batches = 0
    last_saved_step = 0
    accumulation_steps = config.resolved_accumulate_grad_batches
    last_loss = 0.0
    last_metric = 0.0

    def finish_optimizer_step():
        nonlocal global_step
        nonlocal last_saved_step
        nonlocal pending_batches

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

        global_step += 1
        pending_batches = 0
        progress.update(1)
        progress.set_postfix(
            loss=f"{last_loss:.4f}",
            metric=f"{last_metric:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}",
        )

        if global_step % config.checkpoint_save_every_n_train_steps == 0:
            save_model_weights(model, checkpoint_dir, stage_name, global_step)
            last_saved_step = global_step

    for epoch_idx in range(max_epochs):
        if global_step >= max_steps:
            break

        progress.set_description(f"{stage_name} epoch {epoch_idx + 1}/{max_epochs}")
        for batch in dataloader:
            if global_step >= max_steps:
                break

            is_regression = batch["is_regression"]
            if isinstance(is_regression, torch.Tensor):
                is_regression = bool(is_regression.item())
            else:
                is_regression = bool(is_regression)
            batch = move_to_device(batch, device)
            autocast_context = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if device.type == "cuda" and autocast_dtype is not None
                else nullcontext()
            )

            with autocast_context:
                _, loss, metric = model(
                    batch["data"],
                    is_regression=is_regression,
                    labels=batch["labels"],
                )
                scaled_loss = loss / accumulation_steps

            last_loss = float(loss.detach().cpu())
            last_metric = float(metric.detach().cpu())
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            pending_batches += 1
            if pending_batches >= accumulation_steps:
                finish_optimizer_step()

        if 0 < pending_batches and global_step < max_steps:
            finish_optimizer_step()

    if global_step != last_saved_step:
        save_model_weights(model, checkpoint_dir, stage_name, global_step)

    progress.close()


def main():
    config = FINETUNE_CONFIG
    seed_everything(config.random_seed)

    if config.micro_batch_size != 1:
        raise ValueError("This training pipeline assumes a micro batch size of 1")

    initial_checkpoint = config.resolved_resume_checkpoint_path
    if initial_checkpoint is None:
        initial_checkpoint = resolve_checkpoint(config.pretrained_checkpoint)

    model, tokenizer = build_model_and_tokenizer(
        config=config,
        checkpoint=initial_checkpoint,
    )

    run_stage(
        config=config,
        model=model,
        tokenizer=tokenizer,
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
            tokenizer=tokenizer,
            data_root=Path(config.curriculum_stage2_data_root_path),
            output_root=config.curriculum_stage2_output_root_path,
            max_num_rows=config.curriculum_stage2_max_num_rows,
            max_steps=config.curriculum_stage2_max_steps,
            max_epochs=config.curriculum_stage2_max_epochs,
        )


if __name__ == "__main__":
    main()
