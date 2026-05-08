import logging
import random
import re
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sap_rpt_oss.configs import FINETUNE_CONFIG, FinetuneConfig
from sap_rpt_oss.data.ds import MixedRPTDataset, RPTParquetDataset
from sap_rpt_oss.data.tokenizer import Tokenizer
from sap_rpt_oss.model.torch_model import RPT


class _TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            tqdm.write(self.format(record))
            self.flush()
        except Exception:
            self.handleError(record)


def _setup_train_logger(log_dir: Path, stage_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stage_name}_{timestamp}.log"

    logger = logging.getLogger(f"sap_rpt_oss.train.{stage_name}.{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = _TqdmLoggingHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f"Writing training log to {log_path}")
    return logger


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


def move_to_device(x, device: torch.device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move_to_device(v, device) for k, v in x.items()}
    return x


def initialize_model_weights(model: RPT):
    def reset_module_parameters(module):
        reset_parameters = getattr(module, "reset_parameters", None)
        if callable(reset_parameters):
            reset_parameters()

    model.apply(reset_module_parameters)


def _build_single_dataset(
    config: FinetuneConfig,
    tokenizer: Tokenizer,
    data_root: Path,
    max_num_rows: int,
) -> RPTParquetDataset:
    return RPTParquetDataset(
        root_dir=data_root,
        fit_size=None,
        tokenizer=tokenizer,
        target_column=config.target_column,
        predict_chunk_size=config.predict_chunk_size,
        shuffle_table=config.shuffle_table,
        max_num_rows=max_num_rows,
        query_size_range=config.query_size_range,
        balance_classification_tasks=config.balance_classification_tasks,
        random_seed=config.random_seed,
        streaming_read_batch_size=config.streaming_read_batch_size,
        table_rules=config.table_rules,
    )


def build_dataloader(
    config: FinetuneConfig,
    tokenizer: Tokenizer,
    data_root: Path,
    max_num_rows: int,
    secondary_data_root: Optional[Path] = None,
    mixing_ratio: float = 0.8,
) -> DataLoader:
    dataset = _build_single_dataset(config, tokenizer, data_root, max_num_rows)
    if secondary_data_root is not None:
        secondary = _build_single_dataset(
            config, tokenizer, secondary_data_root, max_num_rows
        )
        dataset = MixedRPTDataset(
            primary=dataset,
            secondary=secondary,
            mixing_ratio=mixing_ratio,
            random_seed=config.random_seed,
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
        clip_quantile=0.02,  # paper Section 3.1 — pretraining uses 2/98 quantile clipping
        sentence_embedding_model_name=config.sentence_embedding_model_name,
        sentence_embedder_device=tokenizer_device,
        verbose=False,
    )

    model = RPT(
        model_size=config.model_size,
        regression_type="l2",
        classification_type="cross-entropy",
        weight_sharing=config.weight_sharing,
        combination_type=config.combination_type,
        use_weekday=config.use_weekday,
        sentence_embedding_dim=tokenizer.embedding_dim,
        verbose=True,
    )
    if config.pretrain_from_scratch:
        initialize_model_weights(model)
    elif checkpoint is not None:
        model.load_weights(checkpoint, device=torch.device("cpu"))
    else:
        raise ValueError(
            "resume_checkpoint_path must be set when pretrain_from_scratch is False"
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


_CHECKPOINT_FILE_PATTERN = re.compile(r"^(?P<stage>.+?)-(?P<step>\d+)-step\.pt$")


def find_latest_checkpoint(
    checkpoint_dir: Path, stage_name: Optional[str] = None
) -> tuple[Optional[Path], int]:
    """Return the (path, step) pair of the highest-step `*-N-step.pt` in
    `checkpoint_dir`, or (None, 0) if no checkpoint exists. If `stage_name` is
    given, only files whose prefix matches are considered (so stage-1 and
    stage-2 ckpts written to the same dir don't get mixed up)."""
    if not checkpoint_dir.exists():
        return None, 0

    best_path: Optional[Path] = None
    best_step = -1
    for entry in checkpoint_dir.iterdir():
        if not entry.is_file():
            continue
        match = _CHECKPOINT_FILE_PATTERN.match(entry.name)
        if not match:
            continue
        if stage_name is not None and match.group("stage") != stage_name:
            continue
        step = int(match.group("step"))
        if step > best_step:
            best_step = step
            best_path = entry
    if best_path is None:
        return None, 0
    return best_path, best_step


def run_stage(
    config: FinetuneConfig,
    model: RPT,
    tokenizer: Tokenizer,
    data_root: Path,
    output_root: Path,
    max_num_rows: int,
    max_steps: int,
    secondary_data_root: Optional[Path] = None,
    mixing_ratio: float = 0.8,
    start_step: int = 0,
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
        secondary_data_root=secondary_data_root,
        mixing_ratio=mixing_ratio,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)
    # Fast-forward LR scheduler so warmup/decay state matches resumed step.
    if scheduler is not None and start_step > 0:
        for _ in range(start_step):
            scheduler.step()
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    checkpoint_dir = config.resolved_checkpoint_dir
    stage_name = output_root.name.replace("/", "_")
    logger = _setup_train_logger(output_root, stage_name)
    logger.info(
        f"stage={stage_name} start_step={start_step} max_steps={max_steps} "
        f"accumulate_grad_batches={config.resolved_accumulate_grad_batches} "
        f"log_every_n_steps={config.log_every_n_steps}"
    )
    progress = tqdm(
        total=max_steps, desc=stage_name, unit="step", initial=start_step
    )

    global_step = start_step
    pending_batches = 0
    last_saved_step = start_step
    accumulation_steps = config.resolved_accumulate_grad_batches
    last_loss = 0.0
    last_metric = 0.0
    reg_batch_count = 0
    cls_batch_count = 0

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

        if global_step % config.log_every_n_steps == 0:
            total_batches = reg_batch_count + cls_batch_count
            cls_ratio = (
                cls_batch_count / total_batches if total_batches > 0 else 0.0
            )
            logger.info(
                f"step={global_step} loss={last_loss:.4f} "
                f"metric={last_metric:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e} "
                f"reg_batches={reg_batch_count} cls_batches={cls_batch_count} "
                f"cls_ratio={cls_ratio:.3f}"
            )

        if global_step % config.checkpoint_save_every_n_train_steps == 0:
            save_model_weights(model, checkpoint_dir, stage_name, global_step)
            last_saved_step = global_step
            logger.info(f"checkpoint saved at step={global_step}")

    while global_step < max_steps:
        yielded_batch = False
        for batch in dataloader:
            yielded_batch = True
            if global_step >= max_steps:
                break

            is_regression = batch["is_regression"]
            if isinstance(is_regression, torch.Tensor):
                is_regression = bool(is_regression.item())
            else:
                is_regression = bool(is_regression)
            if is_regression:
                reg_batch_count += 1
            else:
                cls_batch_count += 1
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
        if not yielded_batch:
            raise RuntimeError(
                f"No batches were yielded from training data root: {data_root}"
            )

    if global_step != last_saved_step:
        save_model_weights(model, checkpoint_dir, stage_name, global_step)
        logger.info(f"checkpoint saved at step={global_step} (final)")

    total_batches = reg_batch_count + cls_batch_count
    cls_ratio = cls_batch_count / total_batches if total_batches > 0 else 0.0
    logger.info(
        f"stage={stage_name} finished: total_steps={global_step} "
        f"reg_batches={reg_batch_count} cls_batches={cls_batch_count} "
        f"cls_ratio={cls_ratio:.3f}"
    )
    progress.close()


def main():
    config = FINETUNE_CONFIG
    seed_everything(config.random_seed)

    if config.micro_batch_size != 1:
        raise ValueError("This training pipeline assumes a micro batch size of 1")

    initial_checkpoint = None
    if not config.pretrain_from_scratch:
        initial_checkpoint = config.resolved_resume_checkpoint_path

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
    )

    if config.use_curriculum_stage2:
        # Paper Appendix A.4 mixes T4 (80%) with the Ma et al. data (20%) in
        # stage 2. Primary stream is T4 (defaults to the stage-1 data root if
        # curriculum_stage2_t4_data_root_path is unset), secondary is Ma et al.
        # (curriculum_stage2_data_root_path).
        t4_root = Path(
            config.curriculum_stage2_t4_data_root_path
            if config.curriculum_stage2_t4_data_root_path is not None
            else config.data_root_path
        )
        ma_root = Path(config.curriculum_stage2_data_root_path)
        run_stage(
            config=config,
            model=model,
            tokenizer=tokenizer,
            data_root=t4_root,
            output_root=config.curriculum_stage2_output_root_path,
            max_num_rows=config.curriculum_stage2_max_num_rows,
            max_steps=config.curriculum_stage2_max_steps,
            secondary_data_root=ma_root,
            mixing_ratio=config.curriculum_stage2_mixing_ratio,
        )


if __name__ == "__main__":
    main()
