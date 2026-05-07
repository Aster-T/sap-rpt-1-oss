# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified launcher for the three ablation experiments.

Usage:
    python -m sap_rpt_oss.run_experiment --exp 1   # MiniLM-L6 + FiLM (post-training)
    python -m sap_rpt_oss.run_experiment --exp 2   # Qwen3 + sum (from scratch)
    python -m sap_rpt_oss.run_experiment --exp 3   # Qwen3 + FiLM (from scratch)

By default each invocation auto-resumes from the highest-step checkpoint found
in the experiment's checkpoint directory; pass --no-resume to start over.
"""

import argparse
import dataclasses
from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

from sap_rpt_oss.configs import (
    FinetuneConfig,
    get_exp1_minilm_film_config,
    get_exp2_qwen3_sum_config,
    get_exp3_qwen3_film_config,
)
from sap_rpt_oss.pretrain_run import (
    build_model_and_tokenizer,
    find_latest_checkpoint,
    run_stage,
    seed_everything,
)


EXPERIMENTS = {
    1: ("Exp 1: MiniLM-L6 + FiLM (post-training)", get_exp1_minilm_film_config),
    2: ("Exp 2: Qwen3 + sum (from scratch)", get_exp2_qwen3_sum_config),
    3: ("Exp 3: Qwen3 + FiLM (from scratch)", get_exp3_qwen3_film_config),
}

# The official HF checkpoint filename. Verified by listing repo files via the
# huggingface_hub API at the time of writing; update here if SAP renames.
OFFICIAL_CHECKPOINT_FILENAME = "2025-11-04_sap-rpt-one-oss.pt"
OFFICIAL_CHECKPOINT_REPO = "SAP/sap-rpt-1-oss"


def download_official_ckpt() -> Path:
    """Download the official sap/sap-rpt-1-oss weight-shared checkpoint."""
    return Path(
        hf_hub_download(
            repo_id=OFFICIAL_CHECKPOINT_REPO,
            filename=OFFICIAL_CHECKPOINT_FILENAME,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override default max_steps (useful for dry-runs).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override the training data root path.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any local checkpoint and start fresh "
        "(Exp 1: from official HF ckpt; Exp 2/3: from scratch).",
    )
    args = parser.parse_args()

    name, config_fn = EXPERIMENTS[args.exp]
    print(f"[run_experiment] Starting {name}")

    config: FinetuneConfig = config_fn()
    overrides = {}
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.data_root is not None:
        overrides["data_root_path"] = Path(args.data_root)
    if overrides:
        config = dataclasses.replace(config, **overrides)

    seed_everything(config.random_seed)

    # Stage name is derived from output_root.name in run_stage; mirror it here so
    # find_latest_checkpoint scopes to this experiment's files only.
    stage_name = config.output_root_path.name.replace("/", "_")
    checkpoint_dir = config.resolved_checkpoint_dir

    initial_checkpoint: Optional[Path] = None
    start_step = 0
    if not args.no_resume:
        latest_path, latest_step = find_latest_checkpoint(checkpoint_dir, stage_name)
        if latest_path is not None:
            print(
                f"[run_experiment] Resuming from {latest_path} (step={latest_step})"
            )
            # Make build_model_and_tokenizer load the local ckpt instead of
            # initializing from scratch / re-downloading from HF.
            config = dataclasses.replace(config, pretrain_from_scratch=False)
            initial_checkpoint = latest_path
            start_step = latest_step

    if initial_checkpoint is None and not config.pretrain_from_scratch:
        # Exp 1 cold start: post-training from the official HF checkpoint.
        initial_checkpoint = download_official_ckpt()
        print(f"[run_experiment] Loading official checkpoint from {initial_checkpoint}")

    if start_step >= config.max_steps:
        print(
            f"[run_experiment] start_step={start_step} >= max_steps={config.max_steps}; "
            "nothing to do."
        )
        return

    model, tokenizer = build_model_and_tokenizer(
        config=config, checkpoint=initial_checkpoint
    )
    run_stage(
        config=config,
        model=model,
        tokenizer=tokenizer,
        data_root=config.data_root_path,
        output_root=config.output_root_path,
        max_num_rows=config.stage1_max_num_rows,
        max_steps=config.max_steps,
        start_step=start_step,
    )


if __name__ == "__main__":
    main()
