# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

"""
Unified launcher for the three ablation experiments.

Usage:
    python -m sap_rpt_oss.run_experiment --exp 1   # MiniLM-L6 + FiLM (post-training)
    python -m sap_rpt_oss.run_experiment --exp 2   # Qwen3 + sum (from scratch)
    python -m sap_rpt_oss.run_experiment --exp 3   # Qwen3 + FiLM (from scratch)
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

    initial_checkpoint: Optional[Path] = None
    if not config.pretrain_from_scratch:
        # Exp 1: post-training from the official HF checkpoint.
        initial_checkpoint = download_official_ckpt()
        print(f"[run_experiment] Loading official checkpoint from {initial_checkpoint}")

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
    )


if __name__ == "__main__":
    main()
