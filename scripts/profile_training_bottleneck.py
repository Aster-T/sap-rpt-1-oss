"""
Profile a training loop to find where time goes.

Usage:
    python scripts/profile_training_bottleneck.py
    python scripts/profile_training_bottleneck.py --iters 100 --warmup 10
    python scripts/profile_training_bottleneck.py --num-workers 4
    python scripts/profile_training_bottleneck.py --use-cached-batch  # GPU-only stress

Reports per-stage timing across the training step, samples GPU utilization
during compute, and gives actionable recommendations.
"""

import argparse
import statistics
import subprocess
import threading
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW

from sap_rpt_oss.configs import get_exp1_minilm_film_config
from sap_rpt_oss.pretrain_run import (
    build_dataloader,
    build_model_and_tokenizer,
    infer_autocast_dtype,
    move_to_device,
    seed_everything,
)


@contextmanager
def cuda_timed(stats: dict, key: str, sync: bool = True):
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    stats[key].append(time.perf_counter() - t0)


def fmt_ms(seconds_list):
    if not seconds_list:
        return "n/a"
    arr_ms = [s * 1000 for s in seconds_list]
    return (
        f"mean={statistics.mean(arr_ms):7.2f}  "
        f"p50={statistics.median(arr_ms):7.2f}  "
        f"p95={sorted(arr_ms)[int(0.95 * len(arr_ms))]:7.2f}  "
        f"max={max(arr_ms):7.2f} ms"
    )


class GPUUtilSampler:
    """Background thread that samples nvidia-smi gpu utilization."""

    def __init__(self, gpu_index: int, interval: float = 0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples_util = []
        self.samples_mem = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        cmd = [
            "nvidia-smi",
            f"--id={self.gpu_index}",
            "--query-gpu=utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ]
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=2, check=False
                )
                line = out.stdout.strip()
                if line:
                    util, mem = (x.strip() for x in line.split(","))
                    self.samples_util.append(int(util))
                    self.samples_mem.append(int(mem))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def report(self) -> str:
        if not self.samples_util:
            return "no GPU samples collected"
        return (
            f"GPU util:  mean={statistics.mean(self.samples_util):5.1f}%  "
            f"p50={statistics.median(self.samples_util):5.1f}%  "
            f"max={max(self.samples_util):5.1f}%   "
            f"(memory bus util mean={statistics.mean(self.samples_mem):5.1f}%)"
        )


def run_profile(args):
    config = get_exp1_minilm_film_config()
    if args.num_workers is not None:
        config = type(config)(**{**config.__dict__, "num_workers": args.num_workers})
    seed_everything(config.random_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = infer_autocast_dtype()

    print(f"[profile] device={device} autocast_dtype={autocast_dtype}")
    print(f"[profile] config: max_num_rows={config.stage1_max_num_rows} "
          f"max_num_columns={config.table_rules.max_num_columns} "
          f"num_workers={config.num_workers}")

    # ---- model + optimizer ----
    initial_checkpoint = (
        config.resolved_resume_checkpoint_path
        if not config.pretrain_from_scratch
        else None
    )
    model, tokenizer = build_model_and_tokenizer(
        config=config, checkpoint=initial_checkpoint
    )
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scaler = torch.cuda.amp.GradScaler(
        enabled=device.type == "cuda" and autocast_dtype == torch.float16
    )

    # ---- dataloader ----
    dataloader = build_dataloader(
        config=config,
        tokenizer=tokenizer,
        data_root=config.data_root_path,
        max_num_rows=config.stage1_max_num_rows,
    )
    iter_loader = iter(dataloader)

    # ---- warmup ----
    print(f"[profile] warmup for {args.warmup} iterations...")
    for _ in range(args.warmup):
        batch = next(iter_loader)
        is_regression = bool(batch["is_regression"])
        batch_dev = move_to_device(batch, device)
        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype is not None
            else nullcontext()
        )
        with ctx:
            _, loss, _ = model(
                batch_dev["data"],
                is_regression=is_regression,
                labels=batch_dev["labels"],
            )
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # ---- profiling loop ----
    stats: dict = defaultdict(list)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    sampler = None
    if device.type == "cuda":
        sampler = GPUUtilSampler(gpu_index=torch.cuda.current_device())
        sampler.start()

    print(f"[profile] timing {args.iters} iterations "
          f"(use_cached_batch={args.use_cached_batch})...")

    cached_batch = None
    if args.use_cached_batch:
        cached_batch = next(iter_loader)
        cached_is_reg = bool(cached_batch["is_regression"])
        cached_batch_dev = move_to_device(cached_batch, device)

    wall_t0 = time.perf_counter()
    for it in range(args.iters):
        # 1. data loading wait
        t0 = time.perf_counter()
        if args.use_cached_batch:
            batch = cached_batch
            is_regression = cached_is_reg
            batch_dev = cached_batch_dev
            stats["1_data_wait"].append(0.0)
        else:
            batch = next(iter_loader)
            is_regression = bool(batch["is_regression"])
            stats["1_data_wait"].append(time.perf_counter() - t0)

            # 2. host->device transfer
            with cuda_timed(stats, "2_h2d_transfer"):
                batch_dev = move_to_device(batch, device)

        ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype is not None
            else nullcontext()
        )
        # 3. forward
        with cuda_timed(stats, "3_forward"):
            with ctx:
                _, loss, _ = model(
                    batch_dev["data"],
                    is_regression=is_regression,
                    labels=batch_dev["labels"],
                )

        # 4. backward
        with cuda_timed(stats, "4_backward"):
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

        # 5. optimizer step
        with cuda_timed(stats, "5_optimizer"):
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    wall = time.perf_counter() - wall_t0
    if sampler:
        sampler.stop()

    # ---- report ----
    print()
    print("=" * 80)
    print(f"Per-stage timing over {args.iters} iters "
          f"(wall={wall:.1f}s, throughput={args.iters / wall:.2f} batch/s)")
    print("=" * 80)

    total_mean = 0.0
    keys_in_order = [
        "1_data_wait",
        "2_h2d_transfer",
        "3_forward",
        "4_backward",
        "5_optimizer",
    ]
    means = {}
    for key in keys_in_order:
        if not stats[key]:
            continue
        mean_s = statistics.mean(stats[key])
        means[key] = mean_s
        total_mean += mean_s
        print(f"  {key:18s}  {fmt_ms(stats[key])}")
    print(f"  {'TOTAL (sum mean)':18s}  {total_mean*1000:7.2f} ms/iter")
    print()
    print(f"  {sampler.report() if sampler else 'CPU run, no GPU stats'}")
    if device.type == "cuda":
        print(f"  Peak GPU memory: "
              f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GiB")
    print()

    # ---- diagnosis ----
    print("=" * 80)
    print("Diagnosis")
    print("=" * 80)
    if not means:
        print("  No timing data collected.")
        return

    biggest = max(means, key=lambda k: means[k])
    biggest_pct = means[biggest] / total_mean * 100
    print(f"  Bottleneck stage: {biggest} ({biggest_pct:.1f}% of step time)")
    print()

    data_pct = means.get("1_data_wait", 0.0) / total_mean * 100
    fwd_bwd_pct = (
        means.get("3_forward", 0.0) + means.get("4_backward", 0.0)
    ) / total_mean * 100
    h2d_pct = means.get("2_h2d_transfer", 0.0) / total_mean * 100
    opt_pct = means.get("5_optimizer", 0.0) / total_mean * 100

    if sampler and sampler.samples_util:
        gpu_mean_util = statistics.mean(sampler.samples_util)
    else:
        gpu_mean_util = None

    print("Findings & recommendations:")
    if data_pct > 25:
        print(f"  - Data loading takes {data_pct:.1f}% of step time.")
        print(f"    → Increase config.num_workers (currently {config.num_workers}).")
        print(f"    → Try: --num-workers 4 with this script to validate.")
    elif data_pct > 10:
        print(f"  - Data loading is moderate ({data_pct:.1f}%). num_workers=2~4 may still help.")
    else:
        print(f"  - Data loading is fast ({data_pct:.1f}%); not a bottleneck.")

    if h2d_pct > 5:
        print(f"  - Host->device transfer is {h2d_pct:.1f}%. "
              f"Consider pin_memory=True in DataLoader.")

    if fwd_bwd_pct > 60:
        print(f"  - GPU compute (fwd+bwd) is {fwd_bwd_pct:.1f}% — likely compute-bound.")
        if gpu_mean_util is not None and gpu_mean_util < 70:
            print(f"    But GPU mean utilization is only {gpu_mean_util:.1f}%.")
            print(f"    → Sequence/batch is too small to saturate the GPU.")
            print(f"    → Try increasing max_num_columns or stage1_max_num_rows "
                  f"(VRAM permitting).")
        else:
            print(f"    GPU is {gpu_mean_util:.1f}% utilized — saturated.")

    if opt_pct > 15:
        print(f"  - Optimizer step is {opt_pct:.1f}%. "
              f"Unusual; check parameter count or use fused AdamW.")

    if args.use_cached_batch:
        print()
        print("  NOTE: ran with cached batch (no data pipeline). "
              "GPU stats here are pure compute upper bound.")
        print("  Run again WITHOUT --use-cached-batch to see real-world cost.")
    else:
        print()
        print("  TIP: rerun with --use-cached-batch to see GPU-only ceiling.")
        print("  If GPU util jumps a lot in cached mode, your data pipeline "
              "is the limiter.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override config.num_workers for this profile run.",
    )
    parser.add_argument(
        "--use-cached-batch",
        action="store_true",
        help="Skip dataloader; reuse one batch repeatedly. "
        "Measures pure GPU compute ceiling.",
    )
    args = parser.parse_args()

    run_profile(args)


if __name__ == "__main__":
    main()
