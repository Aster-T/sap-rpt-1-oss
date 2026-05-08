"""
Diagnose a "stuck" Python training process.

Usage:
    python diag_stuck_process.py <PID>

Prints:
  - Process tree, CPU/RSS over a 5s window
  - Open files (parquet / model files / sockets)
  - TCP connections (to spot a hanging HTTP/HF download)
  - Top stack frame per thread via py-spy if available, otherwise GDB fallback
  - Per-GPU memory & this PID's allocation on each visible GPU
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def run(cmd, timeout=10):
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        return out.stdout + (("\n[stderr]\n" + out.stderr) if out.stderr else "")
    except subprocess.TimeoutExpired:
        return f"<timeout running {' '.join(cmd)}>"
    except FileNotFoundError:
        return f"<{cmd[0]} not found>"


def section(title):
    print("\n" + "=" * 8 + f" {title} " + "=" * 8)


def main():
    if len(sys.argv) != 2:
        print("usage: diag_stuck_process.py <PID>")
        sys.exit(1)
    pid = sys.argv[1]
    proc_dir = Path(f"/proc/{pid}")
    if not proc_dir.exists():
        print(f"PID {pid} not found")
        sys.exit(1)

    section("BASIC")
    print(
        run(
            [
                "ps",
                "-p",
                pid,
                "-o",
                "pid,ppid,stat,etime,pcpu,pmem,rss,vsz,wchan:32,cmd",
            ]
        )
    )

    section("THREAD STATES (look for D=disk-wait, R=running, S=sleeping)")
    print(run(["ps", "-T", "-p", pid, "-o", "spid,stat,wchan:40,pcpu,pmem,cmd"]))

    section("CPU / MEM SAMPLE OVER 5s")
    s1 = run(["cat", f"/proc/{pid}/stat"])
    rss1 = run(["cat", f"/proc/{pid}/statm"])
    time.sleep(5)
    s2 = run(["cat", f"/proc/{pid}/stat"])
    rss2 = run(["cat", f"/proc/{pid}/statm"])

    def cpu_jiffies(stat):
        parts = stat.split()
        try:
            return int(parts[13]) + int(parts[14])  # utime + stime
        except (IndexError, ValueError):
            return 0

    delta = cpu_jiffies(s2) - cpu_jiffies(s1)
    hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    cpu_pct = 100.0 * delta / hz / 5
    print(f"CPU%% over last 5s: {cpu_pct:.1f}")
    print(f"RSS pages then -> now: {rss1.split()[1]} -> {rss2.split()[1]}")

    section("OPEN FILES (parquet / .pt / .safetensors / sockets)")
    lsof = shutil.which("lsof")
    if lsof:
        out = run([lsof, "-p", pid])
        for line in out.splitlines():
            if any(
                k in line.lower()
                for k in ("parquet", ".pt", "safetensors", "tcp", "udp", "sock")
            ):
                print(line)
    else:
        # /proc fallback
        fd_dir = proc_dir / "fd"
        if fd_dir.exists():
            for fd in sorted(fd_dir.iterdir(), key=lambda p: int(p.name)):
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if any(
                    k in target.lower()
                    for k in ("parquet", ".pt", "safetensors", "socket")
                ):
                    print(f"{fd.name} -> {target}")

    section("TCP CONNECTIONS (any to huggingface/hf-mirror = network hang)")
    ss = shutil.which("ss")
    if ss:
        print(run([ss, "-tnp", "state", "established"]))
    else:
        print(run(["netstat", "-tnp"]))

    section("STACK TRACE (Python)")
    pyspy = shutil.which("py-spy")
    if pyspy:
        print(run([pyspy, "dump", "--pid", pid], timeout=30))
    else:
        print("py-spy not installed. Install with: pip install py-spy")
        print("Falling back to gdb (C-level stack only):")
        gdb = shutil.which("gdb")
        if gdb:
            print(
                run(
                    [
                        gdb,
                        "-batch",
                        "-p",
                        pid,
                        "-ex",
                        "thread apply all bt",
                        "-ex",
                        "detach",
                        "-ex",
                        "quit",
                    ],
                    timeout=30,
                )
            )

    section("GPU VIEW")
    print(
        run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv",
            ]
        )
    )
    print(
        run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv",
            ]
        )
    )

    section("DONE")
    print("Interpretation hints:")
    print(
        "- If py-spy shows the top frame inside `requests`/`urllib3`/`huggingface_hub`"
    )
    print("  -> stuck on network. Pre-download the model or set HF_HUB_OFFLINE=1.")
    print("- If top frame is `sentence_transformers` / `transformers` forward on CPU")
    print("  -> embedder running on CPU; pass sentence_embedder_device='cuda:3'.")
    print("- If frame is in `pyarrow` / `parquet` / `_iter_table_chunks`")
    print("  -> dataset probe phase; reduce parquet count or skip auto-select.")
    print("- If thread state is mostly 'D' -> disk/IO bottleneck.")


if __name__ == "__main__":
    main()
