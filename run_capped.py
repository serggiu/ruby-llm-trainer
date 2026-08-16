#!/usr/bin/env python3
"""Launch a training command under a memory watchdog.

macOS does not allow hard per-process memory limits (setrlimit for
RLIMIT_AS/RLIMIT_DATA returns EINVAL), so a true "28 GB cap" is impossible at
the kernel level. Instead, this supervisor polls system-wide free memory and
SIGKILLs the training process as soon as free memory drops below a threshold
— the run aborts with an error instead of swapping the machine into a freeze.

With a 32 GB machine and --min-free-gb 4, training effectively cannot use
more than ~28 GB.

Usage:
  python run_capped.py [--min-free-gb 4] -- COMMAND [ARGS...]

Example:
  python run_capped.py --min-free-gb 4 -- .venv/bin/mlx_lm.lora \\
      --model _mlx/model_ornith_4bit --train --data _mlx/cpt_data ...
"""

import argparse
import subprocess
import sys
import time

POLL_SECONDS = 2


def total_mem_gb() -> float:
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
    ).stdout
    return int(out.strip()) / (1024 ** 3)


def free_mem_gb(total_gb: float) -> float:
    out = subprocess.run(
        ["memory_pressure", "-Q"], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if "System-wide memory free percentage" in line:
            pct = float(line.split(":")[-1].strip().rstrip("%"))
            return total_gb * pct / 100.0
    return total_gb  # unknown → assume fine


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-free-gb", type=float, default=4.0,
                        help="Kill the child when free memory drops below this (default 4.0)")
    parser.add_argument("cmd", nargs=argparse.REMAINDER,
                        help="Command to run, after --")
    args = parser.parse_args()

    total = total_mem_gb()
    print(f"[watchdog] {total:.0f} GB total RAM, killing child below {args.min_free_gb:.1f} GB free")

    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    child = subprocess.Popen(cmd)
    try:
        while True:
            time.sleep(POLL_SECONDS)
            if child.poll() is not None:
                print(f"[watchdog] child exited with code {child.returncode}")
                return child.returncode
            free = free_mem_gb(total)
            if free < args.min_free_gb:
                print(f"[watchdog] FREE MEMORY CRITICAL: {free:.1f} GB free "
                      f"(< {args.min_free_gb:.1f} GB) — killing child to prevent a system freeze")
                child.kill()
                child.wait()
                print("[watchdog] child killed. Reduce --max-seq-len (or the dataset size) "
                      "so the run fits under the memory cap, then retry.")
                return 1
    except KeyboardInterrupt:
        child.kill()
        child.wait()
        return 130


if __name__ == "__main__":
    sys.exit(main())
