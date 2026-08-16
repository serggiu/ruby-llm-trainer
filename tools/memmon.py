#!/usr/bin/env python3
"""Poll system-wide free memory (memory_pressure -Q) every 2 s.

Usage: memmon.py [duration_seconds]  (default 1800)

Prints timestamped lines like:
  [12:00:03] free: 18.4 GB (57.5%)

Used to observe RAM during a training run: start it in the background,
run training, then read the log.
"""

import subprocess
import sys
import time

POLL_SECONDS = 2


def free_mem_gb(total_gb: float) -> float:
    out = subprocess.run(
        ["memory_pressure", "-Q"], capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if "System-wide memory free percentage" in line:
            pct = float(line.split(":")[-1].strip().rstrip("%"))
            return total_gb * pct / 100.0
    return total_gb  # unknown -> assume fine


def main() -> None:
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 1800
    out = subprocess.run(
        ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, check=True
    ).stdout
    total = int(out.strip()) / (1024 ** 3)

    deadline = time.time() + duration
    while time.time() < deadline:
        free = free_mem_gb(total)
        print(f"[{time.strftime('%H:%M:%S')}] free: {free:5.1f} GB ({free / total * 100:4.1f}%)",
              flush=True)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
