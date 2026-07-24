#!/usr/bin/env python3
"""PAG_new benchmark sweep (identical runner logic to benchmark_pag.py)."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from common import DatasetConfig, PeakMemoryTracker, build_env_without_thread_limits


def run(config: DatasetConfig):
    """Run PAG_new benchmark script and parse recall-QPS lines."""
    pag_dir = Path(config.pag_new_dir)
    script_name = config.pag_new_run_script if config.pag_new_run_script else f"run_{config.name}.sh"
    script_path = pag_dir / script_name
    if not script_path.exists():
        print(f"[PAG_new] script not found: {script_path}")
        return [], 0.0, 0.0, 0.0

    # Derive index directory from iPath line in the script
    index_dir = pag_dir / f"index_new_{config.name}"
    try:
        script_text = script_path.read_text()
        for line in script_text.splitlines():
            line = line.strip()
            if line.startswith("iPath="):
                ipath_val = line.split("=", 1)[1].strip()
                if "#" in ipath_val:
                    ipath_val = ipath_val[:ipath_val.index("#")].strip()
                ipath_val = ipath_val.strip('"').strip("'")
                ipath_val = ipath_val.replace("${name}", config.name)
                index_dir = pag_dir / ipath_val
                break
    except Exception:
        pass

    has_index = index_dir.exists() and index_dir.is_dir() and any(index_dir.iterdir())
    run_times = 1 if has_index else 2
    points = []
    last_stdout = ""
    build_time = 0.0
    build_peak_mb = 0.0
    query_peak_mb = 0.0

    for i in range(run_times):
        print(f"[PAG_new] run {i + 1}/{run_times}: {script_name}")
        if run_times == 2 and i == 0:
            env = build_env_without_thread_limits()
            run_start = time.time()
            with PeakMemoryTracker() as _bt:
                result = subprocess.run(
                    ["bash", script_name],
                    cwd=str(pag_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            build_time = time.time() - run_start
            build_peak_mb = _bt.peak_mb
            print(f"[PAG_new] index build time: {build_time:.2f}s, peak mem: {build_peak_mb:.1f} MB")
        else:
            env = os.environ.copy()
            with PeakMemoryTracker() as _qt:
                result = subprocess.run(
                    ["bash", script_name],
                    cwd=str(pag_dir),
                    check=True,
                    capture_output=True,
                    text=True,
                    env=env,
                )
            query_peak_mb = _qt.peak_mb
        last_stdout = result.stdout

    for line in last_stdout.splitlines():
        match = re.match(r"^\s*(\d+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+QPS\s*$", line)
        if not match:
            continue
        budget = int(match.group(1))
        recall = float(match.group(2))
        qps = float(match.group(3))
        points.append({"budget": budget, "recall": recall, "qps": qps})

    if not points:
        print("[PAG_new] no parsable points from output")
    else:
        print(f"[PAG_new] parsed {len(points)} points")

        # For standard sweeps, PAG_new should produce 99 points.
        expected_points = 99 if config.top_k in {10, 100} else None
        if expected_points is not None and len(points) != expected_points:
            print(
                f"[PAG_new] WARNING: expected {expected_points} points for top-{config.top_k}, "
                f"but parsed {len(points)}. "
                "Please rebuild PAG_new binary and rerun."
            )
    return points, build_time, build_peak_mb, query_peak_mb
