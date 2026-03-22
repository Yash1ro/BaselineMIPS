#!/usr/bin/env python3
"""PAG benchmark sweep by invoking dataset shell script."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from common import DatasetConfig


def run(config: DatasetConfig):
    """Run PAG benchmark script and parse recall-QPS lines."""
    pag_dir = Path(config.pag_dir)
    script_path = pag_dir / f"run_{config.name}.sh"
    if not script_path.exists():
        print(f"[PAG] script not found: {script_path}")
        return []

    index_dir = pag_dir / config.name / "index"
    run_times = 1 if index_dir.exists() else 2
    points = []
    last_stdout = ""

    for i in range(run_times):
        print(f"[PAG] run {i + 1}/{run_times}: {script_path.name}")
        result = subprocess.run(
            ["bash", script_path.name],
            cwd=config.pag_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        last_stdout = result.stdout

    for line in last_stdout.splitlines():
        match = re.match(r"^\s*(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+QPS\s*$", line)
        if not match:
            continue
        budget = int(match.group(1))
        recall = float(match.group(2))
        qps = float(match.group(3))
        points.append({"budget": budget, "recall": recall, "qps": qps})

    if not points:
        print("[PAG] no parsable points from output")
    else:
        print(f"[PAG] parsed {len(points)} points")
    return points
