#!/usr/bin/env python3
"""Mobius benchmark sweep."""

from __future__ import annotations

import os
import shutil
import subprocess
import time

import numpy as np

from common import (
    DatasetConfig,
    PeakMemoryTracker,
    build_env_without_thread_limits,
    compute_recall,
    read_bin,
)


def _parse_mobius_stdout(text: str, top_k: int):
    rows = []
    for line in text.strip().split("\n"):
        if not line or line.startswith("Loading"):
            continue
        parts = line.strip().split()
        if len(parts) < top_k:
            continue
        try:
            rows.append([int(x) for x in parts[:top_k]])
        except ValueError:
            continue
    return rows


def ensure_graph(config: DatasetConfig) -> bool:
    if os.path.exists(config.mobius_graph) and os.path.exists(config.mobius_data):
        return True

    # Prefer binary file (avoids huge txt conversion for large datasets).
    # The mobius binary auto-detects .bin extension and uses ParserDenseBin.
    if os.path.exists(config.database_bin):
        db_arg = os.path.relpath(config.database_bin, start=config.mobius_dir)
    else:
        if not os.path.exists(config.database_txt):
            db = read_bin(config.database_bin, config.dim)
            np.savetxt(config.database_txt, db, fmt="%.6f", delimiter=" ")
        db_arg = os.path.relpath(config.database_txt, start=config.mobius_dir)
    subprocess.run(
        [config.mobius_build_sh, db_arg, str(config.db_size), str(config.dim)],
        check=True,
        cwd=config.mobius_dir,
        env=build_env_without_thread_limits(),
    )

    default_graph = config.mobius_default_graph
    default_data = config.mobius_default_data
    if os.path.exists(default_graph):
        # Use rename instead of copy to avoid doubling disk usage for large data files.
        if os.path.exists(config.mobius_graph):
            os.remove(config.mobius_graph)
        os.rename(default_graph, config.mobius_graph)
    if os.path.exists(default_data):
        if os.path.exists(config.mobius_data):
            os.remove(config.mobius_data)
        os.rename(default_data, config.mobius_data)

    return os.path.exists(config.mobius_graph) and os.path.exists(config.mobius_data)


def run(config: DatasetConfig, ground_truth):
    """Run Mobius recall-QPS sweep across search_budget values."""
    points = []
    already_built = os.path.exists(config.mobius_graph) and os.path.exists(config.mobius_data)
    build_start = time.time()
    build_peak_mb = 0.0
    if not already_built:
        with PeakMemoryTracker() as _bt:
            if not ensure_graph(config):
                print("[Mobius] graph unavailable, skip")
                return points, 0.0, 0.0, 0.0
        build_peak_mb = _bt.peak_mb
    else:
        if not ensure_graph(config):
            print("[Mobius] graph unavailable, skip")
            return points, 0.0, 0.0, 0.0
    build_time = 0.0 if already_built else time.time() - build_start
    if not already_built:
        print(f"[Mobius] graph build time: {build_time:.2f}s, peak mem: {build_peak_mb:.1f} MB")

    default_graph = config.mobius_default_graph
    default_data = config.mobius_default_data
    # Use symlinks to avoid doubling disk usage for large data files.
    for src, dst in [(config.mobius_graph, default_graph), (config.mobius_data, default_data)]:
        if os.path.exists(dst) or os.path.islink(dst):
            os.remove(dst)
        os.symlink(os.path.abspath(src), dst)

    if not os.path.exists(config.query_txt) and not os.path.exists(config.query_bin):
        queries = read_bin(config.query_bin, config.dim)
        np.savetxt(config.query_txt, queries, fmt="%.6f", delimiter=" ")

    # Prefer binary query file when available.
    if os.path.exists(config.query_bin):
        query_file = config.query_bin
    else:
        query_file = config.query_txt

    query_peak_mb = 0.0
    with PeakMemoryTracker() as _qt:
        for budget in config.mobius_budget_values:
            start = time.time()
            query_arg = os.path.relpath(query_file, start=config.mobius_dir)
            result = subprocess.run(
                [
                    config.mobius_bin,
                    "test",
                    "0",
                    query_arg,
                    str(budget),
                    str(config.db_size),
                    str(config.dim),
                    str(config.top_k),
                    str(budget),
                ],
                check=True,
                capture_output=True,
                text=True,
                cwd=config.mobius_dir,
            )
            elapsed = time.time() - start
            qps = config.query_size / elapsed

            rows = _parse_mobius_stdout(result.stdout, config.top_k)
            if len(rows) != config.query_size:
                print(f"[Mobius] budget={budget} bad result count={len(rows)}")
                continue

            arr = np.asarray(rows, dtype=np.int64)
            recall = compute_recall(arr, ground_truth, config.top_k)
            points.append({"budget": budget, "recall": recall, "qps": qps})
            print(f"[Mobius] budget={budget} recall={recall:.4f} qps={qps:.2f}")
    query_peak_mb = _qt.peak_mb

    if points:
        with open(config.mobius_result, "w", encoding="utf-8") as f:
            for row in arr:
                f.write(" ".join(map(str, row)) + "\n")

    return points, build_time, build_peak_mb, query_peak_mb
