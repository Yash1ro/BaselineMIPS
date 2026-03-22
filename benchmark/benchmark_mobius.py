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
        shutil.copy2(default_graph, config.mobius_graph)
    if os.path.exists(default_data):
        shutil.copy2(default_data, config.mobius_data)

    return os.path.exists(config.mobius_graph) and os.path.exists(config.mobius_data)


def run(config: DatasetConfig, ground_truth):
    """Run Mobius recall-QPS sweep across search_budget values."""
    points = []
    if not ensure_graph(config):
        print("[Mobius] graph unavailable, skip")
        return points

    default_graph = config.mobius_default_graph
    default_data = config.mobius_default_data
    shutil.copy2(config.mobius_graph, default_graph)
    shutil.copy2(config.mobius_data, default_data)

    if not os.path.exists(config.query_txt):
        queries = read_bin(config.query_bin, config.dim)
        np.savetxt(config.query_txt, queries, fmt="%.6f", delimiter=" ")

    for budget in config.mobius_budget_values:
        start = time.time()
        query_arg = os.path.relpath(config.query_txt, start=config.mobius_dir)
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

    if points:
        with open(config.mobius_result, "w", encoding="utf-8") as f:
            for row in arr:
                f.write(" ".join(map(str, row)) + "\n")

    return points
