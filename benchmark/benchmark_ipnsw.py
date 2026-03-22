#!/usr/bin/env python3
"""ip-nsw benchmark sweep."""

from __future__ import annotations

import os
import subprocess
import time

from common import (
    DatasetConfig,
    build_env_without_thread_limits,
    compute_recall,
    load_results,
)


def ensure_index(config: DatasetConfig) -> bool:
    if os.path.exists(config.ipnsw_graph) and os.path.getsize(config.ipnsw_graph) > 0:
        return True

    subprocess.run(
        [
            config.ipnsw_bin,
            "--mode",
            "database",
            "--database",
            config.database_bin,
            "--databaseSize",
            str(config.db_size),
            "--dimension",
            str(config.dim),
            "--outputGraph",
            config.ipnsw_graph,
            "--M",
            str(config.ipnsw_m),
            "--efConstruction",
            str(config.ipnsw_ef_construction),
        ],
        check=True,
        env=build_env_without_thread_limits(),
    )
    return os.path.exists(config.ipnsw_graph) and os.path.getsize(config.ipnsw_graph) > 0


def run(config: DatasetConfig, ground_truth):
    """Run ip-nsw recall-QPS sweep across efSearch values."""
    points = []
    if not ensure_index(config):
        print("[ip-nsw] index unavailable, skip")
        return points

    for ef in config.ipnsw_ef_values:
        if os.path.exists(config.ipnsw_result):
            os.remove(config.ipnsw_result)

        start = time.time()
        subprocess.run(
            [
                config.ipnsw_bin,
                "--mode",
                "query",
                "--query",
                config.query_bin,
                "--querySize",
                str(config.query_size),
                "--dimension",
                str(config.dim),
                "--inputGraph",
                config.ipnsw_graph,
                "--efSearch",
                str(ef),
                "--topK",
                str(config.top_k),
                "--output",
                config.ipnsw_result,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - start
        qps = config.query_size / elapsed
        results = load_results(config.ipnsw_result, expected_k=config.top_k)
        recall = compute_recall(results, ground_truth, config.top_k)
        points.append({"budget": ef, "recall": recall, "qps": qps})
        print(f"[ip-nsw] ef={ef} recall={recall:.4f} qps={qps:.2f}")

    return points
