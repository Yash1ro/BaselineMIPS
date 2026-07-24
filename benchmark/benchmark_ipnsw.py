#!/usr/bin/env python3
"""ip-nsw benchmark sweep."""

from __future__ import annotations

import os
import subprocess
import time

from common import (
    DatasetConfig,
    PeakMemoryTracker,
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
    already_built = os.path.exists(config.ipnsw_graph) and os.path.getsize(config.ipnsw_graph) > 0
    build_start = time.time()
    build_peak_mb = 0.0
    if not already_built:
        with PeakMemoryTracker() as _bt:
            if not ensure_index(config):
                print("[ip-nsw] index unavailable, skip")
                return points, 0.0, 0.0, 0.0
        build_peak_mb = _bt.peak_mb
    else:
        if not ensure_index(config):
            print("[ip-nsw] index unavailable, skip")
            return points, 0.0, 0.0, 0.0
    build_time = 0.0 if already_built else time.time() - build_start
    if not already_built:
        print(f"[ip-nsw] index build time: {build_time:.2f}s, peak mem: {build_peak_mb:.1f} MB")

    query_peak_mb = 0.0
    with PeakMemoryTracker() as _qt:
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
    query_peak_mb = _qt.peak_mb

    return points, build_time, build_peak_mb, query_peak_mb
