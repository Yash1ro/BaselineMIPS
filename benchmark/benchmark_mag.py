#!/usr/bin/env python3
"""MAG benchmark sweep."""

from __future__ import annotations

import subprocess
import time

from common import (
    DatasetConfig,
    PeakMemoryTracker,
    build_env_without_thread_limits,
    compute_recall,
    ensure_parent_dir,
    file_nonempty,
    load_results,
)


def ensure_mag_knng(config: DatasetConfig, knng_k: int = 100) -> bool:
    if file_nonempty(config.mag_knng):
        return True

    ensure_parent_dir(config.mag_knng)
    try:
        subprocess.run(
            [
                "python3",
                config.mag_build_knng_py,
                config.database_bin,
                config.mag_knng,
                str(config.dim),
                str(knng_k),
            ],
            check=True,
            env=build_env_without_thread_limits(),
        )
    except Exception:
        return False

    return file_nonempty(config.mag_knng)


def ensure_mag_index(config: DatasetConfig) -> bool:
    if file_nonempty(config.mag_index):
        return True
    if not ensure_mag_knng(config):
        return False

    ensure_parent_dir(config.mag_index)
    subprocess.run(
        [
            config.mag_test_bin,
            config.database_bin,
            config.mag_knng,
            "60",
            "48",
            "300",
            config.mag_index,
            "index",
            str(config.dim),
            "20",
            "64",
            "8",
        ],
        check=True,
        env=build_env_without_thread_limits(),
    )
    return file_nonempty(config.mag_index)


def run(config: DatasetConfig, ground_truth):
    """Run MAG recall-QPS sweep across efs values."""
    points = []
    already_built = file_nonempty(config.mag_index)
    build_start = time.time()
    build_peak_mb = 0.0
    if not already_built:
        with PeakMemoryTracker() as _bt:
            if not ensure_mag_index(config):
                raise RuntimeError("[MAG] index unavailable")
        build_peak_mb = _bt.peak_mb
    else:
        if not ensure_mag_index(config):
            raise RuntimeError("[MAG] index unavailable")
    build_time = 0.0 if already_built else time.time() - build_start
    if not already_built:
        print(f"[MAG] index build time: {build_time:.2f}s, peak mem: {build_peak_mb:.1f} MB")

    query_peak_mb = 0.0
    with PeakMemoryTracker() as _qt:
        for efs in config.mag_efs:
            start = time.time()
            subprocess.run(
                [
                    config.mag_test_bin,
                    config.database_bin,
                    config.query_bin,
                    config.mag_index,
                    str(efs),
                    str(config.top_k),
                    config.mag_result,
                    "search",
                    str(config.dim),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            elapsed = time.time() - start
            qps = config.query_size / elapsed
            results = load_results(config.mag_result, expected_k=config.top_k)
            recall = compute_recall(results, ground_truth, config.top_k)
            points.append({"budget": efs, "recall": recall, "qps": qps})
            print(f"[MAG] efs={efs} recall={recall:.4f} qps={qps:.2f}")
    query_peak_mb = _qt.peak_mb

    return points, build_time, build_peak_mb, query_peak_mb
