#!/usr/bin/env python3
"""ScaNN benchmark sweep."""

from __future__ import annotations

import os
import time

import test_scann

from common import DatasetConfig, PeakMemoryTracker, build_env_without_thread_limits, compute_recall, load_results, write_neighbors_txt

_THREAD_LIMIT_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
]


def _build_searcher_multithreaded(config, database, max_reorder):
    """Temporarily lift thread-count restrictions while building the ScaNN index."""
    saved = {k: os.environ.pop(k, None) for k in _THREAD_LIMIT_KEYS}
    try:
        searcher = test_scann.build_searcher(
            database,
            top_k=config.top_k,
            distance=config.scann_distance,
            num_leaves=config.scann_num_leaves,
            num_leaves_to_search=config.scann_leaves_to_search,
            training_sample_size=len(database),
            reorder=max_reorder,
        )
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    return searcher


def run(config: DatasetConfig, database, queries, ground_truth):
    """Run ScaNN in non-single-point sweep mode."""
    points = []
    if config.scann_mode == "leaves":
        # Build once with the maximum leaves value (multi-threaded), then sweep
        # leaves_to_search at search time (single-threaded) for fair QPS measurement.
        max_reorder = max(config.scann_reorder_values)
        build_start = time.time()
        with PeakMemoryTracker() as _bt:
            leaves_searcher = _build_searcher_multithreaded(
                config, database,
                max_reorder,
            )
        build_peak_mb = _bt.peak_mb
        build_time = time.time() - build_start
        print(f"[ScaNN] index build time: {build_time:.2f}s, peak mem: {build_peak_mb:.1f} MB")
        with PeakMemoryTracker() as _qt:
            for leaves_to_search in config.scann_leaves_values:
                neighbors, qps = test_scann.search_queries(
                    leaves_searcher,
                    queries,
                    leaves_to_search=leaves_to_search,
                    pre_reorder_num_neighbors=max_reorder,
                    final_num_neighbors=config.top_k,
                )
                write_neighbors_txt(neighbors, config.scann_result)
                recall = compute_recall(load_results(config.scann_result), ground_truth, config.top_k)
                points.append({"budget": leaves_to_search, "recall": recall, "qps": qps})
                print(f"[ScaNN] leaves={leaves_to_search} recall={recall:.4f} qps={qps:.2f}")
        query_peak_mb = _qt.peak_mb
        return points, build_time, build_peak_mb, query_peak_mb

    max_reorder = max(config.scann_reorder_values)
    build_start = time.time()
    with PeakMemoryTracker() as _bt:
        searcher = _build_searcher_multithreaded(config, database, max_reorder)
    build_peak_mb = _bt.peak_mb
    build_time = time.time() - build_start
    print(f"[ScaNN] index build time: {build_time:.2f}s, peak mem: {build_peak_mb:.1f} MB")

    with PeakMemoryTracker() as _qt:
        for reorder in config.scann_reorder_values:
            neighbors, qps = test_scann.search_queries(
                searcher,
                queries,
                leaves_to_search=config.scann_leaves_to_search,
                pre_reorder_num_neighbors=reorder,
                final_num_neighbors=config.top_k,
            )
            write_neighbors_txt(neighbors, config.scann_result)
            recall = compute_recall(load_results(config.scann_result), ground_truth, config.top_k)
            points.append({"budget": reorder, "recall": recall, "qps": qps})
            print(f"[ScaNN] reorder={reorder} recall={recall:.4f} qps={qps:.2f}")
    query_peak_mb = _qt.peak_mb

    return points, build_time, build_peak_mb, query_peak_mb
