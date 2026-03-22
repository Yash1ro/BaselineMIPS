#!/usr/bin/env python3
"""FAISS IVF-PQ benchmark sweep."""

from __future__ import annotations

import time

import numpy as np

from common import DatasetConfig, compute_recall, write_neighbors_txt


try:
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _set_nprobe(index, nprobe: int) -> None:
    if hasattr(index, "index"):
        ivf_index = faiss.extract_index_ivf(index.index)
        ivf_index.nprobe = nprobe
    else:
        index.nprobe = nprobe


def run(config: DatasetConfig, database, queries, ground_truth):
    """Run FAISS IVF-PQ recall-QPS sweep across nprobe values."""
    points = []
    if faiss is None:
        print("[FAISS] faiss not installed, skip")
        return points

    faiss.omp_set_num_threads(1)
    database = np.ascontiguousarray(database)
    queries = np.ascontiguousarray(queries)

    index_desc = f"OPQ{config.faiss_m}_{config.dim},IVF{config.faiss_nlist},PQ{config.faiss_m}x{config.faiss_nbits}"
    index = faiss.index_factory(config.dim, index_desc, faiss.METRIC_INNER_PRODUCT)

    train_size = min(len(database), 250000)
    index.train(database[:train_size])
    index.add(database)

    for nprobe in config.faiss_nprobe_values:
        _set_nprobe(index, nprobe)
        start = time.time()
        _, neighbors = index.search(queries, config.top_k)
        elapsed = time.time() - start
        qps = config.query_size / elapsed

        write_neighbors_txt(neighbors, config.faiss_result)
        recall = compute_recall(neighbors, ground_truth, config.top_k)
        points.append({"budget": nprobe, "recall": recall, "qps": qps})
        print(f"[FAISS] nprobe={nprobe} recall={recall:.4f} qps={qps:.2f}")

    return points
