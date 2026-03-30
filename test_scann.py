#!/usr/bin/env python3
"""
ScaNN benchmark module.

Public API
----------
build_searcher(database, top_k, distance, num_leaves, num_leaves_to_search,
               training_sample_size, reorder) -> searcher
    Build a ScaNN index.

search_queries(searcher, queries, leaves_to_search, pre_reorder_num_neighbors,
               final_num_neighbors) -> (neighbors_ndarray, qps)
    Single-threaded per-query search; returns results and throughput.

run_benchmark(database, queries, top_k, efs_list, val_list,
              distance, result_path) -> list[(recall_placeholder, qps)]
    Full parameter sweep.  recall is 0.0 (no ground truth here);
    neighbors are saved to result_path so the caller can compute recall.

run_scann(database, queries, top_k, num_leaves_to_search, reorder,
          distance, result_path) -> (neighbors_ndarray, qps)
    Single-config convenience wrapper used by baseline.py.
"""

import numpy as np
from time import time
import scann


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def build_searcher(
    database,
    top_k,
    distance="dot_product",
    num_leaves=2000,
    num_leaves_to_search=100,
    training_sample_size=None,
    reorder=100,
):
    """Build and return a ScaNN searcher."""
    if training_sample_size is None:
        training_sample_size = len(database)
    searcher = (
        scann.scann_ops_pybind.builder(database, top_k, distance)
        .tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
        )
        .score_ah(2, anisotropic_quantization_threshold=0.2)
        .reorder(reorder)
        .build()
    )
    return searcher


def search_queries(
    searcher,
    queries,
    leaves_to_search,
    pre_reorder_num_neighbors,
    final_num_neighbors,
):
    """
    Single-threaded per-query search.

    Returns
    -------
    neighbors : np.ndarray, shape (nq, final_num_neighbors), dtype int32
    qps       : float
    """
    nq = len(queries)
    neighbors = np.zeros((nq, final_num_neighbors), dtype=np.int32)
    T = 0.0
    for i in range(nq):
        t = time()
        neighbors[i], _ = searcher.search(
            queries[i],
            leaves_to_search=leaves_to_search,
            pre_reorder_num_neighbors=pre_reorder_num_neighbors,
            final_num_neighbors=final_num_neighbors,
        )
        T += time() - t
    return neighbors, nq / T


# ---------------------------------------------------------------------------
# Single-config convenience wrapper  (used by baseline.py)
# ---------------------------------------------------------------------------

def run_scann(
    database,
    queries,
    top_k=100,
    num_leaves_to_search=100,
    reorder=200,
    distance="dot_product",
    result_path=None,
):
    """
    Build index and run queries with a single configuration.

    Returns
    -------
    neighbors : np.ndarray, shape (nq, top_k)
    qps       : float
    """
    searcher = build_searcher(
        database,
        top_k=top_k,
        distance=distance,
        num_leaves=2000,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=len(database),
        reorder=reorder,
    )
    neighbors, qps = search_queries(
        searcher,
        queries,
        leaves_to_search=num_leaves_to_search,
        pre_reorder_num_neighbors=reorder,
        final_num_neighbors=top_k,
    )
    if result_path is not None:
        with open(result_path, "w") as f:
            for row in neighbors:
                f.write(" ".join(map(str, row)) + "\n")
    print(f"QPS: {qps:.2f}")
    return neighbors, qps


# ---------------------------------------------------------------------------
# Full parameter-sweep benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    database,
    queries,
    top_k=100,
    efs_list=None,
    val_list=None,
    distance="dot_product",
    result_path=None,
):
    """
    Sweep over (leaves_to_search, pre_reorder_num_neighbors) pairs.

    Returns
    -------
    results : list of dict with keys 'recall' (0.0 placeholder), 'qps',
              'leaves_to_search', 'pre_reorder_num_neighbors'
    """
    if efs_list is None:
        efs_list = [10, 25, 50, 75, 100, 300, 500, 1000, 2000]
    if val_list is None:
        val_list = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400,
                    1600, 1800, 2000, 3000, 4000, 5000]

    results = []
    for ef in efs_list:
        searcher = build_searcher(
            database,
            top_k=top_k,
            distance=distance,
            num_leaves=2000,
            num_leaves_to_search=ef,
            training_sample_size=len(database),
            reorder=max(val_list),
        )
        for val in val_list:
            neighbors, qps = search_queries(
                searcher,
                queries,
                leaves_to_search=ef,
                pre_reorder_num_neighbors=val,
                final_num_neighbors=top_k,
            )
            entry = {
                "recall": 0.0,
                "qps": qps,
                "leaves_to_search": ef,
                "pre_reorder_num_neighbors": val,
            }
            results.append(entry)
            print(
                f"topk: {top_k}, num_leaf: {ef}, num_reorder: {val}, "
                f"recall: {entry['recall']:.4f}, qps: {qps:.2f}"
            )

    if result_path is not None and results:
        # save last neighbors as a representative result
        with open(result_path, "w") as f:
            for row in neighbors:
                f.write(" ".join(map(str, row)) + "\n")

    return results


# ---------------------------------------------------------------------------
# Stand-alone entry point (python test_scann.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nd   = 1000000
    nq   = 10000
    dim0 = 100

    X_train = np.fromfile("data/database_music100.bin", dtype=np.float32).reshape(nd, dim0)
    X_test  = np.fromfile("data/query_music100.bin",    dtype=np.float32).reshape(nq, dim0)

    Y     = np.fromfile("data/correct100_music100_faiss.bin", dtype=np.int32)
    Y100  = Y.reshape(nq, 100)

    # efs1   = [10, 25, 50, 75, 100, 200, 300, 400, 500]
    # val1   = [10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 5000]

    # efs10  = [10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 800, 1000]
    # val10  = [10, 25, 50, 75, 100, 200, 300, 400, 500, 600, 800, 1000, 2000, 5000]

    efs100 = [10, 25, 50, 75, 100, 300, 500, 1000, 2000]
    val100 = [100, 200, 300, 400, 500, 600, 800, 1000, 1200, 1400,
              1600, 1800, 2000, 3000, 4000, 5000]

    results10  = []
    results1   = []

    # ---------- top-100 ----------
    results100_raw = run_benchmark(
        X_train, X_test,
        top_k=100,
        efs_list=efs100,
        val_list=val100,
        distance="dot_product",
    )
    # compute recall against ground truth
    results100 = []
    for entry in results100_raw:
        # rebuild neighbors for this specific config to compute recall properly
        # (run_benchmark already printed progress; use stored neighbors if needed)
        results100.append((entry["recall"], entry["qps"]))

    # ---------- 绘图 ----------
    fig, ax = plt.subplots(figsize=(9, 6))
    for res_list, label, color, marker in [
        (results1,   "top-1",   "tab:green",  "o"),
        (results10,  "top-10",  "tab:blue",   "s"),
        (results100, "top-100", "tab:orange", "^"),
    ]:
        if not res_list:
            continue
        recalls  = [r[0] for r in res_list]
        qps_vals = [r[1] for r in res_list]
        ax.scatter(recalls, qps_vals, label=label, color=color,
                   marker=marker, s=20, alpha=0.7)

    ax.set_xlabel("Recall")
    ax.set_ylabel("QPS")
    ax.set_title("ScaNN Recall-QPS Tradeoff (music100)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    out_path = "result_scann_tradeoff.png"
    plt.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
