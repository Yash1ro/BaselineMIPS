#!/usr/bin/env python3
"""ScaNN benchmark sweep."""

from __future__ import annotations

import test_scann

from common import DatasetConfig, compute_recall, load_results, write_neighbors_txt


def run(config: DatasetConfig, database, queries, ground_truth):
    """Run ScaNN in non-single-point sweep mode."""
    points = []
    if config.scann_mode == "leaves":
        for leaves_to_search in config.scann_leaves_values:
            neighbors, qps = test_scann.run_scann(
                database,
                queries,
                top_k=config.top_k,
                num_leaves_to_search=leaves_to_search,
                reorder=max(config.scann_reorder_values),
                distance="dot_product",
                result_path=config.scann_result,
            )
            write_neighbors_txt(neighbors, config.scann_result)
            recall = compute_recall(load_results(config.scann_result), ground_truth, config.top_k)
            points.append({"budget": leaves_to_search, "recall": recall, "qps": qps})
            print(f"[ScaNN] leaves={leaves_to_search} recall={recall:.4f} qps={qps:.2f}")
        return points

    max_reorder = max(config.scann_reorder_values)
    searcher = test_scann.build_searcher(
        database,
        top_k=config.top_k,
        distance="dot_product",
        num_leaves=config.scann_num_leaves,
        num_leaves_to_search=config.scann_leaves_to_search,
        training_sample_size=len(database),
        reorder=max_reorder,
    )

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

    return points
