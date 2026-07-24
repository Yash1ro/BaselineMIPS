#!/usr/bin/env python3
"""Generate top-10 Recall-QPS plots and memory bar charts from existing results.

This script reads existing result files and statistics.log to produce:
1. Per-dataset recall-QPS line charts for top-10
2. Per-dataset memory bar charts
3. A combined multi-dataset comparison chart
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from result_plot import (
    load_results,
    plot_memory_bar_chart,
    plot_qps_recall_multi,
    plot_results,
)

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "results"
IMGS_DIR = BENCHMARK_DIR / "imgs"
STATS_LOG = ROOT / "statistics.log"

ALL_DATASETS = [
    "music100",
    "glove100",
    "glove200",
    "dinov2",
    "book_corpus",
    "gist1m",
    "ir101",
    "openai1536",
    "laion12m",
    "msong",
    "tiny5m",
    "word2vec",
    "text2img10m",
]
TOP_K = 10


def _list_result_files(dataset: str, top_k: int) -> list[Path]:
    return sorted(
        [f for f in RESULTS_DIR.glob(f"{dataset}_top{top_k}_*.txt") if f.stat().st_size > 0],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )


def _file_contains_algorithm(result_file: Path, algorithm: str) -> bool:
    points = load_results(str(result_file))
    return any(p["algorithm"] == algorithm for p in points)


def find_result_by_rank(
    dataset: str,
    top_k: int,
    rank: int = 1,
    required_algorithm: str | None = None,
) -> Path | None:
    """Find the Nth-latest result file, optionally restricted to files containing an algorithm."""
    files = _list_result_files(dataset, top_k)
    if rank < 1:
        raise ValueError("rank must be >= 1")
    if required_algorithm is not None:
        files = [f for f in files if _file_contains_algorithm(f, required_algorithm)]
    return files[rank - 1] if len(files) >= rank else None


def load_stats_memory(top_k: int = 10) -> dict[str, dict[str, float]]:
    """Extract query_peak_mb from statistics.log for a given top_k.

    Returns {dataset: {algo: peak_mb}}.
    """
    result: dict[str, dict[str, float]] = defaultdict(dict)
    if not STATS_LOG.exists():
        return result

    with open(STATS_LOG, "r") as f:
        for line in f:
            if "# [RAW JSON]" not in line:
                continue
            raw = line.split("# [RAW JSON] ", 1)[1].strip()
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "summary":
                continue
            if rec.get("top_k") != top_k:
                continue
            ds = rec.get("dataset", "")
            for algo, info in rec.get("algorithms", {}).items():
                if info.get("status") == "failed":
                    continue
                qm = info.get("query_peak_mb", 0)
                if qm > 0:
                    result[ds][algo] = qm
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate top-k plots from saved benchmark result files.")
    parser.add_argument(
        "--result-rank",
        type=int,
        default=1,
        help="Which result file to use by recency: 1=latest, 2=previous, ...",
    )
    parser.add_argument(
        "--prefer-algorithm",
        default=None,
        help="Prefer result files that contain this algorithm when selecting by rank, e.g. 'pag'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    IMGS_DIR.mkdir(parents=True, exist_ok=True)
    dataset_points: dict[str, list[dict]] = {}

    print("=" * 60)
    print(f"Generating top-{TOP_K} plots from existing results (rank={args.result_rank})")
    print("=" * 60)

    # 1. Per-dataset recall-QPS plots
    for ds in ALL_DATASETS:
        result_file = find_result_by_rank(
            ds,
            TOP_K,
            rank=args.result_rank,
            required_algorithm=args.prefer_algorithm,
        )
        selected_with_fallback = False
        if result_file is None and args.prefer_algorithm is not None:
            result_file = find_result_by_rank(ds, TOP_K, rank=1)
            selected_with_fallback = result_file is not None
        if result_file is None:
            print(f"  [{ds}] No result file found for top-{TOP_K} at rank {args.result_rank}, skipping")
            continue

        if selected_with_fallback:
            print(f"  [{ds}] No rank-{args.result_rank} file containing {args.prefer_algorithm!r}; fallback to {result_file.name}")
        else:
            print(f"  [{ds}] Using {result_file.name}")
        points = load_results(str(result_file))
        if not points:
            print(f"  [{ds}] Empty result file, skipping")
            continue

        algos = set(p["algorithm"] for p in points)
        print(f"  [{ds}] Algorithms: {sorted(algos)}, {len(points)} points")

        dataset_points[ds] = points

        # Individual recall-QPS plot
        plot_path = str(IMGS_DIR / f"{ds}_top{TOP_K}.png")
        plot_results(
            str(result_file), plot_path,
            f"{ds} Recall-QPS (top-{TOP_K})",
            dataset_name=ds, top_k=TOP_K,
        )

    # 2. Combined multi-dataset QPS-recall comparison
    if dataset_points:
        combo_path = str(IMGS_DIR / f"all_datasets_top{TOP_K}_comparison.png")
        plot_qps_recall_multi(
            dataset_points, combo_path,
            title=f"All Datasets — Recall-QPS (top-{TOP_K})",
            top_k=TOP_K,
        )

    # 3. Memory bar charts from statistics.log
    mem_data = load_stats_memory(TOP_K)
    for ds in ALL_DATASETS:
        if ds not in mem_data:
            print(f"  [{ds}] No memory data for top-{TOP_K}")
            continue
        # For single topk bar chart, wrap as {algo: {"top10": value}}
        single_mem = {algo: {f"top{TOP_K}": mb} for algo, mb in mem_data[ds].items()}
        mem_png = str(IMGS_DIR / f"{ds}_top{TOP_K}_memory.png")
        plot_memory_bar_chart(
            single_mem, mem_png,
            title=f"{ds} — Query Peak Memory (top-{TOP_K})",
        )

    # 4. Cross-topk memory bar charts (all available topk values per dataset)
    all_topks_mem: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for tk in [10, 100, 500]:
        tk_mem = load_stats_memory(tk)
        for ds, algo_mem in tk_mem.items():
            for algo, mb in algo_mem.items():
                all_topks_mem[ds][algo][f"top{tk}"] = mb

    for ds in ALL_DATASETS:
        if ds not in all_topks_mem:
            continue
        mem_png = str(IMGS_DIR / f"{ds}_memory_all_topk.png")
        plot_memory_bar_chart(
            dict(all_topks_mem[ds]), mem_png,
            title=f"{ds} — Query Peak Memory by Algorithm × Top-K",
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
