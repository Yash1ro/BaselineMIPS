#!/usr/bin/env python3
"""Integrated end-to-end benchmark runner (without FAISS, with PAG)."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

# Global single-thread limits for fair QPS comparison.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
TOOLS_DIR = THIS_DIR / "tools"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from common import DatasetConfig, flatten_points, load_dataset_groundtruth, read_bin
from result_plot import plot_results, save_results


RUNNER_MODULES = {
    "mag": "benchmark_mag",
    "scann": "benchmark_scann",
    "ipnsw": "benchmark_ipnsw",
    "mobius": "benchmark_mobius",
    "pag": "benchmark_pag",
}


def run_algorithm(name: str, cfg, database, queries, ground_truth):
    mod = importlib.import_module(RUNNER_MODULES[name])
    if name in {"mag", "ipnsw", "mobius"}:
        return mod.run(cfg, ground_truth)
    if name == "pag":
        return mod.run(cfg)
    return mod.run(cfg, database, queries, ground_truth)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument("--dataset", default="music100", choices=["music100", "glove100"], help="Dataset name")
    parser.add_argument(
        "--algorithms",
        default="mag,scann,ipnsw,mobius,pag",
        help="Comma-separated algorithms to run",
    )
    parser.add_argument("--result-txt", default=str(THIS_DIR / "results" / "result.txt"), help="Result txt path")
    parser.add_argument("--plot", default=None, help="Output plot path")
    parser.add_argument("--title", default=None, help="Plot title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(name=args.dataset)

    print("=" * 72)
    print("BENCHMARK PIPELINE")
    print("=" * 72)
    print(f"Dataset : {cfg.name}")
    print(f"Top-K   : {cfg.top_k}")

    plot_path = args.plot or str(THIS_DIR / "imgs" / f"{cfg.name}_top{cfg.top_k}.png")
    plot_title = args.title or f"{cfg.name} Recall-QPS Benchmark"

    print("\n[1/4] Loading data...")
    database = read_bin(cfg.database_bin, cfg.dim)
    queries = read_bin(cfg.query_bin, cfg.dim)
    ground_truth = load_dataset_groundtruth(cfg)
    print(f"Database shape: {database.shape}")
    print(f"Query shape   : {queries.shape}")
    print(f"GT shape      : {ground_truth.shape}")

    selected = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    invalid = [name for name in selected if name not in RUNNER_MODULES]
    if invalid:
        raise ValueError(f"Unknown algorithms: {invalid}")

    print("\n[2/4] Running algorithm sweeps...")
    all_points: list[dict] = []
    for name in selected:
        print(f"\n--- {name.upper()} ---")
        points = run_algorithm(name, cfg, database, queries, ground_truth)
        all_points.extend(flatten_points(name, points))

    if not all_points:
        raise RuntimeError("No benchmark points produced")

    print("\n[3/4] Saving result txt...")
    save_results(all_points, args.result_txt)

    print("\n[4/4] Plotting curves...")
    plot_results(args.result_txt, plot_path, plot_title, dataset_name=cfg.name, top_k=cfg.top_k)

    print("\nDone")
    print(f"Result txt: {args.result_txt}")
    print(f"Plot file : {plot_path}")


if __name__ == "__main__":
    main()
