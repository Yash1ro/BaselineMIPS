#!/usr/bin/env python3
"""Integrated end-to-end benchmark runner (without FAISS, with PAG)."""

from __future__ import annotations

import argparse
import datetime
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

from common import RESULTS_DIR, DatasetConfig, flatten_points, load_dataset_groundtruth, read_bin
from result_plot import plot_results, save_results, update_algorithm_section


RUNNER_MODULES = {
    "mag": "benchmark_mag",
    "scann": "benchmark_scann",
    "ipnsw": "benchmark_ipnsw",
    "mobius": "benchmark_mobius",
    "pag": "benchmark_pag",
}

# DatasetConfig field names that carry sweep / tuning parameters for each algo.
ALGO_PARAM_FIELDS: dict[str, list[str]] = {
    "mag": ["mag_efs"],
    "scann": [
        "scann_distance",
        "scann_mode",
        "scann_num_leaves",
        "scann_leaves_to_search",
        "scann_reorder_values",
        "scann_leaves_values",
    ],
    "ipnsw": ["ipnsw_m", "ipnsw_ef_construction", "ipnsw_ef_values"],
    "mobius": ["mobius_budget_values"],
    "pag": ["pag_hnsw_efc", "pag_hnsw_M", "pag_hnsw_L"],
}


def run_algorithm(name: str, cfg, database, queries, ground_truth):
    mod = importlib.import_module(RUNNER_MODULES[name])
    if name in {"mag", "ipnsw", "mobius"}:
        return mod.run(cfg, ground_truth)
    if name == "pag":
        return mod.run(cfg)
    return mod.run(cfg, database, queries, ground_truth)


def build_metadata(cfg: DatasetConfig, selected_algos: list[str]) -> dict:
    """Build a metadata dict to embed as comments in the result file."""
    metadata: dict = {
        "dataset": cfg.name,
        "db_size": cfg.db_size,
        "dim": cfg.dim,
        "query_size": cfg.query_size,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    params: dict = {}
    for algo in selected_algos:
        algo_params: dict = {}
        for field in ALGO_PARAM_FIELDS.get(algo, []):
            val = getattr(cfg, field, None)
            if val is not None:
                algo_params[field] = val
        if algo_params:
            params[algo] = algo_params
    if params:
        metadata["params"] = params
    return metadata


def find_latest_result_file(dataset: str) -> Path | None:
    """Return the most recently modified ``{dataset}_*.txt`` in RESULTS_DIR."""
    files = list(RESULTS_DIR.glob(f"{dataset}_*.txt"))
    return max(files, key=lambda f: f.stat().st_mtime) if files else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full benchmark pipeline")
    parser.add_argument("--dataset", default="music100", choices=["music100", "glove100", "dinov2", "book_corpus"], help="Dataset name")
    parser.add_argument(
        "--scann-mode",
        default="reorder",
        choices=["reorder", "leaves"],
        help="ScaNN sweep mode",
    )
    parser.add_argument(
        "--algorithms",
        default="mag,scann,ipnsw,mobius,pag",
        help="Comma-separated algorithms to run",
    )
    parser.add_argument(
        "--result-txt",
        default=None,
        help=(
            "Result txt path. Defaults to auto-generated path: "
            "results/{dataset}_{timestamp}.txt for full/partial runs, "
            "or the latest existing results/{dataset}_*.txt when running a single algorithm."
        ),
    )
    parser.add_argument("--plot", default=None, help="Output plot path")
    parser.add_argument("--title", default=None, help="Plot title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(name=args.dataset)
    cfg.scann_mode = args.scann_mode

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
    cfg.db_size = int(database.shape[0])
    cfg.query_size = int(queries.shape[0])
    ground_truth = load_dataset_groundtruth(cfg)
    print(f"Database shape: {database.shape}")
    print(f"Query shape   : {queries.shape}")
    print(f"GT shape      : {ground_truth.shape}")

    selected = [name.strip().lower() for name in args.algorithms.split(",") if name.strip()]
    invalid = [name for name in selected if name not in RUNNER_MODULES]
    if invalid:
        raise ValueError(f"Unknown algorithms: {invalid}")

    # -----------------------------------------------------------------------
    # Determine result file path and whether to use update-in-place mode.
    # -----------------------------------------------------------------------
    is_full_run = set(selected) == set(RUNNER_MODULES.keys())
    is_single_algo = len(selected) == 1

    if args.result_txt is not None:
        # User supplied an explicit path — respect it unconditionally.
        result_txt = args.result_txt
        use_update_mode = False
    elif is_single_algo:
        # Single-algorithm run: update only that algorithm's section in the
        # latest existing result file for this dataset.
        latest = find_latest_result_file(cfg.name)
        if latest is not None:
            result_txt = str(latest)
            use_update_mode = True
            print(f"\n[single-algo mode] Will update {selected[0]} in: {result_txt}")
        else:
            # No prior file exists; create a new timestamped one.
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_txt = str(RESULTS_DIR / f"{cfg.name}_{ts}.txt")
            use_update_mode = False
    else:
        # Full run or partial multi-algo run: always create a new timestamped file.
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_txt = str(RESULTS_DIR / f"{cfg.name}_{ts}.txt")
        use_update_mode = False

    print("\n[2/4] Running algorithm sweeps...")
    all_points: list[dict] = []
    failed_algos: dict[str, str] = {}
    for name in selected:
        print(f"\n--- {name.upper()} ---")
        try:
            points = run_algorithm(name, cfg, database, queries, ground_truth)
            all_points.extend(flatten_points(name, points))
        except Exception as exc:
            import traceback
            err_msg = traceback.format_exc()
            failed_algos[name] = err_msg
            print(f"[WARNING] {name.upper()} failed — skipping.\n{err_msg}")

    if failed_algos:
        print("\n[FAILED ALGORITHMS]")
        for algo, msg in failed_algos.items():
            print(f"  {algo}: {msg.splitlines()[-1]}")

    if not all_points:
        raise RuntimeError("No benchmark points produced (all algorithms failed)")

    metadata = build_metadata(cfg, selected)

    print("\n[3/4] Saving result txt...")
    if use_update_mode:
        # is_single_algo is True here — update only the one algorithm.
        update_algorithm_section(all_points, selected[0], result_txt)
    else:
        save_results(all_points, result_txt, metadata=metadata)

    print("\n[4/4] Plotting curves...")
    plot_results(result_txt, plot_path, plot_title, dataset_name=cfg.name, top_k=cfg.top_k)

    print("\nDone")
    print(f"Result txt: {result_txt}")
    print(f"Plot file : {plot_path}")


if __name__ == "__main__":
    main()
