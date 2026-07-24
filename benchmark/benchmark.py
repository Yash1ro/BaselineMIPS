#!/usr/bin/env python3
"""Integrated end-to-end benchmark runner (without FAISS, with PAG)."""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
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
    "pag_new": "benchmark_pag_new",
    "pag_without_projection": "benchmark_pag_without_projection",
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
    "pag_new": ["pag_new_hnsw_efc", "pag_new_hnsw_M", "pag_new_hnsw_L"],
    "pag_without_projection": ["pag_without_proj_hnsw_efc", "pag_without_proj_hnsw_M", "pag_without_proj_hnsw_L"],
}


def run_algorithm(name: str, cfg, database, queries, ground_truth):
    mod = importlib.import_module(RUNNER_MODULES[name])
    if name in {"mag", "ipnsw", "mobius"}:
        return mod.run(cfg, ground_truth)
    if name in {"pag_new", "pag_without_projection"}:
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
    parser.add_argument("--dataset", default="music100",
                        choices=["music100", "glove100", "glove200", "dinov2", "book_corpus",
                                 "gist1m", "msong", "text2img10m", "ir101", "openai1536", "word2vec"],
                        help="Dataset name")
    parser.add_argument(
        "--scann-mode",
        default="reorder",
        choices=["reorder", "leaves"],
        help="ScaNN sweep mode",
    )
    parser.add_argument(
        "--algorithms",
        default="mag,scann,ipnsw,mobius,pag_new,pag_without_projection",
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
    parser.add_argument("--top-k", type=int, default=None, help="Override top-K (e.g. 500)")
    return parser.parse_args()


def _apply_top500_params(cfg: DatasetConfig) -> None:
    """Widen parameter sweeps so budgets cover the larger top-500 demand."""
    cfg.mag_efs            = [500, 600, 800, 1000, 1500, 2000, 3000]
    cfg.ipnsw_ef_values    = [500, 600, 800, 1000, 1500, 2000, 3000]
    cfg.mobius_budget_values = [500, 600, 800, 1000, 1500, 2000, 3000]
    cfg.scann_reorder_values = [1000, 1500, 2000, 3000, 4000, 5000, 8000, 10000]
    cfg.scann_leaves_values  = [100, 200, 500, 1000, 2000]
    cfg.pag_hnsw_efc       = 500
    cfg.pag_run_script     = "run_music100_top500.sh"


MEM_LOG_PATH = RESULTS_DIR / "log_mem.log"


def _save_mem_log(
    dataset: str,
    top_k: int,
    selected: list[str],
    build_times: dict[str, float],
    build_mem: dict[str, float],
    query_mem: dict[str, float],
    failed_algos: dict[str, str],
) -> None:
    """Append one JSON record per benchmark run to log_mem.log."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record: dict = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": dataset,
        "top_k": top_k,
        "algorithms": {},
    }
    for algo in selected:
        if algo in failed_algos:
            record["algorithms"][algo] = {"status": "failed"}
            continue
        bm = build_mem.get(algo, 0.0)
        record["algorithms"][algo] = {
            "build_time_s": round(build_times.get(algo, 0.0), 3),
            "build_peak_mb": round(bm, 1) if bm > 0 else None,
            "query_peak_mb": round(query_mem.get(algo, 0.0), 1),
        }
    with open(MEM_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Memory log  : {MEM_LOG_PATH}")


def main() -> None:
    args = parse_args()
    cfg = DatasetConfig(name=args.dataset)
    cfg.scann_mode = args.scann_mode
    if args.top_k is not None:
        cfg.top_k = args.top_k
        if cfg.name == "music100" and args.top_k == 500:
            _apply_top500_params(cfg)
        # Set PAG script for the appropriate top_k.
        if args.top_k != 100:
            cfg.pag_run_script = f"run_{cfg.name}_top{args.top_k}.sh"

    print("=" * 72)
    print("BENCHMARK PIPELINE")
    print("=" * 72)
    print(f"Dataset : {cfg.name}")
    print(f"Top-K   : {cfg.top_k}")

    plot_path = args.plot or str(THIS_DIR / "imgs" / f"{cfg.name}_top{cfg.top_k}.png")
    plot_title = args.title or f"{cfg.name} Recall-QPS Benchmark"

    print("\n[1/4] Loading data...")
    if not os.path.exists(cfg.database_bin):
        print(f"\n  ERROR: Database file not found: {cfg.database_bin}")
        print(f"  Please ensure dataset '{cfg.name}' is available under data/{cfg.name}/")
        print(f"  See README for dataset preparation instructions.\n")
        sys.exit(1)
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
    build_times: dict[str, float] = {}
    build_mem: dict[str, float] = {}
    query_mem: dict[str, float] = {}
    for name in selected:
        print(f"\n--- {name.upper()} ---")
        try:
            points, build_time, build_peak_mb, query_peak_mb = run_algorithm(name, cfg, database, queries, ground_truth)
            build_times[name] = build_time
            build_mem[name] = build_peak_mb
            query_mem[name] = query_peak_mb
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

    if build_times:
        print("\n[INDEX BUILD TIMES]")
        print(f"  {'Algorithm':<12} {'Build Time':>12}")
        print(f"  {'-'*12} {'-'*12}")
        for algo, bt in build_times.items():
            if bt > 0:
                print(f"  {algo:<12} {bt:>11.2f}s")
            else:
                print(f"  {algo:<12} {'(prebuilt)':>12}")

    if build_mem or query_mem:
        print("\n[PEAK MEMORY (MB)]")
        print(f"  {'Algorithm':<12} {'Build Peak':>12} {'Query Peak':>12}")
        print(f"  {'-'*12} {'-'*12} {'-'*12}")
        for algo in selected:
            if algo in failed_algos:
                continue
            bm = build_mem.get(algo, 0.0)
            qm = query_mem.get(algo, 0.0)
            bm_str = f"{bm:.1f} MB" if bm > 0 else "(prebuilt)"
            print(f"  {algo:<12} {bm_str:>12} {qm:>10.1f} MB")

    if not all_points:
        raise RuntimeError("No benchmark points produced (all algorithms failed)")

    metadata = build_metadata(cfg, selected)

    print("\n[3/4] Saving result txt...")
    if use_update_mode:
        # is_single_algo is True here — update only the one algorithm.
        update_algorithm_section(all_points, selected[0], result_txt)
    else:
        save_results(all_points, result_txt, metadata=metadata)

    # Save memory log.
    _save_mem_log(
        dataset=cfg.name,
        top_k=cfg.top_k,
        selected=selected,
        build_times=build_times,
        build_mem=build_mem,
        query_mem=query_mem,
        failed_algos=failed_algos,
    )

    print("\n[4/4] Plotting curves...")
    plot_results(result_txt, plot_path, plot_title, dataset_name=cfg.name, top_k=cfg.top_k)

    print("\nDone")
    print(f"Result txt: {result_txt}")
    print(f"Plot file : {plot_path}")


if __name__ == "__main__":
    main()
