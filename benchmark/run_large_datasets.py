#!/usr/bin/env python3
"""Run PAG variants on large datasets (ir101, book_corpus) with index cleanup.

Builds index, queries, saves results, then deletes the index to save disk space.
This allows sequential testing of all 3 variants even with limited disk.

Usage:
    cd /home/gu/baseline
    source exp/bin/activate
    python benchmark/run_large_datasets.py
"""

from __future__ import annotations

import datetime
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "tools"))

from common import PeakMemoryTracker, build_env_without_thread_limits
from result_plot import (
    STYLE_MAP,
    _pareto_frontier,
    load_results,
    save_results,
    update_algorithm_section,
    plot_qps_recall_multi,
    plot_results,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = THIS_DIR / "results"
IMGS_DIR = THIS_DIR / "imgs"

VARIANTS = {
    "pag":                    ROOT / "PAG",
    "pag_without_projection": ROOT / "PAG_without_projection",
    "pag_new":                ROOT / "PAG_new",
}

DATASETS = ["ir101", "book_corpus"]
TOP_K = 10


def _get_index_dir(variant_dir: Path, script_name: str, dataset: str) -> Path:
    script_path = variant_dir / script_name
    for line in script_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("iPath="):
            val = line.split("=", 1)[1].strip()
            if "#" in val:
                val = val[:val.index("#")].strip()
            val = val.strip('"').strip("'")
            val = val.replace("${name}", dataset)
            return variant_dir / val
    return variant_dir / dataset / "index"


def run_variant(variant_name: str, variant_dir: Path, dataset: str) -> list[dict]:
    script_name = f"run_{dataset}_top{TOP_K}.sh"
    script_path = variant_dir / script_name
    if not script_path.exists():
        print(f"  [{variant_name}] script not found: {script_path}")
        return []

    index_dir = _get_index_dir(variant_dir, script_name, dataset)
    has_index = index_dir.exists() and index_dir.is_dir() and any(index_dir.iterdir())
    run_times = 1 if has_index else 2

    last_stdout = ""
    for i in range(run_times):
        is_build = (run_times == 2 and i == 0)
        phase = "BUILD" if is_build else "QUERY"
        print(f"  [{variant_name}] {phase} ({i+1}/{run_times}): {script_name}")

        env = build_env_without_thread_limits() if is_build else os.environ.copy()
        t0 = time.time()
        try:
            result = subprocess.run(
                ["bash", script_name],
                cwd=str(variant_dir),
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            elapsed = time.time() - t0
            last_stdout = result.stdout
            print(f"  [{variant_name}] {phase} done in {elapsed:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"  [{variant_name}] ERROR: {e}")
            if e.stderr:
                print(f"  STDERR: {e.stderr[-1000:]}")
            return []

    # Parse results
    points = []
    for line in last_stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+QPS\s*$", line)
        if m:
            points.append({
                "budget": int(m.group(1)),
                "recall": float(m.group(2)),
                "qps": float(m.group(3)),
            })
    print(f"  [{variant_name}] parsed {len(points)} points")

    # Delete index to free disk space (only for pag_new / pag_without_projection)
    if variant_name != "pag" and index_dir.exists():
        size_bytes = sum(f.stat().st_size for f in index_dir.rglob("*") if f.is_file())
        size_gb = size_bytes / (1024**3)
        print(f"  [{variant_name}] deleting index {index_dir} ({size_gb:.1f}G)")
        shutil.rmtree(index_dir)

    return points


def merge_to_benchmark(dataset: str, variant_name: str, points: list[dict]) -> None:
    """Update existing benchmark result file and re-plot."""
    candidates = sorted(
        [f for f in RESULTS_DIR.glob(f"{dataset}_top{TOP_K}_*.txt") if f.stat().st_size > 0],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"  No existing result file for {dataset} top-{TOP_K}")
        return

    result_file = candidates[0]
    flat = [{"algorithm": variant_name, "budget": p["budget"],
             "recall": p["recall"], "qps": p["qps"]} for p in points]
    try:
        update_algorithm_section(flat, variant_name, str(result_file))
        print(f"Updated {len(flat)} {variant_name} rows in {result_file}")
    except Exception as e:
        print(f"  update_algorithm_section failed: {e}")
        return

    from result_plot import plot_results
    plot_path = str(IMGS_DIR / f"{dataset}_top{TOP_K}.png")
    try:
        plot_results(str(result_file), plot_path,
                     f"{dataset} Recall-QPS (top-{TOP_K})",
                     dataset_name=dataset, top_k=TOP_K)
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"  plot_results failed: {e}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    all_points: dict[str, dict[str, list[dict]]] = {}  # dataset -> variant -> points

    for ds in DATASETS:
        print(f"\n{'='*60}")
        print(f"  DATASET: {ds}  top-{TOP_K}")
        print(f"{'='*60}")
        all_points[ds] = {}

        for vname, vdir in VARIANTS.items():
            print(f"\n--- {vname} ---")
            pts = run_variant(vname, vdir, ds)
            all_points[ds][vname] = pts
            if pts:
                merge_to_benchmark(ds, vname, pts)

            # Show disk space after each variant
            import subprocess as sp
            df_out = sp.run(["df", "-h", "/home"], capture_output=True, text=True).stdout
            for line in df_out.splitlines():
                if "/home" in line:
                    print(f"  Disk: {line.strip()}")

    print(f"\n{'='*60}")
    print("ALL RUNS COMPLETE")
    print(f"{'='*60}")

    # Print summary
    for ds in DATASETS:
        print(f"\n{ds} top-{TOP_K}:")
        for vname, pts in all_points[ds].items():
            if pts:
                best = max(pts, key=lambda p: p["recall"])
                print(f"  {vname}: {len(pts)} points, best recall={best['recall']:.4f} @ {best['qps']:.0f} QPS")
            else:
                print(f"  {vname}: FAILED")

    # -----------------------------------------------------------------------
    # Regenerate all_datasets_top10_comparison.png with all 8 datasets
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Regenerating all_datasets_top10_comparison.png ...")
    print(f"{'='*60}")

    ALL_8_DATASETS = [
        "music100", "glove100", "glove200", "dinov2",
        "book_corpus", "gist1m", "ir101", "openai1536",
    ]

    dataset_points: dict[str, list[dict]] = {}
    for ds in ALL_8_DATASETS:
        candidates = sorted(
            [f for f in RESULTS_DIR.glob(f"{ds}_top{TOP_K}_*.txt") if f.stat().st_size > 0],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            print(f"  [{ds}] No result file found, skipping")
            continue

        result_file = candidates[0]
        points = load_results(str(result_file))
        if not points:
            print(f"  [{ds}] Empty result file, skipping")
            continue

        algos = sorted(set(p["algorithm"] for p in points))
        print(f"  [{ds}] {result_file.name}: {algos}")
        dataset_points[ds] = points

        # Also regenerate individual dataset plot
        plot_path = str(IMGS_DIR / f"{ds}_top{TOP_K}.png")
        try:
            plot_results(str(result_file), plot_path,
                         f"{ds} Recall-QPS (top-{TOP_K})",
                         dataset_name=ds, top_k=TOP_K)
        except Exception as e:
            print(f"  [{ds}] plot_results failed: {e}")

    if dataset_points:
        combo_path = str(IMGS_DIR / f"all_datasets_top{TOP_K}_comparison.png")
        plot_qps_recall_multi(
            dataset_points, combo_path,
            title=f"All Datasets — Recall-QPS (top-{TOP_K})",
            top_k=TOP_K,
        )
        print(f"\nDone! {combo_path}")
    else:
        print("No dataset results available for combined plot.")


if __name__ == "__main__":
    main()
