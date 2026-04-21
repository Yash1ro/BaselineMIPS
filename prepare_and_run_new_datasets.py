#!/usr/bin/env python3
"""Prepare and run PAG/PAG_new/PAG_without_projection benchmarks for new datasets.

Datasets: laion12m, msong, tiny5m, word2vec
Steps:
  1. Extract compressed archives if needed
  2. Convert fvecs → raw float32 bin
  3. Generate ground truth (top-100 brute force)
  4. Create/update run scripts for all 3 PAG variants
  5. Run benchmarks (index build + query)
  6. Generate comparison plots

Usage:
    cd /home/gu/baseline
    source exp/bin/activate
    python prepare_and_run_new_datasets.py
    python prepare_and_run_new_datasets.py --datasets laion12m msong
    python prepare_and_run_new_datasets.py --skip-run   # prepare only
    python prepare_and_run_new_datasets.py --skip-prepare  # run only
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Thread limits for fair measurement
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

ROOT = Path(__file__).resolve().parent
PAG_DIR = ROOT / "PAG"
PAG_NEW_DIR = ROOT / "PAG_new"
PAG_WP_DIR = ROOT / "PAG_without_projection"
BENCHMARK_DIR = ROOT / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
IMGS_DIR = BENCHMARK_DIR / "imgs"

sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "tools"))

from common import PeakMemoryTracker, build_env_without_thread_limits
from result_plot import (
    STYLE_MAP,
    _pareto_frontier,
    load_results,
    plot_qps_recall_multi,
    plot_results,
    save_results,
    update_algorithm_section,
)

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASETS = {
    "laion12m": {
        "n": 12000000, "d": 512, "qn": 1000,
        "efc": 1000, "M": 64, "L": 32,
        "metric": "ip",
        "source": "ready",  # already in PAG/laion12m/
    },
    "msong": {
        "n": 994185, "d": 420, "qn": 1000,
        "efc": 1000, "M": 64, "L": 32,
        "metric": "ip",
        "source": "ready",  # already in PAG/msong/
    },
    "tiny5m": {
        "n": 5000000, "d": 384, "qn": 1000,
        "efc": 1000, "M": 64, "L": 32,
        "metric": "ip",
        "source": "tar.gz",
        "archive": str(ROOT / "tiny5m.tar.gz"),
        "fvecs_dir": str(ROOT / "tiny5m"),
    },
    "word2vec": {
        "n": 1000000, "d": 300, "qn": 1000,
        "efc": 1000, "M": 64, "L": 32,
        "metric": "ip",
        "source": "tar.gz",
        "archive": str(ROOT / "word2vec.tar.gz"),
        "fvecs_dir": str(ROOT / "word2vec"),
    },
}

VARIANTS = {
    "pag":    PAG_DIR,
    "pag_new": PAG_NEW_DIR,
    "pag_without_projection": PAG_WP_DIR,
}

INDEX_PREFIX = {
    "pag": "index_pag",
    "pag_new": "index_pag_new",
    "pag_without_projection": "index_pag_without_projection",
}

TOP_K = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_fvecs(path: str) -> np.ndarray:
    """Read .fvecs file (int32 dim header per row + float32 data)."""
    with open(path, "rb") as f:
        d = struct.unpack("i", f.read(4))[0]
        f.seek(0)
        row_bytes = 4 + d * 4  # int32 dim + d floats
        file_size = os.path.getsize(path)
        n = file_size // row_bytes
        if file_size != n * row_bytes:
            raise ValueError(f"{path}: size {file_size} not divisible by row_bytes {row_bytes}")
        data = np.fromfile(f, dtype=np.int32).reshape(n, d + 1)
        return data[:, 1:].view(np.float32).copy()


def read_bin(path: str, dim: int) -> np.ndarray:
    """Read raw float32 binary file."""
    file_size = os.path.getsize(path)
    n = file_size // (dim * 4)
    return np.fromfile(path, dtype=np.float32).reshape(n, dim)


def generate_groundtruth(database: np.ndarray, queries: np.ndarray,
                         top_k: int, metric: str = "ip",
                         batch_size: int = 200) -> np.ndarray:
    """Compute exact top-K neighbors via batched brute-force."""
    thread_keys = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    saved = {k: os.environ.pop(k, None) for k in thread_keys}
    try:
        n_queries = queries.shape[0]
        gt = np.empty((n_queries, top_k), dtype=np.int32)
        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            q_batch = queries[start:end]
            if metric == "ip":
                scores = q_batch @ database.T
                idx = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
                for i in range(end - start):
                    top_idx = idx[i]
                    gt[start + i] = top_idx[np.argsort(-scores[i, top_idx])]
            else:
                dists = np.sum((q_batch[:, None, :] - database[None, :, :]) ** 2, axis=2)
                idx = np.argpartition(dists, top_k, axis=1)[:, :top_k]
                for i in range(end - start):
                    top_idx = idx[i]
                    gt[start + i] = top_idx[np.argsort(dists[i, top_idx])]
            if (start // batch_size) % 10 == 0:
                print(f"    GT progress: {end}/{n_queries}")
        return gt
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Step 1: Extract archives
# ---------------------------------------------------------------------------

def extract_archive(ds_name: str, info: dict) -> bool:
    """Extract tar.gz if source is archive and not yet extracted."""
    if info["source"] != "tar.gz":
        return True

    archive = info["archive"]
    fvecs_dir = info["fvecs_dir"]
    base_fvecs = os.path.join(fvecs_dir, f"{ds_name}_base.fvecs")

    if os.path.exists(base_fvecs):
        print(f"  [{ds_name}] fvecs already extracted")
        return True

    if not os.path.exists(archive):
        print(f"  [{ds_name}] archive not found: {archive}")
        return False

    # Verify archive integrity
    try:
        with tarfile.open(archive, "r:gz") as tf:
            members = tf.getnames()
            print(f"  [{ds_name}] extracting {len(members)} files from {archive}")
            tf.extractall(path=str(ROOT))
    except Exception as e:
        print(f"  [{ds_name}] extraction failed (file may still be uploading): {e}")
        return False

    if not os.path.exists(base_fvecs):
        print(f"  [{ds_name}] expected {base_fvecs} not found after extraction")
        return False

    return True


# ---------------------------------------------------------------------------
# Step 2: Convert fvecs → bin + auto-detect dimensions
# ---------------------------------------------------------------------------

def convert_and_prepare(ds_name: str, info: dict) -> bool:
    """Convert fvecs to bin format and create shared data in PAG/{dataset}/."""
    data_dir = PAG_DIR / ds_name
    base_bin = data_dir / f"{ds_name}_base.bin"
    query_bin = data_dir / f"{ds_name}_query.bin"
    truth_bin = data_dir / f"{ds_name}_truth.bin"
    meta_json = data_dir / f"{ds_name}_meta.json"

    # If all files exist, skip
    if base_bin.exists() and query_bin.exists() and truth_bin.exists():
        print(f"  [{ds_name}] all bin files already exist in {data_dir}")
        # Auto-detect actual dimensions from existing files
        actual_n = os.path.getsize(base_bin) // (info["d"] * 4)
        actual_qn = os.path.getsize(query_bin) // (info["d"] * 4)
        if actual_n != info["n"]:
            print(f"  [{ds_name}] WARNING: actual n={actual_n} vs expected n={info['n']}, updating")
            info["n"] = actual_n
        if actual_qn != info["qn"]:
            print(f"  [{ds_name}] WARNING: actual qn={actual_qn} vs expected qn={info['qn']}, updating")
            info["qn"] = actual_qn
        return True

    data_dir.mkdir(parents=True, exist_ok=True)

    if info["source"] == "tar.gz":
        fvecs_dir = info["fvecs_dir"]
        base_fvecs = os.path.join(fvecs_dir, f"{ds_name}_base.fvecs")
        query_fvecs = os.path.join(fvecs_dir, f"{ds_name}_query.fvecs")

        if not os.path.exists(base_fvecs):
            print(f"  [{ds_name}] fvecs not found: {base_fvecs}")
            return False

        # Auto-detect dimensions from fvecs
        with open(base_fvecs, "rb") as f:
            actual_dim = struct.unpack("i", f.read(4))[0]
        if actual_dim != info["d"]:
            print(f"  [{ds_name}] auto-detected dim={actual_dim} (was {info['d']})")
            info["d"] = actual_dim

        # Convert base
        if not base_bin.exists():
            print(f"  [{ds_name}] converting base fvecs → bin ...")
            base_data = read_fvecs(base_fvecs)
            info["n"] = base_data.shape[0]
            print(f"  [{ds_name}] base shape: {base_data.shape}")
            base_data.tofile(str(base_bin))
        else:
            base_data = read_bin(str(base_bin), info["d"])
            info["n"] = base_data.shape[0]

        # Convert query
        if not query_bin.exists():
            if os.path.exists(query_fvecs):
                print(f"  [{ds_name}] converting query fvecs → bin ...")
                query_data = read_fvecs(query_fvecs)
                info["qn"] = query_data.shape[0]
                print(f"  [{ds_name}] query shape: {query_data.shape}")
                query_data.tofile(str(query_bin))
            else:
                # If no separate query file, sample from base
                print(f"  [{ds_name}] no query fvecs found, sampling {info['qn']} from base")
                rng = np.random.RandomState(42)
                idx = rng.choice(base_data.shape[0], size=min(info["qn"], base_data.shape[0]), replace=False)
                query_data = base_data[idx]
                query_data.tofile(str(query_bin))
        else:
            query_data = read_bin(str(query_bin), info["d"])
            info["qn"] = query_data.shape[0]

        # Generate ground truth (top-100 to support various top-k)
        if not truth_bin.exists():
            print(f"  [{ds_name}] generating ground truth (top-100) ...")
            if 'base_data' not in dir() or base_data is None:
                base_data = read_bin(str(base_bin), info["d"])
            if 'query_data' not in dir() or query_data is None:
                query_data = read_bin(str(query_bin), info["d"])
            gt = generate_groundtruth(base_data, query_data, 100, metric=info["metric"])
            gt.tofile(str(truth_bin))
            print(f"  [{ds_name}] ground truth saved: {gt.shape}")

    # Save metadata
    meta = {
        "dataset": ds_name,
        "n": info["n"],
        "d": info["d"],
        "qn": info["qn"],
        "efc": info["efc"],
        "M": info["M"],
        "L": info["L"],
        "metric": info["metric"],
        "truth_k": 100,
    }
    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  [{ds_name}] data ready in {data_dir}")
    return True


# ---------------------------------------------------------------------------
# Step 3: Create run scripts for all 3 variants
# ---------------------------------------------------------------------------

SCRIPT_TEMPLATE_PAG = """\
# ------------------------------------------------------------------------------
#  Parameters  (auto-generated shared-data script, top-{top_k})
# ------------------------------------------------------------------------------
name={name}
n={n}  #data size
d={d}  #dimension
qn={qn}  #query size
k={top_k}  #topk

efc={efc}  #HNSW parameter
M={M}  #HNSW parameter
L={L}  #level

dPath={data_prefix}/${{name}}/${{name}}_base.bin  #shared data path
qPath={data_prefix}/${{name}}/${{name}}_query.bin  #shared query path
tPath={data_prefix}/${{name}}/${{name}}_truth.bin  #shared groundtruth path
iPath=./{index_prefix}_${{name}}_top{top_k}/  #algorithm-specific index path

#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${{dPath}} ${{qPath}} ${{tPath}} ${{iPath}} ${{n}} ${{qn}} ${{d}} ${{k}} ${{efc}} ${{M}} ${{L}}
"""


def create_run_scripts(ds_name: str, info: dict) -> None:
    """Create top-10 run scripts for all 3 PAG variants."""
    params = {
        "name": ds_name,
        "n": info["n"],
        "d": info["d"],
        "qn": info["qn"],
        "top_k": TOP_K,
        "efc": info["efc"],
        "M": info["M"],
        "L": info["L"],
    }

    for variant, vdir in VARIANTS.items():
        script_name = f"run_{ds_name}_top{TOP_K}.sh"
        script_path = vdir / script_name

        if variant == "pag":
            data_prefix = "."
        else:
            data_prefix = "../PAG"

        content = SCRIPT_TEMPLATE_PAG.format(
            **params,
            data_prefix=data_prefix,
            index_prefix=INDEX_PREFIX[variant],
        )
        script_path.write_text(content)
        os.chmod(str(script_path), 0o755)
        print(f"  [{ds_name}] created {script_path}")


# ---------------------------------------------------------------------------
# Step 4: Run benchmarks
# ---------------------------------------------------------------------------

def run_variant(variant: str, vdir: Path, ds_name: str) -> tuple[list[dict], float, float, float]:
    """Run one PAG variant for one dataset. Returns (points, build_time, build_peak, query_peak)."""
    script_name = f"run_{ds_name}_top{TOP_K}.sh"
    script_path = vdir / script_name
    if not script_path.exists():
        print(f"  [{variant}] script not found: {script_path}")
        return [], 0.0, 0.0, 0.0

    # Determine index dir from script
    index_dir = None
    for line in script_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("iPath="):
            val = line.split("=", 1)[1].strip()
            if "#" in val:
                val = val[:val.index("#")].strip()
            val = val.strip('"').strip("'")
            val = val.replace("${name}", ds_name)
            index_dir = vdir / val
            break

    if index_dir is None:
        index_dir = vdir / ds_name / "index"

    has_index = index_dir.exists() and index_dir.is_dir() and any(index_dir.iterdir())
    run_times = 1 if has_index else 2

    points = []
    last_stdout = ""
    build_time = 0.0
    build_peak_mb = 0.0
    query_peak_mb = 0.0

    for i in range(run_times):
        print(f"  [{variant}] run {i + 1}/{run_times}: {script_name}")
        try:
            if run_times == 2 and i == 0:
                env = build_env_without_thread_limits()
                t0 = time.time()
                with PeakMemoryTracker() as bt:
                    result = subprocess.run(
                        ["bash", script_name],
                        cwd=str(vdir),
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                build_time = time.time() - t0
                build_peak_mb = bt.peak_mb
                print(f"  [{variant}] build={build_time:.1f}s  peak={build_peak_mb:.1f}MB")
                print(f"  [{variant}] stdout (last 500 chars): {result.stdout[-500:]}")
            else:
                env = os.environ.copy()
                with PeakMemoryTracker() as qt:
                    result = subprocess.run(
                        ["bash", script_name],
                        cwd=str(vdir),
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                    )
                query_peak_mb = qt.peak_mb
            last_stdout = result.stdout
        except subprocess.CalledProcessError as e:
            print(f"  [{variant}] ERROR: {e}")
            print(f"  STDERR: {e.stderr[-2000:] if e.stderr else ''}")
            return [], build_time, build_peak_mb, query_peak_mb

    for line in last_stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+([0-9eE.+-]+)\s+([0-9eE.+-]+)\s+QPS\s*$", line)
        if m:
            points.append({"budget": int(m.group(1)), "recall": float(m.group(2)), "qps": float(m.group(3))})

    print(f"  [{variant}] parsed {len(points)} points")
    return points, build_time, build_peak_mb, query_peak_mb


# ---------------------------------------------------------------------------
# Step 5: Generate plots
# ---------------------------------------------------------------------------

def plot_pag_variants_comparison(
    dataset_results: dict[str, dict[str, list[dict]]],
    output_png: str,
    top_k: int,
) -> None:
    """Multi-panel comparison plot: one subplot per dataset."""
    ds_list = [ds for ds in dataset_results if dataset_results[ds]]
    if not ds_list:
        return

    cols = min(len(ds_list), 4)
    rows = (len(ds_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    for idx, ds in enumerate(ds_list):
        ax = axes[idx // cols][idx % cols]
        for vname, points in dataset_results[ds].items():
            if not points:
                continue
            pts = _pareto_frontier(points)
            recalls = [p["recall"] * 100 for p in pts]
            qps = [p["qps"] for p in pts]
            style = STYLE_MAP.get(vname, {"marker": "o", "linestyle": "-", "color": None})
            ax.plot(recalls, qps,
                    marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], linewidth=2.5, markersize=8, label=vname)

        ax.set_xlabel(f"Recall@{top_k} (%)", fontsize=13)
        ax.set_ylabel("QPS", fontsize=13)
        ax.set_title(ds, fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=11)
        ax.tick_params(axis="both", labelsize=11)

    for idx in range(len(ds_list), rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(f"PAG Variants Comparison — top-{top_k}", fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot: {output_png}")


def add_to_benchmark_plot(dataset: str, variant_name: str, points: list[dict]) -> None:
    """Merge variant results into existing benchmark result file and re-plot."""
    candidates = sorted(
        [f for f in RESULTS_DIR.glob(f"{dataset}_top{TOP_K}_*.txt") if f.stat().st_size > 0],
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        # Create a new result file
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = RESULTS_DIR / f"{dataset}_top{TOP_K}_{ts}.txt"
        flat = [{"algorithm": variant_name, "budget": p["budget"],
                 "recall": p["recall"], "qps": p["qps"]} for p in points]
        save_results(flat, str(result_file), metadata={
            "dataset": dataset,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "top_k": TOP_K,
        })
    else:
        result_file = candidates[0]
        flat = [{"algorithm": variant_name, "budget": p["budget"],
                 "recall": p["recall"], "qps": p["qps"]} for p in points]
        try:
            update_algorithm_section(flat, variant_name, str(result_file))
        except Exception as e:
            print(f"  [PLOT] update_algorithm_section failed: {e}")
            return

    # Re-plot per-dataset
    plot_path = str(IMGS_DIR / f"{dataset}_top{TOP_K}.png")
    try:
        plot_results(str(result_file), plot_path,
                     f"{dataset} Recall-QPS (top-{TOP_K})",
                     dataset_name=dataset, top_k=TOP_K)
    except Exception as e:
        print(f"  [PLOT] plot_results failed: {e}")


def generate_all_datasets_multi_plot(all_results: dict[str, dict[str, list[dict]]]) -> None:
    """Generate the multi-dataset comparison line chart (most important plot)."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build dataset_points for plot_qps_recall_multi
    # First, load existing results from ALL datasets (including previously tested ones)
    ALL_KNOWN_DATASETS = [
        "music100", "glove100", "glove200", "dinov2", "book_corpus",
        "gist1m", "ir101", "openai1536",
        "laion12m", "msong", "tiny5m", "word2vec",
    ]

    dataset_points: dict[str, list[dict]] = {}

    for ds in ALL_KNOWN_DATASETS:
        # Try to load from existing result files
        result_files = sorted(
            [f for f in RESULTS_DIR.glob(f"{ds}_top{TOP_K}_*.txt") if f.stat().st_size > 0],
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if result_files:
            points = load_results(str(result_files[0]))
            if points:
                dataset_points[ds] = points

    # Override with freshly computed results for this run
    for ds, variant_data in all_results.items():
        # Merge fresh data with any existing data
        existing = {}
        if ds in dataset_points:
            for p in dataset_points[ds]:
                existing.setdefault(p["algorithm"], []).append(p)

        for vname, pts in variant_data.items():
            if pts:
                existing[vname] = [{"algorithm": vname, "budget": p["budget"],
                                     "recall": p["recall"], "qps": p["qps"]} for p in pts]

        combined = []
        for algo_pts in existing.values():
            combined.extend(algo_pts)
        if combined:
            dataset_points[ds] = combined

    if dataset_points:
        # Plot all datasets multi-panel comparison
        combo_path = str(IMGS_DIR / f"all_datasets_top{TOP_K}_comparison.png")
        plot_qps_recall_multi(
            dataset_points, combo_path,
            title=f"All Datasets — Recall-QPS (top-{TOP_K})",
            top_k=TOP_K,
        )
        print(f"  Updated all-datasets comparison plot: {combo_path}")

    # Also generate a PAG-variants-only comparison plot for the new datasets
    pag_only: dict[str, list[dict]] = {}
    for ds, variant_data in all_results.items():
        if any(variant_data.values()):
            combined = []
            for vname, pts in variant_data.items():
                for p in pts:
                    combined.append({"algorithm": vname, "budget": p["budget"],
                                      "recall": p["recall"], "qps": p["qps"]})
            if combined:
                pag_only[ds] = combined

    if pag_only:
        pag_combo = str(IMGS_DIR / f"pag_variants_new_datasets_top{TOP_K}_{ts}.png")
        plot_qps_recall_multi(
            pag_only, pag_combo,
            title=f"PAG Variants — New Datasets (top-{TOP_K})",
            top_k=TOP_K,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare and run PAG benchmarks for new datasets")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()),
                   choices=list(DATASETS.keys()))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()),
                   choices=list(VARIANTS.keys()))
    p.add_argument("--skip-prepare", action="store_true", help="Skip data preparation")
    p.add_argument("--skip-run", action="store_true", help="Skip benchmark execution")
    p.add_argument("--skip-plot", action="store_true", help="Skip plot generation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ready_datasets = []

    print("=" * 70)
    print("PAG NEW DATASETS — PREPARE & BENCHMARK")
    print("=" * 70)
    print(f"Datasets : {args.datasets}")
    print(f"Variants : {args.variants}")
    print("=" * 70)

    # --- Phase 1: Prepare data ---
    if not args.skip_prepare:
        print("\n" + "=" * 70)
        print("PHASE 1: DATA PREPARATION")
        print("=" * 70)

        for ds_name in args.datasets:
            info = DATASETS[ds_name]
            print(f"\n--- {ds_name} ---")

            # Extract archive if needed
            if not extract_archive(ds_name, info):
                print(f"  [{ds_name}] SKIPPED (archive not ready)")
                continue

            # Convert and prepare data
            if not convert_and_prepare(ds_name, info):
                print(f"  [{ds_name}] SKIPPED (conversion failed)")
                continue

            # Create run scripts
            create_run_scripts(ds_name, info)
            ready_datasets.append(ds_name)

        print(f"\nReady datasets: {ready_datasets}")
    else:
        ready_datasets = args.datasets

    # --- Phase 2: Run benchmarks ---
    all_results: dict[str, dict[str, list[dict]]] = {}

    if not args.skip_run:
        print("\n" + "=" * 70)
        print("PHASE 2: RUNNING BENCHMARKS")
        print("=" * 70)

        for ds_name in ready_datasets:
            print(f"\n{'─' * 70}")
            print(f"  Dataset: {ds_name}  top-{TOP_K}")
            print(f"{'─' * 70}")

            all_results[ds_name] = {}

            for vname in args.variants:
                vdir = VARIANTS[vname]
                try:
                    pts, bt, bpeak, qpeak = run_variant(vname, vdir, ds_name)
                    all_results[ds_name][vname] = pts

                    if pts:
                        add_to_benchmark_plot(ds_name, vname, pts)

                except Exception:
                    print(f"  [{vname}] FAILED:\n{traceback.format_exc()}")
                    all_results[ds_name][vname] = []

    # --- Phase 3: Generate plots ---
    if not args.skip_plot and all_results:
        print("\n" + "=" * 70)
        print("PHASE 3: GENERATING PLOTS")
        print("=" * 70)

        # Save comparison result file
        comparison_rows = []
        for ds, vdata in all_results.items():
            for vname, pts in vdata.items():
                for p in pts:
                    comparison_rows.append({
                        "algorithm": vname, "budget": p["budget"],
                        "recall": p["recall"], "qps": p["qps"],
                    })
        if comparison_rows:
            comp_txt = str(RESULTS_DIR / f"pag_new_datasets_top{TOP_K}_{ts}.txt")
            save_results(comparison_rows, comp_txt, metadata={
                "dataset": "new_datasets",
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "top_k": TOP_K,
            })

        # PAG variants comparison (multi-panel)
        if all_results:
            cmp_png = str(IMGS_DIR / f"pag_all_variants_new_datasets_top{TOP_K}_{ts}.png")
            plot_pag_variants_comparison(all_results, cmp_png, TOP_K)

        # THE MOST IMPORTANT PLOT: all-datasets multi-comparison
        generate_all_datasets_multi_plot(all_results)

    print("\n" + "=" * 70)
    print("DONE")
    print(f"Ready datasets tested: {list(all_results.keys()) if all_results else ready_datasets}")
    if all_results:
        for ds, vdata in all_results.items():
            for vname, pts in vdata.items():
                if pts:
                    recalls = [p['recall'] for p in pts]
                    print(f"  {ds}/{vname}: {len(pts)} points, recall range [{min(recalls):.4f}, {max(recalls):.4f}]")
    print("=" * 70)


if __name__ == "__main__":
    main()
