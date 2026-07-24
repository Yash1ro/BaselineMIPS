#!/usr/bin/env python3
"""Comprehensive benchmark runner: all datasets × all algorithms × top-{10, 100, 500}.

Tracks peak memory, index build time, and saves everything to statistics.log
with timestamps. At the end, prints a per-dataset summary of index build times
and total elapsed times.

Usage:
    python benchmark/run_full_benchmark.py
    python benchmark/run_full_benchmark.py --datasets music100 glove100
    python benchmark/run_full_benchmark.py --top-ks 10 100
    python benchmark/run_full_benchmark.py --algorithms mag ipnsw
    python benchmark/run_full_benchmark.py --skip-gt-gen   # skip top-500 gt generation
    python benchmark/run_full_benchmark.py --clean-indexes # clean indexes after each dataset
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Thread limits: single-thread for fair QPS measurement.
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
TOOLS_DIR = THIS_DIR / "tools"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS_DIR))

from common import (
    RESULTS_DIR,
    DATA_DIR,
    DatasetConfig,
    PeakMemoryTracker,
    build_env_without_thread_limits,
    compute_recall,
    flatten_points,
    load_dataset_groundtruth,
    load_groundtruth_auto,
    read_bin,
)
from result_plot import plot_results, save_results, plot_memory_bar_chart, plot_qps_recall_multi, load_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "music100",
    "glove100",
    "glove200",
    "dinov2",
    "book_corpus",
    "gist1m",
    "ir101",
    "openai1536",
    "word2vec",
    "msong",
    "text2img10m",
]
ALL_ALGORITHMS = ["mag", "scann", "ipnsw", "mobius", "pag_new", "pag_without_projection"]
ALL_TOP_KS = [10, 100, 500]

# Optional dataset-specific restrictions.
# Keep empty by default so newly added datasets participate in the same
# full-algorithm flow as existing ones.
DATASET_ALLOWED_ALGORITHMS: dict[str, set[str]] = {}
DATASET_SUPPORTED_TOP_KS: dict[str, set[int]] = {}

STATS_LOG = ROOT / "statistics.log"

RUNNER_MODULES = {
    "mag": "benchmark_mag",
    "scann": "benchmark_scann",
    "ipnsw": "benchmark_ipnsw",
    "mobius": "benchmark_mobius",
    "pag_new": "benchmark_pag_new",
    "pag_without_projection": "benchmark_pag_without_projection",
}

# Metric for brute-force groundtruth generation.
DATASET_METRIC = {
    "music100": "ip",
    "glove100": "ip",
    "glove200": "ip",
    "dinov2": "l2",
    "book_corpus": "ip",
    "gist1m": "ip",
    "ir101": "ip",
    "openai1536": "ip",
    "word2vec": "ip",
    "msong": "ip",
    "text2img10m": "ip",
}

PAG_DIR = ROOT / "PAG"


# ---------------------------------------------------------------------------
# Groundtruth generation
# ---------------------------------------------------------------------------

def _generate_groundtruth_brute(database: np.ndarray, queries: np.ndarray,
                                top_k: int, metric: str = "ip",
                                batch_size: int = 200) -> np.ndarray:
    """Compute exact top-K neighbours via batched brute-force.

    Temporarily lifts thread limits so NumPy BLAS runs multi-threaded.
    """
    thread_keys = [
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
    ]
    saved = {k: os.environ.pop(k, None) for k in thread_keys}

    try:
        n_queries = queries.shape[0]
        gt = np.empty((n_queries, top_k), dtype=np.int32)

        if metric == "l2":
            d_norms = np.sum(database ** 2, axis=1)  # (n_db,)

        for start in range(0, n_queries, batch_size):
            end = min(start + batch_size, n_queries)
            q_batch = queries[start:end]

            if metric == "ip":
                scores = q_batch @ database.T  # higher = better
                # argpartition: pick top_k largest
                idx = np.argpartition(-scores, top_k, axis=1)[:, :top_k]
                for i in range(end - start):
                    top_idx = idx[i]
                    gt[start + i] = top_idx[np.argsort(-scores[i, top_idx])]
            else:
                q_norms = np.sum(q_batch ** 2, axis=1, keepdims=True)
                dists = q_norms + d_norms - 2.0 * (q_batch @ database.T)
                idx = np.argpartition(dists, top_k, axis=1)[:, :top_k]
                for i in range(end - start):
                    top_idx = idx[i]
                    gt[start + i] = top_idx[np.argsort(dists[i, top_idx])]

            if (start // batch_size) % 5 == 0:
                print(f"    GT progress: {end}/{n_queries}")

        return gt
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _get_gt_path(cfg: DatasetConfig, top_k: int) -> str:
    """Return the expected groundtruth binary path for a given top_k."""
    if top_k <= 100:
        return cfg.groundtruth_bin_top100
    if top_k == 500 and hasattr(cfg, "groundtruth_bin_top500"):
        path = cfg.groundtruth_bin_top500
        if os.path.exists(path):
            return path
    # Fallback: generated path
    return str(PAG_DIR / cfg.name / f"{cfg.name}_truth{top_k}.bin")


def ensure_groundtruth(cfg: DatasetConfig, top_k: int,
                       database: np.ndarray, queries: np.ndarray,
                       skip_gen: bool = False) -> np.ndarray | None:
    """Load or generate groundtruth for the given top_k.

    Returns the ground truth array (n_queries, top_k) or None if unavailable.
    """
    # For top_k <= 100, the existing top-100 file suffices (sliced).
    if top_k <= 100:
        if not os.path.exists(cfg.groundtruth_bin_top100) and not os.path.exists(cfg.groundtruth_txt_top100):
            print(f"    [SKIP] No top-100 groundtruth for {cfg.name}")
            return None
        gt_full = load_dataset_groundtruth(cfg)
        return gt_full[:, :top_k]

    # For top_k > 100 (e.g. 500): try dedicated files first.
    gt_path = _get_gt_path(cfg, top_k)
    if os.path.exists(gt_path):
        data = np.fromfile(gt_path, dtype=np.int32)
        total_per_row = data.size // cfg.query_size
        if total_per_row >= top_k:
            return data.reshape(cfg.query_size, total_per_row)[:, :top_k]

    # Check a generated path explicitly.
    gen_path = str(PAG_DIR / cfg.name / f"{cfg.name}_truth{top_k}.bin")
    if os.path.exists(gen_path):
        data = np.fromfile(gen_path, dtype=np.int32)
        total_per_row = data.size // cfg.query_size
        if total_per_row >= top_k:
            return data.reshape(cfg.query_size, total_per_row)[:, :top_k]

    if skip_gen:
        print(f"    [SKIP] No top-{top_k} groundtruth for {cfg.name} and --skip-gt-gen set")
        return None

    # Generate brute-force groundtruth.
    print(f"    Generating brute-force top-{top_k} groundtruth for {cfg.name}...")
    metric = DATASET_METRIC[cfg.name]
    t0 = time.time()
    gt = _generate_groundtruth_brute(database, queries, top_k, metric=metric)
    elapsed = time.time() - t0
    os.makedirs(os.path.dirname(gen_path), exist_ok=True)
    gt.astype(np.int32).tofile(gen_path)
    print(f"    Saved {gen_path} ({elapsed:.1f}s)")

    # Also copy to PAG directory if needed.
    pag_truth_dir = PAG_DIR / cfg.name
    if pag_truth_dir.exists():
        pag_truth = pag_truth_dir / f"{cfg.name}_truth{top_k}.bin"
        if not pag_truth.exists():
            shutil.copy2(gen_path, str(pag_truth))
            print(f"    Copied to {pag_truth}")

    return gt


# ---------------------------------------------------------------------------
# PAG script generation for different top-k
# ---------------------------------------------------------------------------

# Parameters extracted from existing PAG scripts.
PAG_PARAMS = {
    "music100":    {"n": 1000000,  "d": 100,  "qn": 10000, "efc": 100,  "M": 32, "L": 32},
    "glove100":    {"n": 1183514,  "d": 100,  "qn": 10000, "efc": 1000, "M": 64, "L": 32},
    "glove200":    {"n": 1183514,  "d": 200,  "qn": 10000, "efc": 1000, "M": 64, "L": 32},
    "dinov2":      {"n": 1281167,  "d": 768,  "qn": 50000, "efc": 1000, "M": 64, "L": 16},
    "book_corpus": {"n": 9250529,  "d": 1024, "qn": 10000, "efc": 1000, "M": 64, "L": 16},
    "gist1m":      {"n": 1000000,  "d": 960,  "qn": 1000,  "efc": 500,  "M": 32, "L": 16},
    "ir101":       {"n": 17091649, "d": 512,  "qn": 20000, "efc": 1000, "M": 64, "L": 32},
    "openai1536":  {"n": 1000000,  "d": 1536, "qn": 1000,  "efc": 500,  "M": 32, "L": 16},
    "word2vec":    {"n": 1000000,  "d": 300,  "qn": 1000,  "efc": 1000, "M": 64, "L": 32},
    "msong":       {"n": 994185,   "d": 420,  "qn": 1000,  "efc": 1000, "M": 64, "L": 32},
    "text2img10m": {"n": 10000000, "d": 200,  "qn": 100000,"efc": 1000, "M": 64, "L": 32},
}

# Wider efc for top-500.
PAG_EFC_TOP500 = {
    "music100": 500,
    "glove100": 1000,
    "glove200": 1000,
    "dinov2": 1000,
    "book_corpus": 1000,
}


def _pag_script_name(dataset: str, top_k: int, script_dir: Path = PAG_DIR) -> str:
    if top_k == 100:
        explicit_top100 = f"run_{dataset}_top100.sh"
        if (script_dir / explicit_top100).exists():
            return explicit_top100
        return f"run_{dataset}.sh"
    return f"run_{dataset}_top{top_k}.sh"


def ensure_pag_script(dataset: str, top_k: int) -> str:
    """Create a PAG run script for the given (dataset, top_k) if it doesn't exist."""
    script_name = _pag_script_name(dataset, top_k, PAG_DIR)
    script_path = PAG_DIR / script_name
    if script_path.exists():
        return script_name

    params = PAG_PARAMS[dataset]
    efc = params["efc"]
    if top_k >= 500:
        efc = PAG_EFC_TOP500.get(dataset, efc)

    # Truth file name: standard = {name}_truth.bin, top-K variants = {name}_truth{K}.bin
    if top_k == 100:
        truth_file = f"${{name}}/${{name}}_truth.bin"
    else:
        truth_file = f"${{name}}/${{name}}_truth{top_k}.bin"

    index_suffix = "" if top_k == 100 else f"_top{top_k}"
    index_dir = f"${{name}}/index{index_suffix}/"

    content = f"""# ------------------------------------------------------------------------------
#  Parameters  (top-{top_k} variant, auto-generated)
# ------------------------------------------------------------------------------
name={dataset}
n={params['n']}  #data size
d={params['d']}     #dimension
qn={params['qn']}    #query size
k={top_k}      #topk

efc={efc}   #HNSW parameter
M={params['M']}       #HNSW parameter
L={params['L']}       #level

dPath=./${{name}}/${{name}}_base.bin   #data path
qPath=./${{name}}/${{name}}_query.bin  #query path
tPath=./{truth_file}        #groundtruth path
iPath=./{index_dir}             #index path

#----Indexing for the first execution and searching for the following executions---------

./build/PEOs ${{dPath}} ${{qPath}} ${{tPath}} ${{iPath}} ${{n}} ${{qn}} ${{d}} ${{k}} ${{efc}} ${{M}} ${{L}}
"""
    script_path.write_text(content)
    os.chmod(str(script_path), 0o755)
    print(f"    Created PAG script: {script_path}")
    return script_name


# ---------------------------------------------------------------------------
# Also copy groundtruth into PAG directory if needed
# ---------------------------------------------------------------------------

def _ensure_pag_truth(cfg: DatasetConfig, top_k: int, gt: np.ndarray) -> None:
    """Ensure PAG dataset directory has the truth file for this top_k.

    PAG binary always reads ``maxk=100`` columns from the truth file,
    so we pad with -1 if gt has fewer than 100 columns.
    """
    pag_data_dir = PAG_DIR / cfg.name
    if not pag_data_dir.exists():
        return

    if top_k == 100:
        truth_name = f"{cfg.name}_truth.bin"
    else:
        truth_name = f"{cfg.name}_truth{top_k}.bin"

    truth_path = pag_data_dir / truth_name
    if truth_path.exists():
        # Verify it has at least 100 columns per row for PAG.
        file_size = truth_path.stat().st_size
        expected_100 = gt.shape[0] * 100 * 4
        if file_size >= expected_100:
            return
        # Fix: file is too small (was written with fewer columns).
        print(f"    Fixing PAG truth file (need 100 cols): {truth_path}")

    # PAG reads maxk=100 columns; pad with the full top-100 gt if available.
    gt_out = gt
    if gt.shape[1] < 100:
        gt_100_path = pag_data_dir / f"{cfg.name}_truth.bin"
        if gt_100_path.exists():
            gt_full = np.fromfile(str(gt_100_path), dtype=np.int32)
            nq = gt.shape[0]
            cols = gt_full.size // nq
            if cols >= 100:
                gt_out = gt_full.reshape(nq, cols)[:, :100]
        if gt_out.shape[1] < 100:
            # Pad with -1 as a last resort.
            pad = np.full((gt.shape[0], 100 - gt.shape[1]), -1, dtype=np.int32)
            gt_out = np.hstack([gt.astype(np.int32), pad])

    gt_out.astype(np.int32).tofile(str(truth_path))
    print(f"    Wrote PAG truth: {truth_path} ({gt_out.shape})")


# ---------------------------------------------------------------------------
# Parameter adjustments per top_k
# ---------------------------------------------------------------------------

def _adjust_params_for_topk(cfg: DatasetConfig, top_k: int) -> None:
    """Widen parameter sweeps when top_k > 100 so search budgets cover demand."""
    cfg.top_k = top_k

    if top_k <= 10:
        # Use standard ranges; they already start above 10.
        return

    if top_k == 100:
        # Default ranges are fine.
        return

    if top_k >= 500:
        # Widen all sweep ranges.
        cfg.mag_efs = [max(v, 500) for v in cfg.mag_efs]
        if max(cfg.mag_efs) < 2000:
            cfg.mag_efs = [500, 600, 800, 1000, 1500, 2000, 3000]

        cfg.ipnsw_ef_values = [max(v, 500) for v in cfg.ipnsw_ef_values]
        if max(cfg.ipnsw_ef_values) < 2000:
            cfg.ipnsw_ef_values = [500, 600, 800, 1000, 1500, 2000, 3000]

        cfg.mobius_budget_values = [max(v, 500) for v in cfg.mobius_budget_values]
        if max(cfg.mobius_budget_values) < 2000:
            cfg.mobius_budget_values = [500, 600, 800, 1000, 1500, 2000, 3000]

        cfg.scann_reorder_values = [max(v, 1000) for v in cfg.scann_reorder_values]
        if max(cfg.scann_reorder_values) < 5000:
            cfg.scann_reorder_values = [1000, 1500, 2000, 3000, 4000, 5000, 8000, 10000]

        cfg.scann_leaves_values = [max(v, 100) for v in cfg.scann_leaves_values]


# ---------------------------------------------------------------------------
# Algorithm runner (mirrors benchmark.py's run_algorithm)
# ---------------------------------------------------------------------------

def run_algorithm(name: str, cfg: DatasetConfig, database, queries, ground_truth):
    """Run a single algorithm and return (points, build_time, build_peak_mb, query_peak_mb)."""
    mod = importlib.import_module(RUNNER_MODULES[name])
    if name in {"mag", "ipnsw", "mobius"}:
        return mod.run(cfg, ground_truth)
    if name in {"pag_new", "pag_without_projection"}:
        return mod.run(cfg)
    return mod.run(cfg, database, queries, ground_truth)


# ---------------------------------------------------------------------------
# Statistics logging
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-friendly string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:.0f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


def _append_stats(record: dict) -> None:
    """Append a human-readable block to statistics.log, with raw JSON at the end."""
    STATS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_LOG, "a", encoding="utf-8") as f:
        # --- Summary record (end-of-run) ---
        if record.get("type") == "summary":
            f.write("\n")
            f.write("┌─────────────────────────────────────────────────────────────────────┐\n")
            f.write("│                        FINAL SUMMARY                                │\n")
            f.write("├───────────────┬──────────────┬──────────────┬────────────────────────┤\n")
            f.write("│ Dataset       │  Total Time  │  Build Time  │  Per-Algorithm Build   │\n")
            f.write("├───────────────┼──────────────┼──────────────┼────────────────────────┤\n")
            for ds, info in record.get("datasets", {}).items():
                total = _fmt_time(info["total_time_s"])
                build = _fmt_time(info["total_build_time_s"])
                bt_parts = ", ".join(f"{a}={_fmt_time(t)}" for a, t in info.get("build_times", {}).items())
                f.write(f"│ {ds:<13} │ {total:>12} │ {build:>12} │ {bt_parts:<22} │\n")
            f.write("├───────────────┴──────────────┴──────────────┴────────────────────────┤\n")
            grand = _fmt_time(record.get("grand_total_time_s", 0))
            f.write(f"│ GRAND TOTAL: {grand:<55} │\n")
            f.write("└──────────────────────────────────────────────────────────────────────┘\n")
            f.write(f"\n# [RAW JSON] {json.dumps(record, ensure_ascii=False)}\n")
            return

        # --- Per-(dataset, top_k) combo record ---
        ds = record.get("dataset", "?")
        tk = record.get("top_k", "?")
        ts = record.get("timestamp", "")
        elapsed = record.get("elapsed_s", 0)

        f.write(f"\n┌── {ds}  top-{tk}  @  {ts}  ({_fmt_time(elapsed)}) ")
        f.write("─" * max(0, 55 - len(ds) - len(str(tk)) - len(ts)) + "┐\n")
        f.write("│\n")
        f.write(f"│  {'Algorithm':<10} {'Build Time':>11} {'Query Time':>11} {'Build Peak':>11} {'Query Peak':>11}  │  Recall-QPS Sweep\n")
        f.write(f"│  {'─'*10} {'─'*11} {'─'*11} {'─'*11} {'─'*11}  │  {'─'*40}\n")

        for algo, info in record.get("algorithms", {}).items():
            if info.get("status") == "failed":
                err = info.get("error", "unknown")
                f.write(f"│  {algo:<10} {'FAILED':>11} {'':>11} {'':>11} {'':>11}  │  {err}\n")
                continue

            bt = info.get("build_time_s", 0)
            qt = info.get("query_time_s", 0)
            bm = info.get("build_peak_mb")
            qm = info.get("query_peak_mb", 0)

            bt_str = _fmt_time(bt) if bt > 0 else "(cached)"
            qt_str = _fmt_time(qt) if qt > 0 else "-"
            bm_str = f"{bm:.0f} MB" if bm and bm > 0 else "-"
            qm_str = f"{qm:.0f} MB"

            pts = info.get("points", [])
            # Show up to 6 representative points (first, last, and evenly spaced).
            if len(pts) <= 6:
                shown = pts
            else:
                indices = [0, len(pts) // 5, 2 * len(pts) // 5, 3 * len(pts) // 5, 4 * len(pts) // 5, len(pts) - 1]
                shown = [pts[i] for i in dict.fromkeys(indices)]

            sweep_str = "  ".join(f"R={p['recall']:.4f}@{p['budget']}" for p in shown)
            if len(pts) > 6:
                sweep_str += f"  ({len(pts)} pts)"

            f.write(f"│  {algo:<10} {bt_str:>11} {qt_str:>11} {bm_str:>11} {qm_str:>11}  │  {sweep_str}\n")

        f.write("│\n")
        f.write(f"└{'─' * 70}┘\n")
        # Also keep raw JSON (prefixed so parsers can still extract it).
        f.write(f"# [RAW JSON] {json.dumps(record, ensure_ascii=False)}\n")


def _log_separator(msg: str) -> None:
    with open(STATS_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'═' * 72}\n  {msg}\n{'═' * 72}\n")


# ---------------------------------------------------------------------------
# Optional index cleanup
# ---------------------------------------------------------------------------

def _path_is_allowed(path: Path, allowed_roots: list[Path]) -> bool:
    """Guardrail for cleanup: only delete artifacts inside known index roots."""
    resolved = path.resolve(strict=False)
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            continue
    return False


def _remove_index_path(path: Path, allowed_roots: list[Path]) -> bool:
    """Remove a file/symlink/directory if it exists and is under an allowed root."""
    if not path.exists() and not path.is_symlink():
        return False
    if not _path_is_allowed(path, allowed_roots):
        print(f"    [clean-indexes] skip unsafe path: {path}")
        return False

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()
    print(f"    [clean-indexes] removed: {path}")
    return True


def _read_pag_index_dir(script_dir: Path, script_name: str,
                        dataset: str, top_k: int,
                        default_dir: Path) -> Path:
    """Read iPath from a PAG-family shell script, falling back to default_dir."""
    script_path = script_dir / script_name
    if not script_path.exists():
        return default_dir

    try:
        for line in script_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line.startswith("iPath="):
                continue
            ipath_val = line.split("=", 1)[1].strip()
            if "#" in ipath_val:
                ipath_val = ipath_val[:ipath_val.index("#")].strip()
            ipath_val = ipath_val.strip('"').strip("'")
            ipath_val = ipath_val.replace("${name}", dataset)
            ipath_val = ipath_val.replace("$name", dataset)
            ipath = Path(ipath_val)
            return ipath if ipath.is_absolute() else script_dir / ipath
    except Exception as exc:
        print(f"    [clean-indexes] warn: could not parse {script_path}: {exc}")

    return default_dir


def clean_indexes_for_dataset(dataset: str, algorithms: list[str], top_ks: list[int]) -> None:
    """Delete persisted index artifacts after a dataset finishes."""
    print(f"\n  Cleaning indexes for {dataset}...")
    removed = 0
    cfg = DatasetConfig(name=dataset)

    allowed_roots = [
        ROOT / "MAG",
        ROOT / "ip-nsw",
        ROOT / "mobius",
        Path(cfg.pag_dir),
        Path(cfg.pag_new_dir),
        Path(cfg.pag_without_proj_dir),
    ]

    if "mag" in algorithms:
        # MAG build time includes KNNG construction, so clean both artifacts.
        for path in [Path(cfg.mag_index), Path(cfg.mag_knng)]:
            removed += int(_remove_index_path(path, allowed_roots))

    if "ipnsw" in algorithms:
        removed += int(_remove_index_path(Path(cfg.ipnsw_graph), allowed_roots))

    if "mobius" in algorithms:
        for path in [
            Path(cfg.mobius_graph),
            Path(cfg.mobius_data),
            Path(cfg.mobius_default_graph),
            Path(cfg.mobius_default_data),
        ]:
            removed += int(_remove_index_path(path, allowed_roots))

    pag_variants = [
        ("pag_new", cfg.pag_new_dir, f"index_new_{cfg.name}", None),
        (
            "pag_without_projection",
            cfg.pag_without_proj_dir,
            f"index_noproj_{cfg.name}",
            None,
        ),
    ]
    for algo, dir_value, default_name, default_subdir in pag_variants:
        if algo not in algorithms:
            continue
        script_dir = Path(dir_value)
        for top_k in top_ks:
            script_name = _pag_script_name(dataset, top_k, script_dir)
            if default_subdir:
                default_dir = script_dir / default_name / (default_subdir if top_k == 100 else f"{default_subdir}_top{top_k}")
            else:
                default_dir = script_dir / default_name
            index_dir = _read_pag_index_dir(script_dir, script_name, dataset, top_k, default_dir)
            removed += int(_remove_index_path(index_dir, allowed_roots))

    if removed == 0:
        print(f"    [clean-indexes] no index artifacts found for {dataset}")
    else:
        print(f"    [clean-indexes] removed {removed} artifact(s) for {dataset}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run comprehensive benchmarks")
    p.add_argument("--datasets", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS)
    p.add_argument("--algorithms", nargs="+", default=ALL_ALGORITHMS, choices=ALL_ALGORITHMS)
    p.add_argument("--top-ks", nargs="+", type=int, default=ALL_TOP_KS)
    p.add_argument("--skip-gt-gen", action="store_true",
                   help="Skip top-K > 100 benchmarks for datasets without precomputed groundtruth")
    p.add_argument("--scann-mode", default="reorder", choices=["reorder", "leaves"])
    p.add_argument("--clean-indexes", action="store_true",
                   help=("Delete persisted index artifacts after each dataset finishes. "
                         "This removes MAG .mag/.knng, ip-NSW graph, Mobius graph/data, "
                         "and PAG/PIF-PAG/PAG-only iPath directories; ScaNN is in-memory."))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    datasets = args.datasets
    algorithms = args.algorithms
    top_ks = sorted(args.top_ks)

    run_timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    print("=" * 72)
    print("COMPREHENSIVE BENCHMARK")
    print("=" * 72)
    print(f"Timestamp  : {run_timestamp}")
    print(f"Datasets   : {datasets}")
    print(f"Algorithms : {algorithms}")
    print(f"Top-Ks     : {top_ks}")
    print(f"Clean idx  : {args.clean_indexes}")
    print(f"Stats log  : {STATS_LOG}")
    print("=" * 72)

    _log_separator(f"BENCHMARK RUN — {run_timestamp}")

    # ------------------------------------------------------------------
    # Collect per-dataset and per-(dataset, top_k, algo) statistics.
    # ------------------------------------------------------------------
    # dataset -> {algo -> build_time} (index build is shared across top_k)
    dataset_build_times: dict[str, dict[str, float]] = {}
    dataset_total_times: dict[str, float] = {}
    # dataset -> {algo -> {topk_label -> peak_mb}}  for memory bar charts
    dataset_query_mem: dict[str, dict[str, dict[str, float]]] = {}
    # dataset -> {topk -> result_txt_path}  for recall-QPS plots
    dataset_result_files: dict[str, dict[int, str]] = {}
    # All results for the summary.
    all_stats: list[dict] = []

    for ds in datasets:
        ds_start = time.time()
        print(f"\n{'#' * 72}")
        print(f"# DATASET: {ds}")
        print(f"{'#' * 72}")

        allowed = DATASET_ALLOWED_ALGORITHMS.get(ds, set(ALL_ALGORITHMS))
        ds_algorithms = [a for a in algorithms if a in allowed]
        if ds_algorithms != algorithms:
            skipped = [a for a in algorithms if a not in allowed]
            print(f"  [INFO] dataset-specific algorithm filter: running {ds_algorithms}, skip {skipped}")
        if not ds_algorithms:
            print(f"  [SKIP] No runnable algorithms selected for dataset {ds}")
            dataset_build_times[ds] = {}
            dataset_query_mem[ds] = {}
            dataset_result_files[ds] = {}
            dataset_total_times[ds] = 0.0
            continue

        # Load data once per dataset.
        cfg_base = DatasetConfig(name=ds)
        cfg_base.scann_mode = args.scann_mode
        missing_data_files = [
            path for path in [cfg_base.database_bin, cfg_base.query_bin]
            if not os.path.exists(path)
        ]
        if missing_data_files:
            print(f"\n  [SKIP] Missing data files for {ds}:")
            for path in missing_data_files:
                print(f"    - {path}")
            dataset_build_times[ds] = {}
            dataset_query_mem[ds] = {}
            dataset_result_files[ds] = {}
            dataset_total_times[ds] = 0.0
            continue

        print(f"\n  Loading data for {ds}...")
        database = read_bin(cfg_base.database_bin, cfg_base.dim)
        queries = read_bin(cfg_base.query_bin, cfg_base.dim)
        cfg_base.db_size = int(database.shape[0])
        cfg_base.query_size = int(queries.shape[0])
        print(f"  Database: {database.shape}, Queries: {queries.shape}")

        dataset_build_times[ds] = {}
        dataset_query_mem[ds] = {}
        dataset_result_files[ds] = {}

        for top_k in top_ks:
            supported_topks = DATASET_SUPPORTED_TOP_KS.get(ds)
            if supported_topks is not None and top_k not in supported_topks:
                print(f"\n  [SKIP] {ds} does not support top-{top_k} in this pipeline")
                continue

            combo_start = time.time()
            print(f"\n  {'─' * 60}")
            print(f"  Top-K = {top_k}  |  Dataset = {ds}")
            print(f"  {'─' * 60}")

            # ---- Groundtruth ------------------------------------------------
            gt = ensure_groundtruth(cfg_base, top_k, database, queries,
                                    skip_gen=args.skip_gt_gen)
            if gt is None:
                print(f"  [SKIP] No groundtruth for {ds} top-{top_k}")
                continue

            # ---- Config for this (dataset, top_k) ---------------------------
            cfg = DatasetConfig(name=ds)
            cfg.scann_mode = args.scann_mode
            cfg.db_size = cfg_base.db_size
            cfg.query_size = cfg_base.query_size
            _adjust_params_for_topk(cfg, top_k)

            # ---- PAG script --------------------------------------------------
            if "pag" in ds_algorithms:
                pag_script_name = ensure_pag_script(ds, top_k)
                cfg.pag_run_script = pag_script_name
                _ensure_pag_truth(cfg, top_k, gt)

            # pag_new uses pre-created scripts with its own index directories
            if "pag_new" in ds_algorithms:
                cfg.pag_new_run_script = _pag_script_name(ds, top_k, Path(cfg.pag_new_dir))

            # pag_without_projection uses pre-created scripts with its own index directories
            if "pag_without_projection" in ds_algorithms:
                cfg.pag_without_proj_run_script = _pag_script_name(ds, top_k, Path(cfg.pag_without_proj_dir))

            # ---- Result file -------------------------------------------------
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            result_txt = str(RESULTS_DIR / f"{ds}_top{top_k}_{ts}.txt")

            # ---- Run algorithms ----------------------------------------------
            all_points: list[dict] = []
            combo_record: dict = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "dataset": ds,
                "top_k": top_k,
                "algorithms": {},
            }

            for algo in ds_algorithms:
                print(f"\n    --- {algo.upper()} (top-{top_k}) ---")
                try:
                    query_start = time.time()
                    points, build_time, build_peak_mb, query_peak_mb = run_algorithm(
                        algo, cfg, database, queries, gt,
                    )
                    query_elapsed = time.time() - query_start - build_time
                    all_points.extend(flatten_points(algo, points))

                    # Only record build_time for the first topk that actually builds.
                    is_first_build = (build_time > 0 and algo not in dataset_build_times[ds])
                    recorded_build_time = build_time if is_first_build else 0.0
                    recorded_build_peak = build_peak_mb if is_first_build else 0.0

                    # Record.
                    combo_record["algorithms"][algo] = {
                        "build_time_s": round(recorded_build_time, 3),
                        "build_peak_mb": round(recorded_build_peak, 1) if recorded_build_peak > 0 else None,
                        "query_time_s": round(query_elapsed, 3),
                        "query_peak_mb": round(query_peak_mb, 1),
                        "num_points": len(points),
                        "points": [
                            {"budget": p["budget"],
                             "recall": round(float(p["recall"]), 6),
                             "qps": round(float(p["qps"]), 2)}
                            for p in points
                        ],
                    }

                    # Track build time (only first real build counts per algo per dataset).
                    if is_first_build:
                        dataset_build_times[ds][algo] = build_time

                    # Track query peak memory per (dataset, algo, topk) for bar chart.
                    if algo not in dataset_query_mem[ds]:
                        dataset_query_mem[ds][algo] = {}
                    dataset_query_mem[ds][algo][f"top{top_k}"] = query_peak_mb

                    print(f"    [{algo.upper()}] {len(points)} points, "
                          f"build={recorded_build_time:.1f}s, "
                          f"query={query_elapsed:.1f}s, "
                          f"build_peak={build_peak_mb:.1f}MB, "
                          f"query_peak={query_peak_mb:.1f}MB")

                except Exception:
                    err = traceback.format_exc()
                    combo_record["algorithms"][algo] = {"status": "failed", "error": err.splitlines()[-1]}
                    print(f"    [WARNING] {algo.upper()} failed:\n{err}")

            combo_elapsed = time.time() - combo_start
            combo_record["elapsed_s"] = round(combo_elapsed, 2)

            # Save results and plot recall-QPS for this (dataset, topk).
            if all_points:
                metadata = {
                    "dataset": ds,
                    "db_size": cfg.db_size,
                    "dim": cfg.dim,
                    "query_size": cfg.query_size,
                    "top_k": top_k,
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                }
                save_results(all_points, result_txt, metadata=metadata)
                dataset_result_files[ds][top_k] = result_txt

                plot_path = str(THIS_DIR / "imgs" / f"{ds}_top{top_k}.png")
                try:
                    plot_results(result_txt, plot_path,
                                 f"{ds} Recall-QPS (top-{top_k})",
                                 dataset_name=ds, top_k=top_k)
                except Exception as e:
                    print(f"    [WARN] Plot failed: {e}")

            # Append to statistics.log.
            _append_stats(combo_record)
            all_stats.append(combo_record)
            print(f"\n  Combo ({ds}, top-{top_k}) done in {combo_elapsed:.1f}s")

        # ---- Per-dataset summary plots ----------------------------------------
        ds_elapsed = time.time() - ds_start
        dataset_total_times[ds] = ds_elapsed
        print(f"\n  Dataset {ds} total time: {ds_elapsed:.1f}s")

        # Memory bar chart: query peak memory per algorithm across topk values.
        if dataset_query_mem.get(ds):
            mem_png = str(THIS_DIR / "imgs" / f"{ds}_memory.png")
            try:
                plot_memory_bar_chart(
                    dataset_query_mem[ds], mem_png,
                    title=f"{ds} — Query Peak Memory by Algorithm",
                )
            except Exception as e:
                print(f"    [WARN] Memory bar chart failed: {e}")

        if args.clean_indexes:
            clean_indexes_for_dataset(ds, ds_algorithms, top_ks)

    # ======================================================================
    # Final summary
    # ======================================================================
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)

    summary: dict = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "type": "summary",
        "datasets": {},
    }

    total_time_all = sum(dataset_total_times.values())

    for ds in datasets:
        build_times = dataset_build_times.get(ds, {})
        total_build = sum(build_times.values())
        total_ds = dataset_total_times.get(ds, 0.0)

        print(f"\n  Dataset: {ds}")
        print(f"    Total time          : {total_ds:>10.1f}s")
        print(f"    Total index build   : {total_build:>10.1f}s")
        if build_times:
            for algo, bt in build_times.items():
                print(f"      {algo:<12}      : {bt:>10.1f}s")

        summary["datasets"][ds] = {
            "total_time_s": round(total_ds, 2),
            "total_build_time_s": round(total_build, 2),
            "build_times": {a: round(t, 2) for a, t in build_times.items()},
        }

    print(f"\n  GRAND TOTAL TIME: {total_time_all:.1f}s")
    summary["grand_total_time_s"] = round(total_time_all, 2)

    _log_separator("SUMMARY")
    _append_stats(summary)

    # Also print a compact table for each (dataset, top_k, algo).
    print("\n" + "─" * 100)
    print(f"{'Dataset':<14} {'Top-K':>6} {'Algorithm':<10} {'Build(s)':>10} {'Query(s)':>10} {'PeakMem(MB)':>12} {'#Points':>8} {'MaxRecall':>10}")
    print("─" * 100)
    for rec in all_stats:
        ds = rec["dataset"]
        tk = rec["top_k"]
        for algo, info in rec.get("algorithms", {}).items():
            if info.get("status") == "failed":
                print(f"{ds:<14} {tk:>6} {algo:<10} {'FAILED':>10} {'':>10} {'':>12} {'':>8} {'':>10}")
                continue
            bt = info.get("build_time_s", 0)
            qt = info.get("query_time_s", 0)
            bm = info.get("build_peak_mb") or 0
            np_ = info.get("num_points", 0)
            pts = info.get("points", [])
            max_recall = max((p["recall"] for p in pts), default=0)
            bm_str = f"{bm:.1f}" if bm > 0 else "-"
            bt_str = f"{bt:.1f}" if bt > 0 else "(cached)"
            qt_str = f"{qt:.1f}" if qt > 0 else "-"
            print(f"{ds:<14} {tk:>6} {algo:<10} {bt_str:>10} {qt_str:>10} {bm_str:>12} {np_:>8} {max_recall:>10.4f}")
    print("─" * 100)

    print(f"\nStatistics saved to: {STATS_LOG}")
    try:
        from baselinePlot import generate_plots

        outputs = generate_plots(ROOT / "baselinePlot", top_k=10)
        print("\nBaseline plots saved to:")
        print(f"  qpsrecall.png : {outputs['qpsrecall_png']}")
        print(f"  qpsrecall.pdf : {outputs['qpsrecall_pdf']}")
        print(f"  resource.png  : {outputs['resource_png']}")
        print(f"  resource.pdf  : {outputs['resource_pdf']}")
    except Exception as exc:
        print(f"\n[WARN] baselinePlot generation failed: {exc}")
    print("Done.")


if __name__ == "__main__":
    main()
