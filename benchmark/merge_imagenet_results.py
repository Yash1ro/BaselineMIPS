#!/usr/bin/env python3
"""Merge imagenet benchmark results from multiple sources and generate a QPS-recall plot.

Usage
-----
# Merge PAG variants + any other algorithm files, then plot:
  python merge_imagenet_results.py

# Also include extra result files (e.g. from other machines):
  python merge_imagenet_results.py --extra /path/to/imagenet_mag_results.txt

# Only regenerate the plot without re-merging:
  python merge_imagenet_results.py --plot-only

Sources merged (in priority order — later entries overwrite earlier ones for the
same algorithm):
  1. benchmark/results/pag_variants_imagenet_top{k}_*.txt   (PAG variants)
  2. benchmark/results/imagenet_top{k}_*.txt                 (standard format, if any)
  3. Any --extra files passed on the command line

Output
------
  benchmark/results/imagenet_top{k}_merged.txt
  benchmark/imgs/imagenet_top10_merged.png
  benchmark/imgs/imagenet_top100_merged.png   (if top-100 data available)
"""

from __future__ import annotations

import argparse
import datetime
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BENCHMARK_DIR = Path(__file__).resolve().parent
ROOT = BENCHMARK_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "results"
IMGS_DIR    = BENCHMARK_DIR / "imgs"

sys.path.insert(0, str(BENCHMARK_DIR))
sys.path.insert(0, str(BENCHMARK_DIR / "tools"))

from result_plot import STYLE_MAP, _pareto_frontier, display_label


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_pag_variants_file(path: Path) -> tuple[int, dict[str, list[dict]]]:
    """Read a pag_variants_imagenet_top{k}_*.txt file.

    Returns (top_k, {variant: [{"budget", "recall", "qps"}, ...]}).
    """
    top_k = 10
    vdata: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("# top_k:"):
            try:
                top_k = int(line.split(":")[1].strip())
            except ValueError:
                pass
            continue
        if line.startswith("#") or line.startswith("algorithm") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        variant, budget, recall, qps = parts
        vdata.setdefault(variant, []).append({
            "budget": int(budget),
            "recall": float(recall),
            "qps":    float(qps),
        })
    return top_k, vdata


def _load_standard_file(path: Path) -> dict[str, list[dict]]:
    """Read a standard imagenet_top{k}_*.txt file (algorithm/budget/recall/qps)."""
    vdata: dict[str, list[dict]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or line.startswith("algorithm") or not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            continue
        algo, budget, recall, qps = parts
        vdata.setdefault(algo, []).append({
            "budget": int(budget),
            "recall": float(recall),
            "qps":    float(qps),
        })
    return vdata


def _write_merged(merged: dict[str, list[dict]], top_k: int) -> Path:
    """Write merged data to benchmark/results/imagenet_top{k}_merged.txt (overwrite)."""
    out = RESULTS_DIR / f"imagenet_top{top_k}_merged.txt"
    ts  = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# dataset: imagenet",
        f"# top_k: {top_k}",
        f"# timestamp: {ts}",
        "algorithm\tbudget\trecall\tqps",
    ]
    for algo in sorted(merged):
        for p in sorted(merged[algo], key=lambda x: x["budget"]):
            lines.append(f"{algo}\t{p['budget']}\t{p['recall']:.8f}\t{p['qps']:.6f}")
    out.write_text("\n".join(lines) + "\n")
    print(f"[write] {out}  ({sum(len(v) for v in merged.values())} points, "
          f"{len(merged)} algorithms: {sorted(merged)})")
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(merged: dict[str, list[dict]], top_k: int, out_png: Path) -> None:
    """Single-panel QPS-recall plot for all algorithms on imagenet."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for algo, points in sorted(merged.items()):
        if not points:
            continue
        pts = _pareto_frontier(points)
        recalls = [p["recall"] * 100 for p in pts]
        qps     = [p["qps"]    for p in pts]
        style   = STYLE_MAP.get(algo, {"marker": "o", "linestyle": "-", "color": None})
        ax.plot(recalls, qps,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2.5, markersize=8,
                label=display_label(algo))

    ax.set_xlabel(f"Recall@{top_k} (%)", fontsize=13)
    ax.set_ylabel("QPS",                 fontsize=13)
    ax.set_title(f"imagenet — Recall-QPS (top-{top_k})", fontsize=15, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(fontsize=11)
    ax.tick_params(axis="both", labelsize=11)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot]  {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge imagenet results and plot")
    p.add_argument("--top-k", nargs="+", type=int, default=[10, 100],
                   help="Top-K values to process (default: 10 100)")
    p.add_argument("--extra", nargs="*", default=[],
                   metavar="FILE",
                   help="Additional result files to merge in (standard tab-sep format)")
    p.add_argument("--plot-only", action="store_true",
                   help="Skip merging; just re-plot from existing imagenet_top{k}_merged.txt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    IMGS_DIR.mkdir(parents=True, exist_ok=True)

    for top_k in args.top_k:
        print(f"\n=== top-{top_k} ===")

        # ---- Load existing merged file (baseline) ----
        merged_path = RESULTS_DIR / f"imagenet_top{top_k}_merged.txt"
        merged: dict[str, list[dict]] = {}

        if merged_path.exists():
            merged = _load_standard_file(merged_path)
            print(f"[load]  existing merged file: {merged_path.name}  "
                  f"algorithms={sorted(merged)}")

        if not args.plot_only:
            # ---- 1. PAG variants files ----
            pag_files = sorted(
                RESULTS_DIR.glob(f"pag_variants_imagenet_top{top_k}_*.txt"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if pag_files:
                _, vdata = _load_pag_variants_file(pag_files[0])
                for variant, pts in vdata.items():
                    merged[variant] = pts
                print(f"[load]  PAG variants: {pag_files[0].name}  "
                      f"variants={sorted(vdata)}")
            else:
                print(f"[skip]  no pag_variants_imagenet_top{top_k}_*.txt found")

            # ---- 2. Standard imagenet result files (non-merged) ----
            std_files = sorted(
                [f for f in RESULTS_DIR.glob(f"imagenet_top{top_k}_*.txt")
                 if "merged" not in f.name],
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            for sf in std_files:
                sdata = _load_standard_file(sf)
                for algo, pts in sdata.items():
                    merged[algo] = pts
                print(f"[load]  standard file: {sf.name}  algorithms={sorted(sdata)}")

            # ---- 3. Extra files from command line ----
            for ef_str in (args.extra or []):
                ef = Path(ef_str)
                if not ef.exists():
                    print(f"[warn]  extra file not found: {ef}")
                    continue
                edata = _load_standard_file(ef)
                for algo, pts in edata.items():
                    merged[algo] = pts
                print(f"[load]  extra file: {ef.name}  algorithms={sorted(edata)}")

            if not merged:
                print(f"[skip]  no data found for top-{top_k}")
                continue

            # ---- Write merged file ----
            _write_merged(merged, top_k)

        # ---- Plot ----
        if not merged:
            if merged_path.exists():
                merged = _load_standard_file(merged_path)
            else:
                print(f"[skip]  no merged data for top-{top_k}, skipping plot")
                continue

        out_png = IMGS_DIR / f"imagenet_top{top_k}_merged.png"
        _plot(merged, top_k, out_png)

    print("\nDone.")


if __name__ == "__main__":
    main()
