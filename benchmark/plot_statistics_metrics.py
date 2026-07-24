#!/usr/bin/env python3
"""Plot per-dataset metric bar charts from statistics.log.

For each dataset, generate one image with 4 bar-chart subplots:
1. index time (build_time_s)
2. index peak memory (build_peak_mb)
3. query time (query_time_s)
4. query peak memory (query_peak_mb)

Each subplot compares algorithms within the same dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STATS_LOG = ROOT / "statistics.log"
IMGS_DIR = ROOT / "benchmark" / "imgs"

DISPLAY_LABELS = {
    "mag": "MAG",
    "ipnsw": "ip-NSW",
    "mobius": "Möbius-Graph",
    "scann": "ScaNN",
    "pag_new": "PIF-PAG",
    "pag_without_projection": "PAG-only",
}


DATASET_DISPLAY_LABELS = {
    "music100": "Music",
    "glove100": "GloVe100",
    "glove200": "GloVe200",
    "gist1m": "GIST",
    "ir101": "Imagenet2M",
    "dinov2": "ImageNet-DINOv2",
    "text2img10m": "Text2Image",
    "imagenet": "Imagenet2M",
    "msong": "Msong",
    "word2vec": "Word2Vec",
    "book_corpus": "BookCorpus",
}


def _display_label(algorithm: str) -> str:
    return DISPLAY_LABELS.get(algorithm, algorithm)


def _display_dataset_label(dataset: str) -> str:
    return DATASET_DISPLAY_LABELS.get(dataset, dataset)

DEFAULT_DATASETS = [
    "music100",
    "glove100",
    "glove200",
    "dinov2",
    "book_corpus",
    "gist1m",
    "ir101",
    "openai1536",
]

METRICS = [
    ("build_time_s", "Index Time (s)", "{:.1f}"),
    ("build_peak_mb", "Index Peak Memory (MB)", "{:.0f}"),
    ("query_time_s", "Query Time (s)", "{:.1f}"),
    ("query_peak_mb", "Query Peak Memory (MB)", "{:.0f}"),
]


def _load_records(top_k: int | None) -> list[dict]:
    records: list[dict] = []
    if not STATS_LOG.exists():
        raise FileNotFoundError(f"statistics log not found: {STATS_LOG}")

    with STATS_LOG.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            raw = None
            if "# [RAW JSON]" in line:
                raw = line.split("# [RAW JSON]", 1)[1].strip()
            elif line.startswith("{") and '"dataset"' in line and '"algorithms"' in line:
                # Backward-compatible fallback: some lines in old logs may be plain JSON.
                raw = line

            if not raw:
                continue

            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if rec.get("type") == "summary":
                continue
            if top_k is not None and rec.get("top_k") != top_k:
                continue

            if not rec.get("dataset") or not isinstance(rec.get("algorithms"), dict):
                continue

            records.append(rec)

    return records


def _build_latest_metric_maps(records: list[dict]) -> dict[str, dict[tuple[str, str], tuple[str, float]]]:
    """Return metric -> {(dataset, algo): (timestamp, value)} using latest valid value."""
    metric_maps: dict[str, dict[tuple[str, str], tuple[str, float]]] = {m[0]: {} for m in METRICS}

    for rec in records:
        ts = str(rec.get("timestamp", ""))
        ds = str(rec.get("dataset", ""))
        algos = rec.get("algorithms", {})

        for algo, info in algos.items():
            if not isinstance(info, dict):
                continue
            if info.get("status") == "failed":
                continue

            key = (ds, str(algo))
            for metric_key, _, _ in METRICS:
                val = info.get(metric_key)
                if val is None:
                    continue
                if not isinstance(val, (int, float)):
                    continue
                if math.isnan(float(val)):
                    continue
                if float(val) <= 0:
                    # 0 usually means cached/not measured; skip to avoid misleading bars.
                    continue

                prev = metric_maps[metric_key].get(key)
                if prev is None or ts >= prev[0]:
                    metric_maps[metric_key][key] = (ts, float(val))

    return metric_maps


def _collect_algorithms_for_dataset(
    dataset: str,
    metric_maps: dict[str, dict[tuple[str, str], tuple[str, float]]],
) -> list[str]:
    algos: set[str] = set()
    for metric_map in metric_maps.values():
        for ds, algo in metric_map.keys():
            if ds == dataset:
                algos.add(algo)
    return sorted(algos)


def _plot_dataset_bars(
    dataset: str,
    top_k: int,
    algos: list[str],
    metric_maps: dict[str, dict[tuple[str, str], tuple[str, float]]],
) -> Path | None:
    if not algos:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), squeeze=False)
    axes_flat = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]

    for ax, (metric_key, metric_title, value_fmt) in zip(axes_flat, METRICS):
        values = []
        has_data = []
        for algo in algos:
            hit = metric_maps[metric_key].get((dataset, algo))
            if hit is None:
                values.append(0.0)
                has_data.append(False)
            else:
                values.append(hit[1])
                has_data.append(True)

        x = np.arange(len(algos))
        colors = ["#4E79A7" if ok else "#D9D9D9" for ok in has_data]
        bars = ax.bar(x, values, color=colors, edgecolor="#333333", linewidth=0.6)

        ax.set_title(metric_title, fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([_display_label(a) for a in algos], rotation=30, ha="right", fontsize=10)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

        if not any(has_data):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center", fontsize=11, color="#666666")
            continue

        ymax = max(values) if values else 0.0
        label_offset = max(ymax * 0.01, 1e-9)
        for bar, val, ok in zip(bars, values, has_data):
            xmid = bar.get_x() + bar.get_width() / 2
            if ok:
                ax.text(xmid, val + label_offset, value_fmt.format(val), ha="center", va="bottom", fontsize=8, rotation=90)
            else:
                ax.text(xmid, label_offset, "N/A", ha="center", va="bottom", fontsize=8, color="#666666", rotation=90)

    fig.suptitle(f"{_display_dataset_label(dataset)} — Metrics by Algorithm (top-{top_k})", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = IMGS_DIR / f"statistics_{dataset}_bars_top{top_k}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot per-dataset bar chart images from statistics.log")
    p.add_argument("--top-k", type=int, default=10, help="Only use this top-k from statistics.log (default: 10)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = _load_records(args.top_k)
    if not records:
        raise RuntimeError(f"No usable records found in {STATS_LOG} for top_k={args.top_k}")

    metric_maps = _build_latest_metric_maps(records)
    generated = 0

    for ds in DEFAULT_DATASETS:
        algos = _collect_algorithms_for_dataset(ds, metric_maps)
        if not algos:
            print(f"[WARN] no metrics for dataset={ds}, skip")
            continue
        out = _plot_dataset_bars(ds, args.top_k, algos, metric_maps)
        if out is not None:
            generated += 1

    if generated == 0:
        raise RuntimeError(f"No per-dataset images generated for top_k={args.top_k}")


if __name__ == "__main__":
    main()
