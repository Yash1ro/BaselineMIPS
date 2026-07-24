#!/usr/bin/env python3
"""Save benchmark results to txt and plot recall-QPS curves from txt."""

from __future__ import annotations

import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results(
    points: list[dict],
    output_txt: str = "result.txt",
    metadata: dict | None = None,
) -> None:
    """Save flat benchmark points to tab-separated txt with optional metadata header.

    Metadata is written as ``# key: value`` comment lines before the data header so
    that the file remains backward-compatible with readers that skip ``#`` lines.

    ``metadata`` may contain the following keys:
      - ``dataset``   : dataset name
      - ``db_size``   : number of database vectors
      - ``dim``       : vector dimension
      - ``query_size``: number of query vectors
      - ``timestamp`` : ISO-8601 timestamp string
      - ``params``    : dict mapping algorithm name → dict of param key/value pairs
    """
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        if metadata:
            for key in ("dataset", "db_size", "dim", "query_size", "timestamp"):
                if key in metadata:
                    f.write(f"# {key}: {metadata[key]}\n")
            if "params" in metadata:
                for algo, params in metadata["params"].items():
                    f.write(f"# --- params:{algo} ---\n")
                    for k, v in sorted(params.items()):
                        f.write(f"# {k}: {v}\n")
        f.write("algorithm\tbudget\trecall\tqps\n")
        for p in points:
            f.write(
                f"{p['algorithm']}\t{p['budget']}\t{float(p['recall']):.8f}\t{float(p['qps']):.6f}\n"
            )
    print(f"Saved {len(points)} rows to {output_txt}")


def load_results(input_txt: str) -> list[dict]:
    """Load tab-separated benchmark points from txt.

    Lines starting with ``#`` are treated as metadata comments and skipped,
    which makes this function compatible with both the legacy format (no
    comments) and the new format (metadata comment header).
    """
    points: list[dict] = []
    with open(input_txt, "r", encoding="utf-8") as f:
        header_found = False
        for line in f:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            if not header_found:
                header = text.split("\t")
                if header != ["algorithm", "budget", "recall", "qps"]:
                    raise ValueError(f"Unexpected header in {input_txt}: {header}")
                header_found = True
                continue
            algo, budget, recall, qps = text.split("\t")
            points.append(
                {
                    "algorithm": algo,
                    "budget": budget,
                    "recall": float(recall),
                    "qps": float(qps),
                }
            )
    return points


def _read_raw_file(input_txt: str) -> tuple[list[str], list[dict]]:
    """Read a result file preserving raw comment lines and parsed data points.

    Returns ``(comment_lines, points)`` where *comment_lines* are the raw ``#``
    header lines (including newlines) and *points* is the parsed data.
    """
    comment_lines: list[str] = []
    points: list[dict] = []
    with open(input_txt, "r", encoding="utf-8") as f:
        header_found = False
        for line in f:
            stripped = line.rstrip("\n").strip()
            if stripped.startswith("#"):
                comment_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            if not stripped:
                continue
            if not header_found:
                if stripped == "algorithm\tbudget\trecall\tqps":
                    header_found = True
                continue
            algo, budget, recall, qps = stripped.split("\t")
            points.append(
                {
                    "algorithm": algo,
                    "budget": budget,
                    "recall": float(recall),
                    "qps": float(qps),
                }
            )
    return comment_lines, points


def update_algorithm_section(
    new_points: list[dict],
    algorithm: str,
    output_txt: str,
) -> None:
    """Replace only the rows for *algorithm* in an existing result file.

    All ``#`` metadata comment lines are preserved.  The ``# timestamp:`` line
    is updated to the current time to reflect when the file was last modified.
    All other algorithms' rows are left untouched.
    """
    comment_lines, existing_points = _read_raw_file(output_txt)

    # Replace rows belonging to the target algorithm.
    kept = [p for p in existing_points if p["algorithm"] != algorithm]
    kept.extend(new_points)

    # Refresh the timestamp comment.
    ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    updated_comments: list[str] = []
    for line in comment_lines:
        if line.startswith("# timestamp:"):
            updated_comments.append(f"# timestamp: {ts}\n")
        else:
            updated_comments.append(line)

    with open(output_txt, "w", encoding="utf-8") as f:
        for line in updated_comments:
            f.write(line)
        f.write("algorithm\tbudget\trecall\tqps\n")
        for p in kept:
            f.write(
                f"{p['algorithm']}\t{p['budget']}\t{float(p['recall']):.8f}\t{float(p['qps']):.6f}\n"
            )
    print(f"Updated {len(new_points)} {algorithm} rows in {output_txt}")


def _pareto_frontier(values: list[dict]) -> list[dict]:
    """Keep only non-dominated recall-QPS points for line plots."""
    ordered = sorted(values, key=lambda x: (x["recall"], x["qps"]))
    frontier_rev: list[dict] = []
    best_qps = float("-inf")
    for point in reversed(ordered):
        if point["qps"] > best_qps:
            frontier_rev.append(point)
            best_qps = point["qps"]
    return list(reversed(frontier_rev))


def plot_results(
    input_txt: str,
    output_png: str | None = None,
    title: str = "Recall-QPS Benchmark",
    dataset_name: str | None = None,
    top_k: int | None = None,
    log_y: bool = False,
) -> None:
    """Plot grouped recall-QPS curves from txt file."""
    points = load_results(input_txt)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        grouped[p["algorithm"]].append(p)

    if not dataset_name:
        dataset_name = "dataset"
    if top_k is None:
        top_k = 100

    fig, ax = plt.subplots(figsize=(12, 7))
    for algorithm, values in grouped.items():
        # Keep all points in result.txt, but hide the first 4 ScaNN points in plots.
        if algorithm.lower() == "scann" and len(values) > 4:
            values = values[4:]
        values = _pareto_frontier(values)
        recalls = [v["recall"] * 100 for v in values]
        qps = [v["qps"] for v in values]
        style = STYLE_MAP.get(algorithm.lower(), {"marker": "o", "linestyle": "-", "color": None})
        ax.plot(
            recalls,
            qps,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.5,
            markersize=8,
            label=display_label(algorithm),
        )

    ax.set_xlabel(f"Recall@{top_k} (%)", fontsize=16)
    ax.set_ylabel("QPS", fontsize=16)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=14)
    fig.tight_layout()

    if not output_png:
        output_path = Path(input_txt).resolve().parent.parent / "imgs" / f"{dataset_name}_top{top_k}.png"
    else:
        user_path = Path(output_png)
        if user_path.suffix:
            output_path = user_path
        else:
            output_path = user_path / f"{dataset_name}_top{top_k}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


STYLE_MAP = {
    "mag":    {"marker": "o", "linestyle": "-",  "color": "#E74C3C"},
    "ipnsw":  {"marker": "s", "linestyle": "-",  "color": "#3498DB"},
    "mobius": {"marker": "^", "linestyle": "-",  "color": "#2ECC71"},
    "pag":    {"marker": "D", "linestyle": "-",  "color": "#9B59B6"},
    "scann":  {"marker": "P", "linestyle": "-",  "color": "#F39C12"},
    "pag_new":                {"marker": "v", "linestyle": "--", "color": "#FF1493"},
    "pag_without_projection": {"marker": "X", "linestyle": "--", "color": "#E67E22"},
}


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


def display_label(algorithm: str) -> str:
    """Paper-facing label for an internal algorithm key."""
    return DISPLAY_LABELS.get(algorithm, algorithm)


def display_dataset_label(dataset: str) -> str:
    """Paper-facing label for an internal dataset key."""
    return DATASET_DISPLAY_LABELS.get(dataset, dataset)


def plot_memory_bar_chart(
    mem_data: dict[str, dict[str, float]],
    output_png: str,
    title: str = "Peak Memory by Algorithm",
    ylabel: str = "Peak Memory (MB)",
) -> None:
    """Draw a grouped bar chart of peak memory for each algorithm.

    Parameters
    ----------
    mem_data : dict[algo_name, dict[topk_label, peak_mb]]
        e.g. {"mag": {"top10": 1021.7, "top100": 1200.0}, ...}
    output_png : output file path
    title : chart title
    """
    import numpy as _np

    algorithms = sorted(mem_data.keys())
    if not algorithms:
        return

    topk_labels = sorted({k for d in mem_data.values() for k in d}, key=lambda x: int(x.replace("top", "")))
    n_algo = len(algorithms)
    n_topk = len(topk_labels)

    x = _np.arange(n_algo)
    width = 0.8 / max(n_topk, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_algo * 2), 7))
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]

    for i, tk in enumerate(topk_labels):
        vals = [mem_data[algo].get(tk, 0) for algo in algorithms]
        offset = (i - n_topk / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width * 0.9, label=tk,
                       color=colors[i % len(colors)], edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Algorithm", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([display_label(algo) for algo in algorithms], fontsize=13)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved memory bar chart to {output_png}")


def plot_qps_recall_multi(
    dataset_results: dict[str, list[dict]],
    output_png: str,
    title: str = "QPS-Recall Comparison",
    top_k: int = 10,
) -> None:
    """Plot QPS-Recall line charts for multiple datasets on separate subplots.

    Parameters
    ----------
    dataset_results : dict[dataset_name, list[point_dicts]]
        Each point has keys: algorithm, budget, recall, qps
    output_png : output file path
    """
    n_ds = len(dataset_results)
    if n_ds == 0:
        return

    cols = min(n_ds, 3)
    rows = (n_ds + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows), squeeze=False)

    for idx, (ds_name, points) in enumerate(dataset_results.items()):
        ax = axes[idx // cols][idx % cols]
        grouped: dict[str, list[dict]] = defaultdict(list)
        for p in points:
            grouped[p["algorithm"]].append(p)

        for algo, vals in grouped.items():
            vals = _pareto_frontier(vals)
            recalls = [v["recall"] * 100 for v in vals]
            qps = [v["qps"] for v in vals]
            style = STYLE_MAP.get(algo.lower(), {"marker": "o", "linestyle": "-", "color": None})
            ax.plot(recalls, qps, marker=style["marker"], linestyle=style["linestyle"],
                    color=style["color"], linewidth=2, markersize=7, label=display_label(algo))

        ax.set_xlabel(f"Recall@{top_k} (%)", fontsize=13)
        ax.set_ylabel("QPS", fontsize=13)
        ax.set_title(display_dataset_label(ds_name), fontsize=14, fontweight="bold")
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=10)

    # Hide unused axes
    for idx in range(n_ds, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multi-dataset QPS-recall plot to {output_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results from txt file")
    parser.add_argument("--input", default="benchmark/results/result.txt", help="Input result txt")
    parser.add_argument("--output", default=None, help="Output png path or output directory")
    parser.add_argument("--title", default="Recall-QPS Benchmark", help="Plot title")
    parser.add_argument("--dataset", default="dataset", help="Dataset name for default output naming")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k value for default output naming")
    parser.add_argument("--log-y", action="store_true", help="Use logarithmic scale for the QPS axis")
    args = parser.parse_args()

    plot_results(args.input, args.output, args.title, dataset_name=args.dataset, top_k=args.top_k, log_y=args.log_y)


if __name__ == "__main__":
    main()
