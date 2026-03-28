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


def plot_results(
    input_txt: str,
    output_png: str | None = None,
    title: str = "Recall-QPS Benchmark",
    dataset_name: str | None = None,
    top_k: int | None = None,
) -> None:
    """Plot grouped recall-QPS curves from txt file."""
    points = load_results(input_txt)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for p in points:
        grouped[p["algorithm"]].append(p)

    style_map = {
        "mag": {"marker": "o", "linestyle": "-"},
        "ipnsw": {"marker": "s", "linestyle": "-"},
        "mobius": {"marker": "^", "linestyle": "-"},
        "pag": {"marker": "D", "linestyle": "-"},
        "scann": {"marker": "P", "linestyle": "-"},
    }

    plt.figure(figsize=(12, 7))
    for algorithm, values in grouped.items():
        # Keep all points in result.txt, but hide the first 4 ScaNN points in plots.
        if algorithm.lower() == "scann" and len(values) > 4:
            values = values[4:]
        values = sorted(values, key=lambda x: x["recall"])
        recalls = [v["recall"] * 100 for v in values]
        qps = [v["qps"] for v in values]
        style = style_map.get(algorithm.lower(), {"marker": "o", "linestyle": "-"})
        plt.plot(
            recalls,
            qps,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            markersize=5,
            label=algorithm,
        )

    plt.xlabel("Recall@K (%)")
    plt.ylabel("QPS")
    plt.title(title)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="best")
    plt.tight_layout()

    if not dataset_name:
        dataset_name = "dataset"
    if top_k is None:
        top_k = 100

    if not output_png:
        output_path = Path(input_txt).resolve().parent.parent / "imgs" / f"{dataset_name}_top{top_k}.png"
    else:
        user_path = Path(output_png)
        if user_path.suffix:
            output_path = user_path
        else:
            output_path = user_path / f"{dataset_name}_top{top_k}.png"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results from txt file")
    parser.add_argument("--input", default="benchmark/results/result.txt", help="Input result txt")
    parser.add_argument("--output", default=None, help="Output png path or output directory")
    parser.add_argument("--title", default="Recall-QPS Benchmark", help="Plot title")
    parser.add_argument("--dataset", default="dataset", help="Dataset name for default output naming")
    parser.add_argument("--top-k", type=int, default=100, help="Top-k value for default output naming")
    args = parser.parse_args()

    plot_results(args.input, args.output, args.title, dataset_name=args.dataset, top_k=args.top_k)


if __name__ == "__main__":
    main()
