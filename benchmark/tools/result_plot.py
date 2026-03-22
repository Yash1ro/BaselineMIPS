#!/usr/bin/env python3
"""Save benchmark results to txt and plot recall-QPS curves from txt."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_results(points: list[dict], output_txt: str = "result.txt") -> None:
    """Save flat benchmark points to tab-separated txt."""
    Path(output_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("algorithm\tbudget\trecall\tqps\n")
        for p in points:
            f.write(
                f"{p['algorithm']}\t{p['budget']}\t{float(p['recall']):.8f}\t{float(p['qps']):.6f}\n"
            )
    print(f"Saved {len(points)} rows to {output_txt}")


def load_results(input_txt: str) -> list[dict]:
    """Load tab-separated benchmark points from txt."""
    points: list[dict] = []
    with open(input_txt, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        if header != ["algorithm", "budget", "recall", "qps"]:
            raise ValueError(f"Unexpected header in {input_txt}: {header}")
        for line in f:
            text = line.strip()
            if not text:
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
