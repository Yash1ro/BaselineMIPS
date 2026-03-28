#!/usr/bin/env python3
"""Common utilities for benchmark modules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np


BENCHMARK_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCHMARK_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = BENCHMARK_DIR / "results"
IMGS_DIR = BENCHMARK_DIR / "imgs"

MAG_DIR = PROJECT_ROOT / "MAG"
IPNSW_DIR = PROJECT_ROOT / "ip-nsw"
MOBIUS_DIR = PROJECT_ROOT / "mobius"
PAG_DIR = PROJECT_ROOT / "PAG"


@dataclass
class DatasetConfig:
    """Dataset and algorithm parameters for benchmark runs."""

    name: str = "music100"
    dim: int = 100
    top_k: int = 100
    db_size: int = 1000000
    query_size: int = 10000

    database_bin: str = str(DATA_DIR / "database_music100.bin")
    query_bin: str = str(DATA_DIR / "query_music100.bin")
    database_txt: str = str(DATA_DIR / "database_music100.txt")
    query_txt: str = str(DATA_DIR / "query_music100.txt")

    groundtruth_txt_top100: str = str(DATA_DIR / "correct100_music100.txt")
    groundtruth_bin_top100: str = str(DATA_DIR / "correct100_music100.bin")

    mag_knng: str = str(MAG_DIR / "music100.knng")
    mag_index: str = str(MAG_DIR / "music100.mag")
    mag_result: str = str(MAG_DIR / "result_mag.txt")
    mag_build_knng_py: str = str(BENCHMARK_DIR / "tools" / "build_knng.py")
    mag_test_bin: str = str(MAG_DIR / "build" / "test" / "test_mag")
    mag_efs: list[int] = field(default_factory=lambda: [100, 200, 400, 600, 800, 1000])

    scann_result: str = str(RESULTS_DIR / "result_scann.txt")
    scann_distance: str = "dot_product"
    scann_mode: str = "reorder"
    scann_leaves_to_search: int = 100
    scann_reorder_values: list[int] = field(default_factory=lambda: [400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000])
    scann_leaves_values: list[int] = field(default_factory=lambda: [20, 50, 100, 200, 500, 1000])
    scann_num_leaves: int = 2000

    ipnsw_graph: str = str(IPNSW_DIR / "out_graph.hnsw")
    ipnsw_result: str = str(IPNSW_DIR / "result.txt")
    ipnsw_bin: str = str(IPNSW_DIR / "main")
    ipnsw_m: int = 32
    ipnsw_ef_construction: int = 500
    ipnsw_ef_values: list[int] = field(default_factory=lambda: [100, 200, 400, 600, 800, 1000, 1500, 2000])

    mobius_graph: str = str(MOBIUS_DIR / "bfsg_music100.graph")
    mobius_data: str = str(MOBIUS_DIR / "bfsg_music100.data")
    mobius_result: str = str(MOBIUS_DIR / "result.txt")
    mobius_dir: str = str(MOBIUS_DIR)
    mobius_bin: str = str(MOBIUS_DIR / "mobius")
    mobius_build_sh: str = str(MOBIUS_DIR / "build_graph.sh")
    mobius_default_graph: str = str(MOBIUS_DIR / "bfsg.graph")
    mobius_default_data: str = str(MOBIUS_DIR / "bfsg.data")
    mobius_budget_values: list[int] = field(default_factory=lambda: [50, 80, 100, 150, 200, 300, 500, 800, 1000])

    pag_dir: str = str(PAG_DIR)
    pag_hnsw_efc: int = 500
    pag_hnsw_M: int = 32
    pag_hnsw_L: int = 32

    faiss_result: str = str(RESULTS_DIR / "result_faiss_ivfpq.txt")
    faiss_nlist: int = 4000
    faiss_nprobe_values: list[int] = field(default_factory=lambda: [50, 100, 150, 200, 300, 500])
    faiss_m: int = 5
    faiss_nbits: int = 12

    def __post_init__(self) -> None:
        if self.name == "music100":
            return

        if self.name == "glove100":
            self.dim = 100
            self.top_k = 100
            self.db_size = 1183514
            self.query_size = 10000

            self.database_bin = str(DATA_DIR / "glove100_base.bin")
            self.query_bin = str(DATA_DIR / "glove100_query.bin")
            self.database_txt = str(DATA_DIR / "glove100_base.txt")
            self.query_txt = str(DATA_DIR / "glove100_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "glove100_truth.bin")
            self.groundtruth_bin_top100 = str(DATA_DIR / "glove100_truth.bin")

            self.mag_knng = str(MAG_DIR / "glove100.knng")
            self.mag_index = str(MAG_DIR / "glove100.mag")
            self.mag_result = str(MAG_DIR / "result_mag_glove100.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000, 1200, 1500, 2000]

            self.scann_result = str(RESULTS_DIR / "result_scann_glove100.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 2000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_glove100.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_glove100.txt")

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_glove100.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_glove100.data")
            self.mobius_result = str(MOBIUS_DIR / "result_glove100.txt")
            self.mobius_budget_values = [100, 150, 200, 300, 500, 1000, 1500, 2000, 3000]

            self.pag_hnsw_efc = 500
            self.pag_hnsw_M = 32
            self.pag_hnsw_L = 16

            self.faiss_result = str(RESULTS_DIR / "result_faiss_glove100.txt")
            return

        if self.name == "glove200":
            self.dim = 200
            self.top_k = 100
            self.db_size = 1183514
            self.query_size = 10000

            self.database_bin = str(DATA_DIR / "glove200_base.bin")
            self.query_bin = str(DATA_DIR / "glove200_query.bin")
            self.database_txt = str(DATA_DIR / "glove200_base.txt")
            self.query_txt = str(DATA_DIR / "glove200_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "glove200_truth.bin")
            self.groundtruth_bin_top100 = str(DATA_DIR / "glove200_truth.bin")

            self.mag_knng = str(MAG_DIR / "glove200.knng")
            self.mag_index = str(MAG_DIR / "glove200.mag")
            self.mag_result = str(MAG_DIR / "result_mag_glove200.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000, 1200, 1500, 2000]

            self.scann_result = str(RESULTS_DIR / "result_scann_glove200.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 2000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_glove200.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_glove200.txt")

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_glove200.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_glove200.data")
            self.mobius_result = str(MOBIUS_DIR / "result_glove200.txt")
            self.mobius_budget_values = [100, 150, 200, 300, 500, 1000, 1500, 2000, 3000]

            self.pag_hnsw_efc = 500
            self.pag_hnsw_M = 32
            self.pag_hnsw_L = 16

            self.faiss_result = str(RESULTS_DIR / "result_faiss_glove200.txt")
            return

        if self.name == "dinov2":
            self.dim = 768
            self.top_k = 100
            # Keep defaults aligned with common ImageNet-1K split;
            # actual counts are updated from loaded arrays in benchmark.py.
            self.db_size = 1281167
            self.query_size = 50000

            self.database_bin = str(DATA_DIR / "dinov2_base.bin")
            self.query_bin = str(DATA_DIR / "dinov2_query.bin")
            self.database_txt = str(DATA_DIR / "dinov2_base.txt")
            self.query_txt = str(DATA_DIR / "dinov2_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "dinov2_truth.txt")
            self.groundtruth_bin_top100 = str(DATA_DIR / "dinov2_truth.bin")

            self.mag_knng = str(MAG_DIR / "dinov2.knng")
            self.mag_index = str(MAG_DIR / "dinov2.mag")
            self.mag_result = str(MAG_DIR / "result_mag_dinov2.txt")
            self.mag_efs = [100, 200, 400, 600]

            # DINOv2 in baseline.py is configured with L2 ground truth.
            self.scann_result = str(RESULTS_DIR / "result_scann_dinov2.txt")
            self.scann_distance = "squared_l2"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 2000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [200, 300, 500, 1000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_dinov2.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_dinov2.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_dinov2.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_dinov2.data")
            self.mobius_result = str(MOBIUS_DIR / "result_dinov2.txt")
            self.mobius_budget_values = [50, 80, 100, 150, 200, 300, 500]

            self.pag_hnsw_efc = 500
            self.pag_hnsw_M = 32
            self.pag_hnsw_L = 16

            self.faiss_result = str(RESULTS_DIR / "result_faiss_dinov2.txt")
            return

        if self.name == "book_corpus":
            self.dim = 1024
            self.top_k = 100
            # actual counts are updated from loaded arrays in benchmark.py.
            self.db_size = 9250529
            self.query_size = 10000

            self.database_bin = str(DATA_DIR / "book_corpus_base.bin")
            self.query_bin = str(DATA_DIR / "book_corpus_query.bin")
            self.database_txt = str(DATA_DIR / "book_corpus_base.txt")
            self.query_txt = str(DATA_DIR / "book_corpus_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "book_corpus_truth.txt")
            self.groundtruth_bin_top100 = str(DATA_DIR / "book_corpus_truth.bin")

            self.mag_knng = str(MAG_DIR / "book_corpus.knng")
            self.mag_index = str(MAG_DIR / "book_corpus.mag")
            self.mag_result = str(MAG_DIR / "result_mag_book_corpus.txt")
            self.mag_efs = [100, 200, 400, 500, 1000, 1500, 2000]

            self.scann_result = str(RESULTS_DIR / "result_scann_book_corpus.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 4000
            self.scann_leaves_to_search = 200
            self.scann_reorder_values = [100, 200, 400, 800, 1000, 1500, 2000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_book_corpus.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_book_corpus.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 500, 1000, 1500, 3000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_book_corpus.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_book_corpus.data")
            self.mobius_result = str(MOBIUS_DIR / "result_book_corpus.txt")
            self.mobius_budget_values = [80, 100, 150, 200, 500, 1000 ,1500, 2000]

            self.pag_hnsw_efc = 2000
            self.pag_hnsw_M = 64
            self.pag_hnsw_L = 16

            self.faiss_result = str(RESULTS_DIR / "result_faiss_book_corpus.txt")
            return

        raise ValueError(f"Unsupported dataset: {self.name}")


THREAD_LIMIT_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
]


def build_env_without_thread_limits() -> dict[str, str]:
    """Build environment for index construction jobs that should use default threading."""
    env = os.environ.copy()
    for key in THREAD_LIMIT_KEYS:
        env.pop(key, None)
    return env


def file_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def read_bin(file_path: str, dim: int) -> np.ndarray:
    """Read vectors from raw float32 or fvecs-like binary file."""
    file_size = os.path.getsize(file_path)
    raw_stride = dim * 4
    fvecs_stride = (dim + 1) * 4

    if file_size % raw_stride == 0:
        n = file_size // raw_stride
        data = np.fromfile(file_path, dtype=np.float32)
        return data.reshape(n, dim)

    if file_size % fvecs_stride == 0:
        n = file_size // fvecs_stride
        data = np.fromfile(file_path, dtype=np.int32).reshape(n, dim + 1)
        return data[:, 1:].view(np.float32)

    raise ValueError(f"Unsupported binary format for {file_path}")


def load_groundtruth_auto(file_path: str, n_queries: int, top_k: int) -> np.ndarray:
    """Load top-k ground truth from txt or binary int32 file."""
    if file_path.endswith(".txt"):
        rows: list[list[int]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                row = line.strip()
                if row:
                    rows.append(list(map(int, row.split()))[:top_k])
        return np.asarray(rows, dtype=np.int32)

    data = np.fromfile(file_path, dtype=np.int32)
    return data.reshape(n_queries, -1)[:, :top_k]


def load_dataset_groundtruth(config: DatasetConfig) -> np.ndarray:
    """Prefer binary ground truth when available for the selected dataset."""
    if os.path.exists(config.groundtruth_bin_top100):
        return load_groundtruth_auto(config.groundtruth_bin_top100, config.query_size, config.top_k)
    return load_groundtruth_auto(config.groundtruth_txt_top100, config.query_size, config.top_k)


def compute_recall(results: np.ndarray, ground_truth: np.ndarray, top_k: int) -> float:
    """Compute average Recall@K."""
    n_queries = results.shape[0]
    total_recall = 0.0
    for i in range(n_queries):
        intersection = np.intersect1d(results[i], ground_truth[i])
        total_recall += len(intersection) / top_k
    return total_recall / n_queries


def load_results(result_file: str, expected_k: int | None = None) -> np.ndarray:
    """Load result ids and pad short rows with -1 if needed."""
    rows: list[list[int]] = []
    with open(result_file, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(list(map(int, text.split())))

    if not rows:
        return np.empty((0, 0), dtype=np.int64)

    k = expected_k if expected_k is not None else max(len(row) for row in rows)
    padded = [row[:k] + [-1] * max(0, k - len(row)) for row in rows]
    return np.asarray(padded, dtype=np.int64)


def write_neighbors_txt(neighbors: np.ndarray, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in neighbors:
            f.write(" ".join(map(str, row)) + "\n")


def ensure_parent_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def flatten_points(algorithm: str, points: Iterable[dict]) -> list[dict]:
    """Normalize algorithm points to a flat list for result serialization."""
    flat: list[dict] = []
    for p in points:
        flat.append(
            {
                "algorithm": algorithm,
                "budget": p.get("budget"),
                "recall": float(p.get("recall", 0.0)),
                "qps": float(p.get("qps", 0.0)),
            }
        )
    return flat
