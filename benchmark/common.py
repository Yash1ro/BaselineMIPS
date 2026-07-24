#!/usr/bin/env python3
"""Common utilities for benchmark modules."""

from __future__ import annotations

import os
import threading
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
PAG_NEW_DIR = PROJECT_ROOT / "PAG_new"
PAG_WITHOUT_PROJ_DIR = PROJECT_ROOT / "PAG_without_projection"


@dataclass
class DatasetConfig:
    """Dataset and algorithm parameters for benchmark runs."""

    name: str = "music100"
    dim: int = 100
    top_k: int = 100
    db_size: int = 1000000
    query_size: int = 10000

    database_bin: str = str(DATA_DIR / "music100" / "music100_base.bin")
    query_bin: str = str(DATA_DIR / "music100" / "music100_query.bin")
    database_txt: str = str(DATA_DIR / "music100" / "music100_base.txt")
    query_txt: str = str(DATA_DIR / "music100" / "music100_query.txt")

    groundtruth_txt_top100: str = str(DATA_DIR / "music100" / "music100_truth.bin")
    groundtruth_bin_top100: str = str(DATA_DIR / "music100" / "music100_truth.bin")
    groundtruth_bin_top500: str = str(DATA_DIR / "music100" / "music100_truth500.bin")

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
    pag_run_script: str = ""  # overridable; empty means auto-derive from name
    pag_hnsw_efc: int = 500
    pag_hnsw_M: int = 32
    pag_hnsw_L: int = 32

    pag_new_dir: str = str(PAG_NEW_DIR)
    pag_new_run_script: str = ""  # overridable; empty means auto-derive from name

    pag_without_proj_dir: str = str(PAG_WITHOUT_PROJ_DIR)
    pag_without_proj_run_script: str = ""  # overridable; empty means auto-derive from name

    faiss_result: str = str(RESULTS_DIR / "result_faiss_ivfpq.txt")
    faiss_nlist: int = 4000
    faiss_nprobe_values: list[int] = field(default_factory=lambda: [50, 100, 150, 200, 300, 500])
    faiss_m: int = 5
    faiss_nbits: int = 12

    def _prefer_dataset_bin_layout(self, prefix: str | None = None,
                                   truth_prefix: str | None = None) -> None:
        """Prefer data/<dataset> binaries, with compatibility for PAG/<dataset>."""
        prefix = prefix or self.name
        truth_prefix = truth_prefix or prefix
        data_dir = DATA_DIR / self.name
        pag_dir = PAG_DIR / self.name

        def choose(current: str, filename: str) -> str:
            data_path = data_dir / filename
            if data_path.exists():
                return str(data_path)
            if os.path.exists(current):
                return current
            pag_path = pag_dir / filename
            if pag_path.exists():
                return str(pag_path)
            return current

        self.database_bin = choose(self.database_bin, f"{prefix}_base.bin")
        self.query_bin = choose(self.query_bin, f"{prefix}_query.bin")
        self.groundtruth_txt_top100 = choose(self.groundtruth_txt_top100, f"{truth_prefix}_truth.bin")
        self.groundtruth_bin_top100 = choose(self.groundtruth_bin_top100, f"{truth_prefix}_truth.bin")
        self.groundtruth_bin_top500 = choose(self.groundtruth_bin_top500, f"{truth_prefix}_truth500.bin")

    def __post_init__(self) -> None:
        if self.name == "music100":
            # Prefer the shared data directory, but keep compatibility with
            # older layouts that stored Music-100 under PAG/music100.
            pag_music_dir = PAG_DIR / "music100"
            if not os.path.exists(self.database_bin) and (pag_music_dir / "music100_base.bin").exists():
                self.database_bin = str(pag_music_dir / "music100_base.bin")
            if not os.path.exists(self.query_bin) and (pag_music_dir / "music100_query.bin").exists():
                self.query_bin = str(pag_music_dir / "music100_query.bin")
            if not os.path.exists(self.groundtruth_bin_top100) and (pag_music_dir / "music100_truth.bin").exists():
                self.groundtruth_txt_top100 = str(pag_music_dir / "music100_truth.bin")
                self.groundtruth_bin_top100 = str(pag_music_dir / "music100_truth.bin")
            if not os.path.exists(self.groundtruth_bin_top500) and (pag_music_dir / "music100_truth500.bin").exists():
                self.groundtruth_bin_top500 = str(pag_music_dir / "music100_truth500.bin")
            self._prefer_dataset_bin_layout()
            return

        if self.name == "glove100":
            self.dim = 100
            self.top_k = 100
            self.db_size = 1183514
            self.query_size = 10000

            self.database_bin = str(PAG_DIR / "glove100" / "glove100_base.bin")
            self.query_bin = str(PAG_DIR / "glove100" / "glove100_query.bin")
            self.database_txt = str(DATA_DIR / "glove100_base.txt")
            self.query_txt = str(DATA_DIR / "glove100_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "glove100" / "glove100_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "glove100" / "glove100_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "glove100" / "glove100_truth500.bin")

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
            self._prefer_dataset_bin_layout()
            return

        if self.name == "glove200":
            self.dim = 200
            self.top_k = 100
            self.db_size = 1183514
            self.query_size = 10000

            self.database_bin = str(PAG_DIR / "glove200" / "glove200_base.bin")
            self.query_bin = str(PAG_DIR / "glove200" / "glove200_query.bin")
            self.database_txt = str(DATA_DIR / "glove200_base.txt")
            self.query_txt = str(DATA_DIR / "glove200_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "glove200" / "glove200_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "glove200" / "glove200_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "glove200" / "glove200_truth500.bin")

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
            self._prefer_dataset_bin_layout()
            return

        if self.name == "dinov2":
            self.dim = 768
            self.top_k = 100
            # Keep defaults aligned with common ImageNet-1K split;
            # actual counts are updated from loaded arrays in benchmark.py.
            self.db_size = 1281167
            self.query_size = 50000

            self.database_bin = str(PAG_DIR / "dinov2" / "dinov2_base.bin")
            self.query_bin = str(PAG_DIR / "dinov2" / "dinov2_query.bin")
            self.database_txt = str(DATA_DIR / "dinov2_base.txt")
            self.query_txt = str(DATA_DIR / "dinov2_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "dinov2" / "dinov2_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "dinov2" / "dinov2_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "dinov2" / "dinov2_truth500.bin")

            self.mag_knng = str(MAG_DIR / "dinov2.knng")
            self.mag_index = str(MAG_DIR / "dinov2.mag")
            self.mag_result = str(MAG_DIR / "result_mag_dinov2.txt")
            self.mag_efs = [100, 200, 400, 600]

            # Use dot-product ScaNN for consistency with the benchmark baselines.
            self.scann_result = str(RESULTS_DIR / "result_scann_dinov2.txt")
            self.scann_distance = "dot_product"
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
            self._prefer_dataset_bin_layout()
            return

        if self.name == "book_corpus":
            self.dim = 1024
            self.top_k = 100
            # actual counts are updated from loaded arrays in benchmark.py.
            self.db_size = 9250529
            self.query_size = 10000

            self.database_bin = str(PAG_DIR / "book_corpus" / "book_corpus_base.bin")
            self.query_bin = str(PAG_DIR / "book_corpus" / "book_corpus_query.bin")
            self.database_txt = str(DATA_DIR / "book_corpus_base.txt")
            self.query_txt = str(DATA_DIR / "book_corpus_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "book_corpus" / "book_corpus_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "book_corpus" / "book_corpus_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "book_corpus" / "book_corpus_truth500.bin")

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
            self._prefer_dataset_bin_layout()
            return

        if self.name == "gist1m":
            self.dim = 960
            self.top_k = 100
            self.db_size = 1000000
            self.query_size = 1000

            self.database_bin = str(PAG_DIR / "gist1m" / "gist1m_base.bin")
            self.query_bin = str(PAG_DIR / "gist1m" / "gist1m_query.bin")
            self.database_txt = str(DATA_DIR / "gist1m_base.txt")
            self.query_txt = str(DATA_DIR / "gist1m_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "gist1m" / "gist1m_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "gist1m" / "gist1m_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "gist1m" / "gist1m_truth500.bin")

            self.mag_knng = str(MAG_DIR / "gist1m.knng")
            self.mag_index = str(MAG_DIR / "gist1m.mag")
            self.mag_result = str(MAG_DIR / "result_mag_gist1m.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000]

            self.scann_result = str(RESULTS_DIR / "result_scann_gist1m.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 2000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [400, 500, 600, 800, 1000, 1500, 2000, 3000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_gist1m.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_gist1m.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600, 800, 1000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_gist1m.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_gist1m.data")
            self.mobius_result = str(MOBIUS_DIR / "result_gist1m.txt")
            self.mobius_budget_values = [50, 80, 100, 150, 200, 300, 500, 800, 1000]

            self.pag_hnsw_efc = 500
            self.pag_hnsw_M = 32
            self.pag_hnsw_L = 16

            self.faiss_result = str(RESULTS_DIR / "result_faiss_gist1m.txt")
            self._prefer_dataset_bin_layout()
            return

        if self.name == "ir101":
            self.dim = 512
            self.top_k = 100
            self.db_size = 17091649
            self.query_size = 20000

            self.database_bin = str(PAG_DIR / "ir101" / "ir101_base.bin")
            self.query_bin = str(PAG_DIR / "ir101" / "ir101_query.bin")
            self.database_txt = str(DATA_DIR / "ir101_base.txt")
            self.query_txt = str(DATA_DIR / "ir101_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "ir101" / "ir101_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "ir101" / "ir101_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "ir101" / "ir101_truth500.bin")

            self.mag_knng = str(MAG_DIR / "ir101.knng")
            self.mag_index = str(MAG_DIR / "ir101.mag")
            self.mag_result = str(MAG_DIR / "result_mag_ir101.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000, 1500, 2000]

            self.scann_result = str(RESULTS_DIR / "result_scann_ir101.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 4000
            self.scann_leaves_to_search = 200
            self.scann_reorder_values = [100, 200, 400, 800, 1000, 1500, 2000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_ir101.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_ir101.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600, 800, 1000, 1500, 2000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_ir101.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_ir101.data")
            self.mobius_result = str(MOBIUS_DIR / "result_ir101.txt")
            self.mobius_budget_values = [80, 100, 150, 200, 500, 1000, 1500, 2000]

            self.pag_hnsw_efc = 1000
            self.pag_hnsw_M = 64
            self.pag_hnsw_L = 32

            self.faiss_result = str(RESULTS_DIR / "result_faiss_ir101.txt")
            self._prefer_dataset_bin_layout()
            return

        if self.name == "openai1536":
            self.dim = 1536
            self.top_k = 100
            self.db_size = 1000000
            self.query_size = 1000

            self.database_bin = str(PAG_DIR / "openai1536" / "openai1536_base.bin")
            self.query_bin = str(PAG_DIR / "openai1536" / "openai1536_query.bin")
            self.database_txt = str(DATA_DIR / "openai1536_base.txt")
            self.query_txt = str(DATA_DIR / "openai1536_query.txt")

            self.groundtruth_txt_top100 = str(PAG_DIR / "openai1536" / "openai1536_truth.bin")
            self.groundtruth_bin_top100 = str(PAG_DIR / "openai1536" / "openai1536_truth.bin")
            self.groundtruth_bin_top500 = str(PAG_DIR / "openai1536" / "openai1536_truth500.bin")

            self.mag_knng = str(MAG_DIR / "openai1536.knng")
            self.mag_index = str(MAG_DIR / "openai1536.mag")
            self.mag_result = str(MAG_DIR / "result_mag_openai1536.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000]

            self.scann_result = str(RESULTS_DIR / "result_scann_openai1536.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 1000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [100, 200, 400, 800, 1000, 1500, 2000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_openai1536.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_openai1536.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600, 800, 1000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_openai1536.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_openai1536.data")
            self.mobius_result = str(MOBIUS_DIR / "result_openai1536.txt")
            self.mobius_budget_values = [50, 80, 100, 150, 200, 300, 500, 800, 1000]

            self.pag_hnsw_efc = 500
            self.pag_hnsw_M = 32
            self.pag_hnsw_L = 16

            self.faiss_result = str(RESULTS_DIR / "result_faiss_openai1536.txt")
            self._prefer_dataset_bin_layout()
            return

        if self.name == "word2vec":
            self.dim = 300
            self.top_k = 100
            self.db_size = 1000000
            self.query_size = 1000

            self.database_bin = str(DATA_DIR / "word2vec" / "word2vec_base.bin")
            self.query_bin = str(DATA_DIR / "word2vec" / "word2vec_query.bin")
            self.database_txt = str(DATA_DIR / "word2vec_base.txt")
            self.query_txt = str(DATA_DIR / "word2vec_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "word2vec" / "word2vec_truth.bin")
            self.groundtruth_bin_top100 = str(DATA_DIR / "word2vec" / "word2vec_truth.bin")
            self.groundtruth_bin_top500 = str(DATA_DIR / "word2vec" / "word2vec_truth500.bin")

            self.mag_knng = str(MAG_DIR / "word2vec.knng")
            self.mag_index = str(MAG_DIR / "word2vec.mag")
            self.mag_result = str(MAG_DIR / "result_mag_word2vec.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000]

            self.scann_result = str(RESULTS_DIR / "result_scann_word2vec.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 2000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [400, 500, 600, 800, 1000, 1500, 2000, 3000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_word2vec.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_word2vec.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600, 800, 1000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_word2vec.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_word2vec.data")
            self.mobius_result = str(MOBIUS_DIR / "result_word2vec.txt")
            self.mobius_budget_values = [50, 80, 100, 150, 200, 300, 500, 800, 1000]

            self.pag_hnsw_efc = 1000
            self.pag_hnsw_M = 64
            self.pag_hnsw_L = 32

            self.faiss_result = str(RESULTS_DIR / "result_faiss_word2vec.txt")
            self._prefer_dataset_bin_layout()
            return

        if self.name == "msong":
            self.dim = 420
            self.top_k = 100
            self.db_size = 994185
            self.query_size = 1000

            self.database_bin = str(DATA_DIR / "msong" / "msong_base.bin")
            self.query_bin = str(DATA_DIR / "msong" / "msong_query.bin")
            self.database_txt = str(DATA_DIR / "msong_base.txt")
            self.query_txt = str(DATA_DIR / "msong_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "msong" / "msong_truth.bin")
            self.groundtruth_bin_top100 = str(DATA_DIR / "msong" / "msong_truth.bin")
            self.groundtruth_bin_top500 = str(DATA_DIR / "msong" / "msong_truth500.bin")

            self.mag_knng = str(MAG_DIR / "msong.knng")
            self.mag_index = str(MAG_DIR / "msong.mag")
            self.mag_result = str(MAG_DIR / "result_mag_msong.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000]

            self.scann_result = str(RESULTS_DIR / "result_scann_msong.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 2000
            self.scann_leaves_to_search = 100
            self.scann_reorder_values = [400, 500, 600, 800, 1000, 1500, 2000, 3000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_msong.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_msong.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600, 800, 1000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_msong.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_msong.data")
            self.mobius_result = str(MOBIUS_DIR / "result_msong.txt")
            self.mobius_budget_values = [50, 80, 100, 150, 200, 300, 500, 800, 1000]

            self.pag_hnsw_efc = 1000
            self.pag_hnsw_M = 64
            self.pag_hnsw_L = 32

            self.faiss_result = str(RESULTS_DIR / "result_faiss_msong.txt")
            self._prefer_dataset_bin_layout()
            return

        if self.name == "text2img10m":
            self.dim = 200
            self.top_k = 100
            self.db_size = 10000000
            self.query_size = 100000

            self.database_bin = str(DATA_DIR / "text2img10m" / "text2img_base.bin")
            self.query_bin = str(DATA_DIR / "text2img10m" / "text2img_query.bin")
            self.database_txt = str(DATA_DIR / "text2img_base.txt")
            self.query_txt = str(DATA_DIR / "text2img_query.txt")

            self.groundtruth_txt_top100 = str(DATA_DIR / "text2img10m" / "text2img10m_truth.bin")
            self.groundtruth_bin_top100 = str(DATA_DIR / "text2img10m" / "text2img10m_truth.bin")
            self.groundtruth_bin_top500 = str(DATA_DIR / "text2img10m" / "text2img10m_truth500.bin")

            self.mag_knng = str(MAG_DIR / "text2img10m.knng")
            self.mag_index = str(MAG_DIR / "text2img10m.mag")
            self.mag_result = str(MAG_DIR / "result_mag_text2img10m.txt")
            self.mag_efs = [100, 200, 400, 600, 800, 1000]

            self.scann_result = str(RESULTS_DIR / "result_scann_text2img10m.txt")
            self.scann_distance = "dot_product"
            self.scann_mode = "reorder"
            self.scann_num_leaves = 4000
            self.scann_leaves_to_search = 200
            self.scann_reorder_values = [200, 400, 600, 800, 1000, 1500, 2000, 3000]

            self.ipnsw_graph = str(IPNSW_DIR / "out_graph_text2img10m.hnsw")
            self.ipnsw_result = str(IPNSW_DIR / "result_text2img10m.txt")
            self.ipnsw_m = 32
            self.ipnsw_ef_construction = 500
            self.ipnsw_ef_values = [100, 200, 400, 600, 800, 1000, 1500, 2000]

            self.mobius_graph = str(MOBIUS_DIR / "bfsg_text2img10m.graph")
            self.mobius_data = str(MOBIUS_DIR / "bfsg_text2img10m.data")
            self.mobius_result = str(MOBIUS_DIR / "result_text2img10m.txt")
            self.mobius_budget_values = [80, 100, 150, 200, 300, 500, 800, 1000]

            self.pag_hnsw_efc = 1000
            self.pag_hnsw_M = 64
            self.pag_hnsw_L = 32

            self.faiss_result = str(RESULTS_DIR / "result_faiss_text2img10m.txt")
            self._prefer_dataset_bin_layout(prefix="text2img", truth_prefix="text2img10m")
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
    if config.top_k >= 500 and hasattr(config, 'groundtruth_bin_top500') and os.path.exists(config.groundtruth_bin_top500):
        return load_groundtruth_auto(config.groundtruth_bin_top500, config.query_size, config.top_k)
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


# ---------------------------------------------------------------------------
# Peak memory tracking via /proc (Linux)
# ---------------------------------------------------------------------------

def _scan_process_tree_rss_mb(root_pid: int) -> float:
    """Return total VmRSS (MB) of root_pid and all its descendants."""
    procs: dict[int, tuple[int, int]] = {}  # pid -> (ppid, vmrss_kb)
    try:
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            ppid = -1
            vmrss = 0
            try:
                with open(f"/proc/{pid}/status") as fh:
                    for line in fh:
                        if line.startswith("VmRSS:"):
                            vmrss = int(line.split()[1])
                        elif line.startswith("PPid:"):
                            ppid = int(line.split()[1])
                        if vmrss and ppid >= 0:
                            break
            except Exception:
                continue
            procs[pid] = (ppid, vmrss)
    except Exception:
        pass

    total_kb = 0
    queue = [root_pid]
    visited: set[int] = set()
    while queue:
        pid = queue.pop()
        if pid in visited:
            continue
        visited.add(pid)
        total_kb += procs.get(pid, (-1, 0))[1]
        for child_pid, (ppid, _) in procs.items():
            if ppid == pid and child_pid not in visited:
                queue.append(child_pid)
    return total_kb / 1024.0


class PeakMemoryTracker:
    """Background-thread peak RSS tracker for the current process tree.

    Polls /proc every *poll_interval* seconds and records the highest
    total VmRSS (MB) seen for the Python process and all its descendants.
    Use as a context manager::

        with PeakMemoryTracker() as t:
            do_work()
        print(t.peak_mb)
    """

    def __init__(self, poll_interval: float = 0.2):
        self._root_pid = os.getpid()
        self._interval = poll_interval
        self.peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.wait(self._interval):
            mb = _scan_process_tree_rss_mb(self._root_pid)
            if mb > self.peak_mb:
                self.peak_mb = mb

    def __enter__(self) -> "PeakMemoryTracker":
        self.peak_mb = _scan_process_tree_rss_mb(self._root_pid)
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
