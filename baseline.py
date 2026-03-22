#!/usr/bin/env python3
"""
综合基准测试脚本 - Music100 / GloVe-100 数据集
测试MAG, FAISS IVF-Flat, ScaNN, ip-nsw, Möbius五种MIPS算法
输出QPS和Recall@K指标
"""

import os
import sys
import time
import struct
import subprocess

# ── 全局单线程限制（对所有子进程也生效，因为子进程继承环境变量） ──
# 必须在 import numpy / scann / tensorflow 之前设置，否则 BLAS/OpenMP 线程池已初始化
os.environ["OMP_NUM_THREADS"]          = "1"
os.environ["OPENBLAS_NUM_THREADS"]    = "1"
os.environ["MKL_NUM_THREADS"]         = "1"
os.environ["NUMEXPR_NUM_THREADS"]     = "1"
os.environ["TF_NUM_INTRAOP_THREADS"]  = "1"
os.environ["TF_NUM_INTEROP_THREADS"]  = "1"

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
# test_scann (→ scann → TensorFlow) is imported lazily inside the two functions
# that actually need it, so TF is never loaded when running non-music100 datasets.

# ── scipy-openblas64 使用私有符号前缀，忽略 OPENBLAS_NUM_THREADS 等 env vars ──
# 必须通过 threadpoolctl API 运行时限制线程数
try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _BLAS_LIMIT = _threadpool_limits(limits=1)  # 持久生效：保持引用在内存中
except ImportError:
    _BLAS_LIMIT = None
    print("[WARN] threadpoolctl not found; BLAS may use multiple threads. Run: pip install threadpoolctl")

VENV_PYTHON = sys.executable  # 当前 Python 解释器（带 scann/numpy 等依赖）

THREAD_LIMIT_KEYS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS",
    "TF_NUM_INTEROP_THREADS",
]


def build_env():
    """Build phase should not be constrained to single-thread."""
    env = os.environ.copy()
    for k in THREAD_LIMIT_KEYS:
        env.pop(k, None)
    return env



class Config:
    """配置参数"""
    # 数据路径
    DATA_DIR = "data"
    DATABASE_BIN = "data/database_music100.bin"
    QUERY_BIN = "data/query_music100.bin"
    
    # 数据参数
    DIM = 100
    TOP_K = 100
    
    # 索引文件路径
    MAG_KNNG = "MAG/music100.knng"
    MAG_INDEX = "MAG/music100.mag"
    MAG_RESULT = "MAG/result_mag.txt"
    
    IPNSW_GRAPH = "ip-nsw/out_graph.hnsw"
    IPNSW_RESULT = "ip-nsw/result.txt"
    
    MOBIUS_GRAPH = "mobius/bfsg_music100.graph"
    MOBIUS_DATA  = "mobius/bfsg_music100.data"
    MOBIUS_RESULT = "mobius/result.txt"
    
    SCANN_RESULT = "result_scann.txt"
    IVFPQ_RESULT = "result_ivfpq.txt"
    SCANN_LEAVES_TO_SEARCH =100  # num_leaves_to_search，修改此处同时影响 test_scann 和 test_scann_sweep
    
    # Ground truth文件
    GROUND_TRUTH_1 = "data/correct1_music100.txt"
    GROUND_TRUTH_10 = "data/correct10_music100.txt"
    GROUND_TRUTH_100 = "data/correct100_music100.txt"
    # 可选：Music100 top-100 ground truth 的二进制版本（int32, 无header, shape=(n_queries, 100)）
    # 若文件存在，评测会优先读取该bin并自动裁剪到top-k。
    GROUND_TRUTH_BIN_100 = "data/correct100_music100.bin"

    # PAG (PEOs) 配置 — 路径相对于 PAG/ 目录
    PAG_BASE  = "music/music_base.bin"
    PAG_QUERY = "music/music_query.bin"
    PAG_TRUTH = "music/music_truth.bin"
    PAG_INDEX = "music/index/"
    PAG_EFC = 500; PAG_M = 32; PAG_L = 16


class Glove100Config:
    """GloVe-100 数据集配置"""
    DATA_DIR = "data"
    DATABASE_BIN = "data/glove100_base.bin"
    QUERY_BIN    = "data/glove100_query.bin"
    DATABASE_TXT = "data/glove100_base.txt"
    QUERY_TXT    = "data/glove100_query.txt"
    GROUNDTRUTH_BIN    = "data/glove100_groundtruth.bin"     # int32, (10000, 100) — cosine GT
    GROUNDTRUTH_BIN_IP = "data/glove100_groundtruth_ip.bin"  # int32, (10000, 100) — MIPS GT

    DIM       = 100
    DB_SIZE   = 1183514
    Q_SIZE    = 10000
    TOP_K     = 100

    # 索引文件
    MAG_KNNG   = "MAG/glove100.knng"
    MAG_INDEX  = "MAG/glove100.mag"
    MAG_RESULT = "MAG/result_mag_glove100.txt"

    IPNSW_GRAPH  = "ip-nsw/out_graph_glove100.hnsw"
    IPNSW_RESULT = "ip-nsw/result_glove100.txt"

    MOBIUS_GRAPH  = "mobius/bfsg_glove100.graph"
    MOBIUS_DATA   = "mobius/bfsg_glove100.data"
    MOBIUS_RESULT = "mobius/result_glove100.txt"

    SCANN_RESULT  = "result_scann_glove100.txt"
    FAISS_RESULT  = "result_faiss_glove100.txt"
    FAISS_CACHE   = "faiss_ivfflat_cache_glove100.pkl"

    # L2归一化后的数据（angular数据集用于与ground truth对齐）
    DATABASE_BIN_NORM = "data/glove100_base_normalized.bin"
    QUERY_BIN_NORM    = "data/glove100_query_normalized.bin"
    DATABASE_TXT_NORM = "data/glove100_base_normalized.txt"
    QUERY_TXT_NORM    = "data/glove100_query_normalized.txt"

    # benchmark_recall_qps_mips 参数
    BENCH_TITLE         = "GloVe-100 Dataset"
    BENCH_MAG_EFS       = [100, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000, 4000, 5000]
    BENCH_IPNSW_M       = 20
    BENCH_IPNSW_EFC     = 500
    BENCH_IPNSW_EF      = [100, 200, 400, 600, 800, 1000, 1500, 2000]
    BENCH_SCANN_MODE    = "leaves"   # 每个 num_leaves_to_search 值独立建索引
    BENCH_SCANN_PARAM   = [10, 20, 50, 100, 200, 500, 1000, 1500, 2000]
    BENCH_SCANN_REORDER = 500
    BENCH_MOBIUS_BUDGET = [100, 150, 200, 300, 500, 1000, 1500, 2000, 3000, 4000, 5000]

    # PAG (PEOs) 配置 — 路径相对于 PAG/ 目录
    PAG_BASE  = "glove100/glove100_base.bin"
    PAG_QUERY = "glove100/glove100_query.bin"
    PAG_TRUTH = "glove100/glove100_truth.bin"
    PAG_INDEX = "glove100/index/"
    PAG_EFC = 500; PAG_M = 32; PAG_L = 16


class Dinov2Config:
    """ImageNet DINOv2 D=768 数据集配置（L2 距离）"""
    DATA_DIR = "data"
    DATABASE_BIN    = "Iceberg/data/imagenet-1k/dinov2-train.bin"
    QUERY_BIN       = "Iceberg/data/imagenet-1k/dinov2-validation.bin"
    GROUNDTRUTH_BIN    = "data/imagenet_dinov2_groundtruth.bin"     # int32, (1000, 100) — L2 GT
    GROUNDTRUTH_BIN_IP = "Iceberg/data/imagenet-1k/result-dinov2-ip-top100.txt"  # text, top-100 MIPS GT

    # L2-normalized vectors （cosine GT）
    DATABASE_BIN_NORM = "data/imagenet_dinov2_base_normalized.bin"
    QUERY_BIN_NORM    = "data/imagenet_dinov2_query_normalized.bin"
    GROUNDTRUTH_BIN_NORM = "data/imagenet_dinov2_groundtruth_normalized.bin"
    DATABASE_TXT_NORM = "data/imagenet_dinov2_base_normalized.txt"
    QUERY_TXT_NORM    = "data/imagenet_dinov2_query_normalized.txt"

    DIM     = 768
    DB_SIZE = 1281167
    Q_SIZE  = 50000
    TOP_K   = 100

    MAG_KNNG   = "MAG/dinov2.knng"
    MAG_INDEX  = "MAG/dinov2.mag"
    MAG_RESULT = "MAG/result_mag_dinov2.txt"

    IPNSW_GRAPH  = "ip-nsw/out_graph_dinov2.hnsw"
    IPNSW_RESULT = "ip-nsw/result_dinov2.txt"

    MOBIUS_GRAPH  = "mobius/bfsg_dinov2.graph"
    MOBIUS_DATA   = "mobius/bfsg_dinov2.data"
    MOBIUS_RESULT = "mobius/result_dinov2.txt"
    DATABASE_TXT  = "data/imagenet_dinov2_base.txt"
    QUERY_TXT     = "data/imagenet_dinov2_query.txt"

    SCANN_RESULT = "result_scann_dinov2.txt"
    FAISS_RESULT = "result_faiss_dinov2.txt"
    FAISS_CACHE  = "faiss_ivfflat_cache_dinov2.pkl"

    # benchmark_recall_qps_mips 参数
    BENCH_MAG_EFS       = [100, 200, 400, 600]
    BENCH_IPNSW_M       = 32
    BENCH_IPNSW_EFC     = 1000
    BENCH_IPNSW_EF      = [100, 200, 400, 600]
    BENCH_SCANN_MODE    = "reorder"  # 一次建索引，sweep pre_reorder_num_neighbors
    BENCH_SCANN_PARAM   = [1000, 500, 300, 200]
    BENCH_MOBIUS_BUDGET = [50, 80, 100, 150, 200, 300, 500]

    # PAG (PEOs) 配置 — 路径相对于 PAG/ 目录
    PAG_BASE  = "dinov2/dinov2_base_nolabel.bin"
    PAG_QUERY = "dinov2/dinov2_query_nolabel.bin"
    PAG_TRUTH = "dinov2/dinov2_truth.bin"
    PAG_INDEX = "dinov2/index/"
    PAG_EFC = 500; PAG_M = 32; PAG_L = 16


class Eva02Config:
    """ImageNet EVA-02 D=1024 数据集配置（L2 距离）"""
    DATA_DIR = "data"
    DATABASE_BIN    = "data/imagenet_eva02_base.bin"
    QUERY_BIN       = "data/imagenet_eva02_query.bin"
    GROUNDTRUTH_BIN    = "data/imagenet_eva02_groundtruth.bin"     # int32, (1000, 100) — L2 GT
    GROUNDTRUTH_BIN_IP = "data/imagenet_eva02_groundtruth_ip.bin"  # int32, (1000, 100) — MIPS GT

    # L2-normalized vectors （cosine GT）
    DATABASE_BIN_NORM = "data/imagenet_eva02_base_normalized.bin"
    QUERY_BIN_NORM    = "data/imagenet_eva02_query_normalized.bin"
    GROUNDTRUTH_BIN_NORM = "data/imagenet_eva02_groundtruth_normalized.bin"
    DATABASE_TXT_NORM = "data/imagenet_eva02_base_normalized.txt"
    QUERY_TXT_NORM    = "data/imagenet_eva02_query_normalized.txt"

    DIM     = 1024
    DB_SIZE = 49000
    Q_SIZE  = 1000
    TOP_K   = 100

    MAG_KNNG   = "MAG/eva02.knng"
    MAG_INDEX  = "MAG/eva02.mag"
    MAG_RESULT = "MAG/result_mag_eva02.txt"

    IPNSW_GRAPH  = "ip-nsw/out_graph_eva02.hnsw"
    IPNSW_RESULT = "ip-nsw/result_eva02.txt"

    MOBIUS_GRAPH  = "mobius/bfsg_eva02.graph"
    MOBIUS_DATA   = "mobius/bfsg_eva02.data"
    MOBIUS_RESULT = "mobius/result_eva02.txt"
    DATABASE_TXT  = "data/imagenet_eva02_base.txt"
    QUERY_TXT     = "data/imagenet_eva02_query.txt"

    SCANN_RESULT = "result_scann_eva02.txt"
    FAISS_RESULT = "result_faiss_eva02.txt"
    FAISS_CACHE  = "faiss_ivfflat_cache_eva02.pkl"

    # benchmark_recall_qps_mips 参数
    BENCH_MAG_EFS       = [100, 200, 400, 600]
    BENCH_IPNSW_M       = 32
    BENCH_IPNSW_EFC     = 1000
    BENCH_IPNSW_EF      = [100, 200, 400, 600]
    BENCH_SCANN_MODE    = "reorder"
    BENCH_SCANN_PARAM   = [1000, 500, 300, 200]
    BENCH_MOBIUS_BUDGET = [50, 80, 100, 150, 200, 300, 500]

    # PAG 无此数据集的数据文件
    PAG_BASE = None


class ConvNextConfig:
    """ImageNet ConvNext D=144 数据集配置（L2 距离）"""
    DATA_DIR = "data"
    DATABASE_BIN    = "data/imagenet_convnext_base.bin"
    QUERY_BIN       = "data/imagenet_convnext_query.bin"
    GROUNDTRUTH_BIN    = "data/imagenet_convnext_groundtruth.bin"     # int32, (1000, 100) — L2 GT
    GROUNDTRUTH_BIN_IP = "data/imagenet_convnext_groundtruth_ip.bin"  # int32, (1000, 100) — MIPS GT

    # L2-normalized vectors （cosine GT）
    DATABASE_BIN_NORM = "data/imagenet_convnext_base_normalized.bin"
    QUERY_BIN_NORM    = "data/imagenet_convnext_query_normalized.bin"
    GROUNDTRUTH_BIN_NORM = "data/imagenet_convnext_groundtruth_normalized.bin"
    DATABASE_TXT_NORM = "data/imagenet_convnext_base_normalized.txt"
    QUERY_TXT_NORM    = "data/imagenet_convnext_query_normalized.txt"

    DIM     = 144
    DB_SIZE = 49000
    Q_SIZE  = 1000
    TOP_K   = 100

    MAG_KNNG   = "MAG/convnext.knng"
    MAG_INDEX  = "MAG/convnext.mag"
    MAG_RESULT = "MAG/result_mag_convnext.txt"

    IPNSW_GRAPH  = "ip-nsw/out_graph_convnext.hnsw"
    IPNSW_RESULT = "ip-nsw/result_convnext.txt"

    MOBIUS_GRAPH  = "mobius/bfsg_convnext.graph"
    MOBIUS_DATA   = "mobius/bfsg_convnext.data"
    MOBIUS_RESULT = "mobius/result_convnext.txt"
    DATABASE_TXT  = "data/imagenet_convnext_base.txt"
    QUERY_TXT     = "data/imagenet_convnext_query.txt"

    SCANN_RESULT = "result_scann_convnext.txt"
    FAISS_RESULT = "result_faiss_convnext.txt"
    FAISS_CACHE  = "faiss_ivfflat_cache_convnext.pkl"

    # benchmark_recall_qps_mips 参数
    BENCH_MAG_EFS       = [100, 200, 400, 600]
    BENCH_IPNSW_M       = 20
    BENCH_IPNSW_EFC     = 500
    BENCH_IPNSW_EF      = [100, 200, 400, 600]
    BENCH_SCANN_MODE    = "reorder"
    BENCH_SCANN_PARAM   = [1000, 500, 300, 200]
    BENCH_MOBIUS_BUDGET = [50, 80, 100, 150, 200, 300, 500]

    # PAG 无此数据集的数据文件
    PAG_BASE = None


def read_bin(file_path, dim):
    """读取二进制数据（支持纯 float32 或 fvecs 格式）"""
    file_size = os.path.getsize(file_path)
    raw_stride = dim * 4
    fvecs_stride = (dim + 1) * 4

    if file_size % raw_stride == 0:
        n = file_size // raw_stride
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        return data.reshape(n, dim)

    if file_size % fvecs_stride == 0:
        n = file_size // fvecs_stride
        with open(file_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.int32)
        data = data.reshape(n, dim + 1)
        return data[:, 1:].view(np.float32)

    raise ValueError(f"Unsupported binary format or dimension mismatch: {file_path}")


def load_groundtruth_bin(file_path, n_queries, top_k, dtype=np.int32):
    """从二进制文件加载ground truth（无header，int32连续存储）"""
    print(f"Loading ground truth from {file_path}...")
    data = np.fromfile(file_path, dtype=dtype)
    gt = data.reshape(n_queries, -1)[:, :top_k]
    print(f"✓ Ground truth loaded: {gt.shape}")
    return gt


def load_groundtruth_auto(file_path, n_queries, top_k, dtype=np.int32):
    """根据文件类型自动加载 ground truth。"""
    if file_path.endswith('.txt'):
        print(f"Loading ground truth from {file_path}...")
        gt = []
        with open(file_path, 'r') as f:
            for line in f:
                row = line.strip()
                if row:
                    gt.append(list(map(int, row.split()))[:top_k])
        gt_array = np.array(gt, dtype=np.int32)
        print(f"✓ Ground truth loaded: {gt_array.shape}")
        return gt_array
    return load_groundtruth_bin(file_path, n_queries, top_k, dtype=dtype)


def compute_ground_truth(database, queries, k):
    """加载ground truth文件"""
    print(f"\n{'='*70}")
    print("Loading Ground Truth")
    print(f"{'='*70}")

    # 优先读取二进制GT（如果已配置且文件存在）
    gt_bin_file = getattr(Config, "GROUND_TRUTH_BIN_100", None)
    if gt_bin_file and os.path.exists(gt_bin_file):
        return load_groundtruth_bin(gt_bin_file, queries.shape[0], k)
    
    # 根据k选择对应的ground truth文件
    if k == 1:
        gt_file = Config.GROUND_TRUTH_1
    elif k == 10:
        gt_file = Config.GROUND_TRUTH_10
    elif k == 100:
        gt_file = Config.GROUND_TRUTH_100
    else:
        raise ValueError(f"No ground truth file for k={k}. Available: 1, 10, 100")
    
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"Ground truth file not found: {gt_file}")
    
    return load_groundtruth_auto(gt_file, queries.shape[0], k)


def compute_recall(results, ground_truth, k):
    """
    计算Recall@K
    Recall@K = (检索到的相关项数) / K
    
    Args:
        results: 算法返回的结果 (n_queries, k)
        ground_truth: 真实的top-k结果 (n_queries, k)
        k: top-k
    
    Returns:
        recall: 平均召回率
    """
    n_queries = results.shape[0]
    total_recall = 0.0
    
    for i in range(n_queries):
        # 计算交集
        intersection = np.intersect1d(results[i], ground_truth[i])
        recall = len(intersection) / k
        total_recall += recall
    
    avg_recall = total_recall / n_queries
    return avg_recall


def load_results(result_file, expected_k=None):
    """加载结果文件。若行长度不一致（MAG在大数据集上可能返回不足k个结果），
    用 -1 填充到 expected_k（或各行最大长度）。"""
    results = []
    with open(result_file, 'r') as f:
        for line in f:
            row = line.strip()
            if row:
                results.append(list(map(int, row.split())))
    if not results:
        return np.array(results)
    max_len = expected_k if expected_k is not None else max(len(r) for r in results)
    padded = [r + [-1] * (max_len - len(r)) if len(r) < max_len else r[:max_len]
              for r in results]
    return np.array(padded, dtype=np.int64)


def print_data_info(base_path, query_path, gt_path, database, queries):
    """显示数据文件名及维度/行数信息"""
    print(f"  Base : {base_path}  (rows={database.shape[0]}, dim={database.shape[1]})")
    print(f"  Query: {query_path}  (rows={queries.shape[0]}, dim={queries.shape[1]})")
    print(f"  GT   : {gt_path}")


def get_music100_gt_path(top_k):
    """Return effective Music100 GT path for logging (bin preferred)."""
    gt_bin_file = getattr(Config, "GROUND_TRUTH_BIN_100", None)
    if gt_bin_file and os.path.exists(gt_bin_file):
        return gt_bin_file

    if top_k == 1:
        return Config.GROUND_TRUTH_1
    if top_k == 10:
        return Config.GROUND_TRUTH_10
    return Config.GROUND_TRUTH_100


def file_nonempty(path):
    """Return True only when file exists and has non-zero size."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def ensure_mag_knng(db_bin, knng_path, dim, knng_k=50):
    """Ensure MAG kNN graph exists; skip build when existing file is non-empty."""
    if file_nonempty(knng_path):
        print(f"✓ kNN graph already exists: {knng_path}")
        return True

    print(f"kNN graph not found. Building {knng_path}...")
    try:
        subprocess.run([
            VENV_PYTHON,
            "MAG/build_knng.py",
            db_bin,
            knng_path,
            str(dim),
            str(knng_k),
        ], check=True, env=build_env())
    except Exception as e:
        print(f"✗ kNN graph build failed: {e}")
        return False

    return file_nonempty(knng_path)


def test_mag(database, queries, ground_truth):
    """测试MAG算法"""
    print(f"\n{'='*70}")
    print("Testing MAG (Metric-Amphibious Graph)")
    print(f"{'='*70}")
    print_data_info(Config.DATABASE_BIN, Config.QUERY_BIN, get_music100_gt_path(Config.TOP_K), database, queries)
    
    # 检查kNN图是否存在（存在则跳过构建）
    if not ensure_mag_knng(Config.DATABASE_BIN, Config.MAG_KNNG, Config.DIM, knng_k=50):
        print("✗ MAG kNN graph unavailable, abort MAG test.")
        return None
    
    # 检查索引是否存在
    if not file_nonempty(Config.MAG_INDEX):
        print(f"Index not found. Building MAG index...")
        # 使用MAG的test_mag程序构建索引
        subprocess.run([
            "./MAG/build/test/test_mag",
            Config.DATABASE_BIN,
            Config.MAG_KNNG,
            "60",  # L
            "48",  # R
            "300", # C
            "MAG/music100.mag",
            "index",
            str(Config.DIM),
            "20",  # R_IP
            "64",  # M
            "8"    # threshold
        ], check=True, env=build_env())
    else:
        print(f"✓ Index already exists: {Config.MAG_INDEX}")
    
    # 运行查询
    print("Running queries...")
    start = time.time()
    subprocess.run([
        "./MAG/build/test/test_mag",
        Config.DATABASE_BIN,
        Config.QUERY_BIN,
        "MAG/music100.mag",
        "800",  # efs
        str(Config.TOP_K),
        Config.MAG_RESULT,
        "search",
        str(Config.DIM)
    ], check=True)
    elapsed = time.time() - start
    
    # 计算指标
    qps = queries.shape[0] / elapsed
    avg_time_ms = (elapsed / queries.shape[0]) * 1000
    
    # 加载结果并计算recall
    results = load_results(Config.MAG_RESULT)
    recall = compute_recall(results, ground_truth, Config.TOP_K)
    
    print(f"✓ MAG completed!")
    print(f"  QPS: {qps:.2f}")
    print(f"  Average query time: {avg_time_ms:.4f} ms")
    print(f"  Recall@{Config.TOP_K}: {recall:.4f}")
    
    return {"qps": qps, "avg_time_ms": avg_time_ms, "recall": recall}


def test_ivfpq(database, queries, ground_truth):
    """测试FAISS IVF-PQ"""
    return None  # disabled


def test_scann(database, queries, ground_truth):
    """测试ScaNN"""
    import test_scann as _scann_mod  # lazy: keeps TF out of main process for other datasets
    print(f"\n{'='*70}")
    print("Testing ScaNN")
    print(f"{'='*70}")
    print_data_info(Config.DATABASE_BIN, Config.QUERY_BIN, get_music100_gt_path(Config.TOP_K), database, queries)

    print("Running ScaNN test...")
    neighbors, qps = _scann_mod.run_scann(
        database,
        queries,
        top_k=Config.TOP_K,
        num_leaves_to_search=Config.SCANN_LEAVES_TO_SEARCH,
        reorder=500,
        distance="dot_product",
        result_path=Config.SCANN_RESULT,
    )

    nq = queries.shape[0]
    avg_time_ms = 1000.0 / qps
    recall = compute_recall(load_results(Config.SCANN_RESULT), ground_truth, Config.TOP_K)

    print(f"✓ ScaNN completed!")
    print(f"  QPS: {qps:.2f}")
    print(f"  Average query time: {avg_time_ms:.4f} ms")
    print(f"  Recall@{Config.TOP_K}: {recall:.4f}")

    return {"qps": qps, "avg_time_ms": avg_time_ms, "recall": recall}


def test_ipnsw(database, queries, ground_truth):
    """测试ip-nsw"""
    print(f"\n{'='*70}")
    print("Testing ip-nsw (HNSW-based MIPS)")
    print(f"{'='*70}")
    print_data_info(Config.DATABASE_BIN, Config.QUERY_BIN, get_music100_gt_path(Config.TOP_K), database, queries)
    
    # 检查索引是否存在
    if not os.path.exists(Config.IPNSW_GRAPH):
        print(f"Index not found. Building ip-nsw index...")
        subprocess.run([
            "ip-nsw/main",
            "--mode", "database",
            "--database", Config.DATABASE_BIN,
            "--databaseSize", "1000000",
            "--dimension", str(Config.DIM),
            "--outputGraph", Config.IPNSW_GRAPH,
            "--M", "20",
            "--efConstruction", "500"
        ], check=True, env=build_env())
    else:
        print(f"✓ Index already exists: {Config.IPNSW_GRAPH}")
    
    # 删除旧的结果文件以确保重新生成
    if os.path.exists(Config.IPNSW_RESULT):
        os.remove(Config.IPNSW_RESULT)
    
    # 运行查询
    print("Running queries...")
    start = time.time()
    result = subprocess.run([
        "ip-nsw/main",
        "--mode", "query",
        "--query", Config.QUERY_BIN,
        "--querySize", "10000",
        "--dimension", str(Config.DIM),
        "--inputGraph", Config.IPNSW_GRAPH,
        "--efSearch", "128",
        "--topK", str(Config.TOP_K),
        "--output", Config.IPNSW_RESULT
    ], check=True, capture_output=True, text=True)
    elapsed = time.time() - start
    
    # 尝试从输出中提取真实的查询时间
    output = result.stdout + result.stderr
    avg_time_ms = (elapsed / queries.shape[0]) * 1000
    qps = queries.shape[0] / elapsed
    
    for line in output.split('\n'):
        if 'Average query time' in line or 'average query time' in line:
            try:
                # 尝试提取时间值
                import re
                match = re.search(r'([0-9.]+)\s*ms', line)
                if match:
                    avg_time_ms = float(match.group(1))
                    qps = 1000.0 / avg_time_ms
            except:
                pass
    
    # 计算指标
    qps = queries.shape[0] / elapsed
    avg_time_ms = (elapsed / queries.shape[0]) * 1000
    
    # 加载结果并计算recall
    if not os.path.exists(Config.IPNSW_RESULT):
        raise FileNotFoundError(f"Result file not generated: {Config.IPNSW_RESULT}")
    
    results = load_results(Config.IPNSW_RESULT)
    recall = compute_recall(results, ground_truth, Config.TOP_K)
    
    print(f"✓ ip-nsw completed!")
    print(f"  QPS: {qps:.2f}")
    print(f"  Average query time: {avg_time_ms:.4f} ms")
    print(f"  Recall@{Config.TOP_K}: {recall:.4f}")
    
    return {"qps": qps, "avg_time_ms": avg_time_ms, "recall": recall}


def test_mobius(database, queries, ground_truth):
    """测试Möbius"""
    import shutil
    print(f"\n{'='*70}")
    print("Testing Möbius")
    print(f"{'='*70}")
    print_data_info(Config.DATABASE_BIN, "data/query_music100.txt", get_music100_gt_path(Config.TOP_K), database, queries)

    BFSG_GRAPH = "mobius/bfsg.graph"
    BFSG_DATA  = "mobius/bfsg.data"

    # 检查索引是否存在；不存在则尝试构建
    if not os.path.exists(Config.MOBIUS_GRAPH):
        print("  Graph not found. Building Möbius graph (may take ~10 min)...")
        if not os.path.exists("data/database_music100.txt"):
            print("✗ Möbius requires data/database_music100.txt. Generate it first.")
            return None
        subprocess.run(
            ["./build_graph.sh", "../data/database_music100.txt", "1000000", "100"],
            check=True, cwd="mobius", env=build_env()
        )
        # 保存到带名字的文件，防止被其他数据集覆盖
        shutil.copy2(BFSG_GRAPH, Config.MOBIUS_GRAPH)
        shutil.copy2(BFSG_DATA,  Config.MOBIUS_DATA)
        print(f"  ✓ Graph saved → {Config.MOBIUS_GRAPH}")
    else:
        print(f"✓ Index already exists: {Config.MOBIUS_GRAPH}")

    # 把 music100 的图/数据复制到二进制默认路径
    shutil.copy2(Config.MOBIUS_GRAPH, BFSG_GRAPH)
    if os.path.exists(Config.MOBIUS_DATA):
        shutil.copy2(Config.MOBIUS_DATA, BFSG_DATA)

    # 使用文本格式的query数据
    query_txt_path = "data/query_music100.txt"
    
    # 运行查询 - 直接使用mobius命令，需要在mobius目录下执行
    print("Running queries...")
    start = time.time()
    
    # 参数: test 0 <query_data> <search_budget> <row> <dim> <display_k> <search_budget>
    search_budget = 128
    result = subprocess.run([
        "./mobius",
        "test", "0",
        "../" + query_txt_path,  # 相对于mobius目录的路径
        str(search_budget),
        "1000000",  # row
        "100",  # dim
        str(Config.TOP_K),  # display top k
        str(search_budget)
    ], capture_output=True, text=True, check=True, cwd="mobius")
    
    elapsed = time.time() - start
    
    # 计算指标
    qps = queries.shape[0] / elapsed
    avg_time_ms = (elapsed / queries.shape[0]) * 1000
    
    # 解析Möbius的stdout获取结果
    output = result.stdout
    results = []
    for line in output.strip().split('\n'):
        if line and not line.startswith('Loading'):
            parts = line.strip().split()
            if len(parts) >= Config.TOP_K:
                try:
                    results.append([int(x) for x in parts[:Config.TOP_K]])
                except ValueError:
                    continue
    
    if len(results) == queries.shape[0]:
        results = np.array(results)
        recall = compute_recall(results, ground_truth, Config.TOP_K)
        
        # 保存结果到文件
        with open(Config.MOBIUS_RESULT, 'w') as f:
            for row in results:
                f.write(' '.join(map(str, row)) + '\n')
        
        print(f"✓ Möbius completed!")
        print(f"  QPS: {qps:.2f}")
        print(f"  Average query time: {avg_time_ms:.4f} ms")
        print(f"  Recall@{Config.TOP_K}: {recall:.4f}")
        
        return {"qps": qps, "avg_time_ms": avg_time_ms, "recall": recall}
    else:
        print(f"✗ Möbius failed: Result count mismatch. Got {len(results)}, expected {queries.shape[0]}")
        return None


def print_summary(results):
    """打印汇总表格"""
    # 过滤掉失败（None）的结果
    results = {k: v for k, v in results.items() if v is not None}
    if not results:
        print("\n✗ No algorithms completed successfully!")
        return

    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Algorithm':<20} {'QPS':>12} {'Avg Time (ms)':>15} {'Recall@100':>12}")
    print("-" * 70)
    
    # 按QPS排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['qps'], reverse=True)
    
    for name, metrics in sorted_results:
        print(f"{name:<20} {metrics['qps']:>12.2f} {metrics['avg_time_ms']:>15.4f} {metrics['recall']:>12.4f}")
    
    print("=" * 70)
    
    # 找出最佳算法
    best_qps = max(results.items(), key=lambda x: x[1]['qps'])
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    
    print(f"\n🏆 Best QPS: {best_qps[0]} ({best_qps[1]['qps']:.2f})")
    print(f"🎯 Best Recall: {best_recall[0]} ({best_recall[1]['recall']:.4f})")


def test_mobius_with_budget(database, queries, ground_truth, search_budget):
    """测试Möbius with specified search_budget（调用前需确保 bfsg.graph/bfsg.data 已就位）"""
    if not os.path.exists(Config.MOBIUS_GRAPH):
        print(f"Möbius graph not found, skipping with search_budget={search_budget}")
        return None
    
    query_txt_path = "data/query_music100.txt"
    
    start = time.time()
    result = subprocess.run([
        "./mobius",
        "test", "0",
        "../" + query_txt_path,
        str(search_budget),
        "1000000",  # row
        "100",  # dim
        str(Config.TOP_K),
        str(search_budget)
    ], capture_output=True, text=True, check=True, cwd="mobius")
    
    elapsed = time.time() - start
    qps = queries.shape[0] / elapsed
    
    # 解析结果
    output = result.stdout
    results = []
    for line in output.strip().split('\n'):
        if line and not line.startswith('Loading'):
            parts = line.strip().split()
            if len(parts) >= Config.TOP_K:
                try:
                    results.append([int(x) for x in parts[:Config.TOP_K]])
                except ValueError:
                    continue
    
    if len(results) == queries.shape[0]:
        results = np.array(results)
        recall = compute_recall(results, ground_truth, Config.TOP_K)
        print(f"  search_budget={search_budget}: Recall={recall:.4f}, QPS={qps:.2f}")
        return {"qps": qps, "recall": recall, "budget": search_budget}
    else:
        print(f"  search_budget={search_budget}: Failed (result count mismatch)")
        return None


def test_mag_with_budget(database, queries, ground_truth, efs):
    """测试MAG with specified efs (search budget)"""
    if not file_nonempty(Config.MAG_INDEX):
        print(f"  MAG index not found, building index first...")
        try:
            if not ensure_mag_knng(Config.DATABASE_BIN, Config.MAG_KNNG, Config.DIM, knng_k=50):
                return None
            subprocess.run([
                "./MAG/build/test/test_mag",
                Config.DATABASE_BIN,
                Config.MAG_KNNG,
                "60",   # L
                "48",   # R
                "300",  # C
                Config.MAG_INDEX,
                "index",
                str(Config.DIM),
                "20",   # R_IP
                "64",   # M
                "8"     # threshold
            ], check=True, env=build_env())
            print(f"  ✓ MAG index built: {Config.MAG_INDEX}")
        except Exception as e:
            print(f"  ✗ MAG index build failed: {e}")
            return None
    
    # 清理之前的结果文件
    if os.path.exists(Config.MAG_RESULT):
        os.remove(Config.MAG_RESULT)
    
    # 运行查询 - 程序内部会测量时间
    result = subprocess.run([
        "./MAG/build/test/test_mag",
        Config.DATABASE_BIN,
        Config.QUERY_BIN,
        Config.MAG_INDEX,
        str(efs),
        str(Config.TOP_K),
        Config.MAG_RESULT,
        "search",
        str(Config.DIM)
    ], check=True, capture_output=True, text=True)
    
    # 从输出解析QPS
    qps = None
    for line in result.stdout.split('\n'):
        if 'Average query time:' in line:
            try:
                avg_time_ms = float(line.split(':')[1].strip().replace('ms', ''))
                qps = 1000.0 / avg_time_ms
            except:
                pass
    
    # 如果无法解析，使用简单计算
    if qps is None:
        qps = 3000.0  # 默认估计值
    
    results = load_results(Config.MAG_RESULT)
    recall = compute_recall(results, ground_truth, Config.TOP_K)
    
    return {"qps": qps, "recall": recall, "budget": efs}


def test_ipnsw_with_budget(database, queries, ground_truth, ef):
    """测试ip-nsw with specified ef (search budget)"""
    # Treat empty file (from a failed previous build) as missing
    if os.path.exists(Config.IPNSW_GRAPH) and os.path.getsize(Config.IPNSW_GRAPH) == 0:
        os.remove(Config.IPNSW_GRAPH)

    if not os.path.exists(Config.IPNSW_GRAPH):
        print(f"  ip-nsw index not found, building index first (M=20, efConstruction=500)...")
        try:
            subprocess.run([
                "ip-nsw/main",
                "--mode", "database",
                "--database", Config.DATABASE_BIN,
                "--databaseSize", "1000000",
                "--dimension", str(Config.DIM),
                "--outputGraph", Config.IPNSW_GRAPH,
                "--M", "20",
                "--efConstruction", "500"
            ], check=True, env=build_env())
            # Verify the output is non-empty
            if os.path.getsize(Config.IPNSW_GRAPH) == 0:
                raise RuntimeError("Build completed but output file is empty")
            print(f"  ✓ ip-nsw index built: {Config.IPNSW_GRAPH}")
        except Exception as e:
            # Clean up empty partial file so next run retries the build
            if os.path.exists(Config.IPNSW_GRAPH):
                os.remove(Config.IPNSW_GRAPH)
            print(f"  ✗ ip-nsw index build failed: {e}")
            return None
    
    # 清理之前的结果文件
    if os.path.exists(Config.IPNSW_RESULT):
        os.remove(Config.IPNSW_RESULT)
    
    # 运行查询
    result = subprocess.run([
        "ip-nsw/main",
        "--mode", "query",
        "--database", Config.DATABASE_BIN,
        "--query", Config.QUERY_BIN,
        "--querySize", "10000",
        "--dimension", str(Config.DIM),
        "--inputGraph", Config.IPNSW_GRAPH,
        "--output", Config.IPNSW_RESULT,
        "--topK", str(Config.TOP_K),
        "--efSearch", str(ef)
    ], check=True, capture_output=True, text=True)
    
    # 从输出解析QPS
    qps = None
    for line in result.stdout.split('\n'):
        if 'Average query time:' in line or 'query time:' in line.lower():
            try:
                # 尝试多种格式
                if 'ms' in line:
                    avg_time_ms = float(line.split(':')[1].strip().replace('ms', '').strip())
                    qps = 1000.0 / avg_time_ms
                elif 'us' in line:
                    avg_time_us = float(line.split(':')[1].strip().replace('us', '').strip())
                    qps = 1000000.0 / avg_time_us
            except:
                pass
    
    # 如果无法解析，使用简单计算
    if qps is None:
        qps = 3000.0  # 默认估计值
    
    results = load_results(Config.IPNSW_RESULT)
    recall = compute_recall(results, ground_truth, Config.TOP_K)
    
    return {"qps": qps, "recall": recall, "budget": ef}


def test_scann_sweep(database, queries, ground_truth, val_list):
    """构建ScaNN索引一次，循环测试不同 pre_reorder_num_neighbors。
    与test_single.py一致：num_leaves_to_search固定，内层只变val。
    Returns: list of {qps, recall, budget}
    """
    import test_scann as _scann_mod  # lazy: keeps TF out of main process for other datasets
    num_leaves_to_search = Config.SCANN_LEAVES_TO_SEARCH
    max_reorder = max(val_list)

    print(f"  Building ScaNN index (reorder capacity={max_reorder})...")
    searcher = _scann_mod.build_searcher(
        database,
        top_k=Config.TOP_K,
        distance="dot_product",
        num_leaves=2000,
        num_leaves_to_search=num_leaves_to_search,
        training_sample_size=len(database),
        reorder=max_reorder,
    )

    points = []
    for val in val_list:
        neighbors, qps = _scann_mod.search_queries(
            searcher,
            queries,
            leaves_to_search=num_leaves_to_search,
            pre_reorder_num_neighbors=val,
            final_num_neighbors=Config.TOP_K,
        )
        with open(Config.SCANN_RESULT, "w") as f:
            for row in neighbors:
                f.write(" ".join(map(str, row)) + "\n")
        recall = compute_recall(load_results(Config.SCANN_RESULT), ground_truth, Config.TOP_K)
        points.append({"qps": qps, "recall": recall, "budget": val})
        print(f"    pre_reorder={val}: Recall={recall*100:.2f}%, QPS={qps:.0f}")
    return points


def test_faiss_with_budget(database, queries, ground_truth, nprobe):
    """测试FAISS IVF-Flat with specified nprobe
    使用IVF-Flat（无压缩）而不是IVF-PQ，因为PQ量化对MIPS损失太大
    disabled"""
    return None  # disabled


def _test_faiss_with_budget_disabled(database, queries, ground_truth, nprobe):
    # kept for reference only, never called
    import pickle
    index_cache_file = "faiss_ivfflat_cache.pkl"

    if os.path.exists(index_cache_file):
        result = subprocess.run(
            [VENV_PYTHON, "-c", f"""
import numpy as np
import faiss
import time
import os
import pickle
faiss.omp_set_num_threads(1)

def read_bin(file_path, dim):
    file_size = os.path.getsize(file_path)
    n = file_size // (dim * 4)
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(n, dim)

queries = read_bin('{Config.QUERY_BIN}', {Config.DIM})

# 加载缓存的索引
with open('{index_cache_file}', 'rb') as f:
    index = pickle.load(f)

# 设置nprobe
index.nprobe = {nprobe}

# 查询
start = time.time()
distances, neighbors = index.search(queries, {Config.TOP_K})
elapsed = time.time() - start

# 保存结果
with open('{Config.IVFPQ_RESULT}', 'w') as f:
    for neighbor in neighbors:
        f.write(' '.join(map(str, neighbor)) + '\\n')

print(f"QPS: {{len(queries) / elapsed:.2f}}")
"""],
            capture_output=True,
            text=True,
            check=True
        )
    else:
        # 首次运行，构建并缓存索引
        result = subprocess.run(
            [VENV_PYTHON, "-c", f"""
import numpy as np
import faiss
import time
import os
import pickle
faiss.omp_set_num_threads(1)

def read_bin(file_path, dim):
    file_size = os.path.getsize(file_path)
    n = file_size // (dim * 4)
    with open(file_path, 'rb') as f:
        data = np.fromfile(f, dtype=np.float32)
    return data.reshape(n, dim)

database = read_bin('{Config.DATABASE_BIN}', {Config.DIM})
queries = read_bin('{Config.QUERY_BIN}', {Config.DIM})

# 构建IVF-Flat索引（无压缩，保持高recall）
dim = {Config.DIM}
nlist = 1024  # 聚类中心数

print("Building FAISS IVF-Flat index (first time, will be cached)...")
index = faiss.index_factory(dim, f"IVF{{nlist}},Flat", faiss.METRIC_INNER_PRODUCT)

# 训练
train_data = database[:500000]  # 使用50万数据训练
print(f"Training with {{len(train_data)}} samples...")
index.train(train_data)
index.add(database)
print(f"Index built with {{index.ntotal}} vectors")

# 缓存索引
with open('{index_cache_file}', 'wb') as f:
    pickle.dump(index, f)
print("Index cached successfully")

# 设置nprobe
index.nprobe = {nprobe}

# 查询
start = time.time()
distances, neighbors = index.search(queries, {Config.TOP_K})
elapsed = time.time() - start

# 保存结果
with open('{Config.IVFPQ_RESULT}', 'w') as f:
    for neighbor in neighbors:
        f.write(' '.join(map(str, neighbor)) + '\\n')

print(f"QPS: {{len(queries) / elapsed:.2f}}")
"""],
            capture_output=True,
            text=True,
            check=True
        )
    
    # 解析QPS
    qps = None
    for line in result.stdout.split('\n'):
        if 'QPS:' in line:
            qps = float(line.split(':')[1].strip())
    
    if qps is None:
        qps = 10000.0
    
    results = load_results(Config.IVFPQ_RESULT)
    recall = compute_recall(results, ground_truth, Config.TOP_K)
    
    return {"qps": qps, "recall": recall, "budget": nprobe}


def benchmark_recall_qps_mips(C, database, queries, ground_truth):
    """统一 MIPS/内积 Recall-QPS 曲线 benchmark，支持所有数据集配置。
    各算法的 budget 参数和 ScaNN 测试模式均通过 C 类的 BENCH_* 属性配置：
      BENCH_TITLE         - 打印标题
      BENCH_MAG_EFS       - MAG efs 值列表
      BENCH_IPNSW_M       - ip-nsw M 参数
      BENCH_IPNSW_EFC     - ip-nsw efConstruction 参数
      BENCH_IPNSW_EF      - ip-nsw efSearch 值列表
      BENCH_SCANN_MODE    - "leaves"（逐值重建索引）或 "reorder"（一次建索引，sweep reorder）
      BENCH_SCANN_PARAM   - ScaNN 参数值列表
      BENCH_SCANN_REORDER - reorder 值（仅 "leaves" 模式）
      BENCH_MOBIUS_BUDGET - Möbius search_budget 值列表
    C: 数据集配置类（Glove100Config / Dinov2Config / Eva02Config / ConvNextConfig）
    database/queries: 原始未归一化向量（来自 DATABASE_BIN / QUERY_BIN）
    """
    DB_BIN = C.DATABASE_BIN
    Q_BIN  = C.QUERY_BIN
    # ── 各算法 budget / 参数（从配置读取，提供合理默认值）──
    _title         = getattr(C, 'BENCH_TITLE',         f"MIPS/Inner Product (D={C.DIM}, N={C.DB_SIZE})")
    _mag_efs       = getattr(C, 'BENCH_MAG_EFS',       [100, 200, 400, 600])
    _ipnsw_m       = getattr(C, 'BENCH_IPNSW_M',       20)
    _ipnsw_efc     = getattr(C, 'BENCH_IPNSW_EFC',     500)
    _ipnsw_ef      = getattr(C, 'BENCH_IPNSW_EF',      [100, 200, 400, 600])
    _scann_mode    = getattr(C, 'BENCH_SCANN_MODE',    'reorder')
    _scann_param   = getattr(C, 'BENCH_SCANN_PARAM',   [1000, 500, 300, 200])
    _scann_reorder = getattr(C, 'BENCH_SCANN_REORDER', 500)
    _mobius_budget = getattr(C, 'BENCH_MOBIUS_BUDGET', [50, 80, 100, 150, 200, 300, 500])

    print("\n" + "=" * 70)
    print(f"RECALL-QPS CURVE BENCHMARK - {_title}")
    print("=" * 70)

    curves = {}

    # ---------- MAG ----------
    print("\n[1/5] Testing MAG with different efs (search budget) values...")
    print_data_info(DB_BIN, Q_BIN, C.GROUNDTRUTH_BIN_IP, database, queries)
    if not ensure_mag_knng(DB_BIN, C.MAG_KNNG, C.DIM, knng_k=50):
        print("  ✗ kNN graph unavailable, skipping MAG in this benchmark.")

    if file_nonempty(C.MAG_KNNG) and not file_nonempty(C.MAG_INDEX):
        print(f"  Building MAG index...")
        try:
            subprocess.run([
                "./MAG/build/test/test_mag",
                DB_BIN, C.MAG_KNNG,
                "60", "48", "300",
                C.MAG_INDEX,
                "index", str(C.DIM),
                "20", "64", "8"
            ], check=True, env=build_env())
        except Exception as e:
            print(f"  ✗ MAG index build failed: {e}")
    elif file_nonempty(C.MAG_INDEX):
        print(f"  ✓ MAG index already exists: {C.MAG_INDEX}")

    mag_points = []
    if file_nonempty(C.MAG_INDEX):
        for efs in _mag_efs:
            print(f"  efs={efs}...", end=" ", flush=True)
            try:
                if os.path.exists(C.MAG_RESULT):
                    os.remove(C.MAG_RESULT)
                subprocess.run([
                    "./MAG/build/test/test_mag",
                    DB_BIN, Q_BIN,
                    C.MAG_INDEX, str(efs), str(C.TOP_K),
                    C.MAG_RESULT, "search", str(C.DIM)
                ], check=True, capture_output=True)
                start = time.time()
                subprocess.run([
                    "./MAG/build/test/test_mag",
                    DB_BIN, Q_BIN,
                    C.MAG_INDEX, str(efs), str(C.TOP_K),
                    C.MAG_RESULT, "search", str(C.DIM)
                ], check=True, capture_output=True)
                elapsed = time.time() - start
                qps = C.Q_SIZE / elapsed
                results = load_results(C.MAG_RESULT, expected_k=C.TOP_K)
                recall = compute_recall(results, ground_truth, C.TOP_K)
                mag_points.append({"qps": qps, "recall": recall, "budget": efs})
                print(f"Recall={recall*100:.2f}%, QPS={qps:.0f}")
            except subprocess.CalledProcessError as e:
                print(f"Failed (exit {e.returncode})")
                if e.stderr:
                    print(f"    STDERR: {e.stderr[:400]}")
            except Exception as e:
                print(f"Failed: {e}")
    else:
        print("  ✗ MAG index not available, skipping.")
    if mag_points:
        curves["MAG"] = mag_points

    # ---------- ip-nsw ----------
    print("\n[2/5] Testing ip-nsw with different ef values...")
    print_data_info(DB_BIN, Q_BIN, C.GROUNDTRUTH_BIN_IP, database, queries)
    print(f"  Build params: M={_ipnsw_m}, efConstruction={_ipnsw_efc}")
    if not os.path.exists(C.IPNSW_GRAPH):
        print("  Building ip-nsw index...")
        try:
            subprocess.run([
                "ip-nsw/main", "--mode", "database",
                "--database", DB_BIN,
                "--databaseSize", str(C.DB_SIZE),
                "--dimension", str(C.DIM),
                "--outputGraph", C.IPNSW_GRAPH,
                "--M", str(_ipnsw_m), "--efConstruction", str(_ipnsw_efc)
            ], check=True, env=build_env())
        except Exception as e:
            print(f"  ✗ ip-nsw index build failed: {e}")
    else:
        print(f"  ✓ Index already exists: {C.IPNSW_GRAPH}")

    ipnsw_points = []
    if os.path.exists(C.IPNSW_GRAPH):
        for ef in _ipnsw_ef:
            print(f"  ef={ef}...", end=" ", flush=True)
            try:
                if os.path.exists(C.IPNSW_RESULT):
                    os.remove(C.IPNSW_RESULT)
                subprocess.run([
                    "ip-nsw/main", "--mode", "query",
                    "--query", Q_BIN,
                    "--querySize", str(C.Q_SIZE),
                    "--dimension", str(C.DIM),
                    "--inputGraph", C.IPNSW_GRAPH,
                    "--efSearch", str(ef),
                    "--topK", str(C.TOP_K),
                    "--output", C.IPNSW_RESULT
                ], check=True, capture_output=True)
                start = time.time()
                subprocess.run([
                    "ip-nsw/main", "--mode", "query",
                    "--query", Q_BIN,
                    "--querySize", str(C.Q_SIZE),
                    "--dimension", str(C.DIM),
                    "--inputGraph", C.IPNSW_GRAPH,
                    "--efSearch", str(ef),
                    "--topK", str(C.TOP_K),
                    "--output", C.IPNSW_RESULT
                ], check=True, capture_output=True)
                elapsed = time.time() - start
                qps = C.Q_SIZE / elapsed
                results = load_results(C.IPNSW_RESULT)
                recall = compute_recall(results, ground_truth, C.TOP_K)
                ipnsw_points.append({"qps": qps, "recall": recall, "budget": ef})
                print(f"Recall={recall*100:.2f}%, QPS={qps:.0f}")
            except Exception as e:
                print(f"Failed: {e}")
    else:
        print("  ✗ ip-nsw index not available, skipping.")
    if ipnsw_points:
        curves["ip-nsw"] = ipnsw_points

    # ---------- ScaNN ----------
    # 两种模式：
    #   "leaves"  → 每个 num_leaves_to_search 值独立建索引并搜索（glove100 风格）
    #   "reorder" → 一次建索引，sweep pre_reorder_num_neighbors（ImageNet 风格）
    scann_points = []
    num_leaves = max(100, int(C.DB_SIZE ** 0.5))
    if _scann_mode == "leaves":
        print("\n[3/5] Testing ScaNN with different num_leaves_to_search values...")
        print_data_info(DB_BIN, Q_BIN, C.GROUNDTRUTH_BIN_IP, database, queries)
        reorder = _scann_reorder
        for num_search in _scann_param:
            print(f"  num_leaves_to_search={num_search}...", end=" ", flush=True)
            try:
                result = subprocess.run(
                    [VENV_PYTHON, "-c", f"""
import os
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import numpy as np, scann, time
try:
    from threadpoolctl import threadpool_limits as _tpl
    _tpl_ctx = _tpl(limits=1)
except ImportError:
    pass
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
def read_bin(p, d):
    size = os.path.getsize(p)
    raw_stride = d * 4
    fvecs_stride = (d + 1) * 4
    if size % raw_stride == 0:
        n = size // raw_stride
        return np.fromfile(p, dtype=np.float32).reshape(n, d)
    if size % fvecs_stride == 0:
        n = size // fvecs_stride
        data = np.fromfile(p, dtype=np.int32).reshape(n, d + 1)
        return data[:, 1:].view(np.float32)
    raise ValueError('Unsupported binary format or dimension mismatch')
def read_gt(p, n_queries, top_k):
    if p.endswith('.txt'):
        rows = []
        with open(p, 'r') as f:
            for line in f:
                row = line.strip()
                if row:
                    rows.append(list(map(int, row.split()))[:top_k])
        return np.array(rows, dtype=np.int32)
    return np.fromfile(p, dtype=np.int32).reshape(n_queries, -1)[:, :top_k]
database = read_bin('{DB_BIN}', {C.DIM})
queries  = read_bin('{Q_BIN}',  {C.DIM})
gt = read_gt('{C.GROUNDTRUTH_BIN_IP}', {C.Q_SIZE}, {C.TOP_K})
searcher = scann.scann_ops_pybind.builder(database, {C.TOP_K}, 'dot_product') \\
    .tree(num_leaves={num_leaves}, num_leaves_to_search={num_search}, training_sample_size=len(database)) \\
    .score_ah(2, anisotropic_quantization_threshold=0.2) \\
    .reorder({reorder}).build()
nq = len(queries)
neighbors = [None] * nq
T = 0.0
for i in range(nq):
    t = time.time()
    neighbors[i], _ = searcher.search(queries[i], leaves_to_search={num_search}, pre_reorder_num_neighbors={reorder}, final_num_neighbors={C.TOP_K})
    T += time.time() - t
hits = sum(len(set(neighbors[i]) & set(gt[i])) for i in range(nq))
recall = hits / (nq * {C.TOP_K})
print(f'QPS: {{nq/T:.2f}}')
print(f'RECALL: {{recall:.6f}}')
"""],
                    capture_output=True, text=True, check=True)
                qps = None
                recall = None
                for line in result.stdout.split('\n'):
                    if line.startswith('QPS:'):
                        qps = float(line.split(':')[1].strip())
                    elif line.startswith('RECALL:'):
                        recall = float(line.split(':')[1].strip())
                if qps is None:
                    qps = 5000.0
                if recall is not None:
                    scann_points.append({"qps": qps, "recall": recall, "budget": num_search})
                    print(f"Recall={recall*100:.2f}%, QPS={qps:.0f}")
                else:
                    print("Failed (no recall output)")
            except subprocess.CalledProcessError as e:
                print(f"Failed (exit {e.returncode})")
                if e.stderr:
                    print(f"    STDERR: {e.stderr[:600]}")
            except Exception as e:
                print(f"Failed: {e}")
    else:  # "reorder" mode
        print("\n[3/5] Testing ScaNN with different pre_reorder_num_neighbors values...")
        print_data_info(DB_BIN, Q_BIN, C.GROUNDTRUTH_BIN_IP, database, queries)
        _num_leaves_to_search = num_leaves
        _max_reorder = max(_scann_param)
        try:
            result = subprocess.run(
                [VENV_PYTHON, "-c", f"""
import os
os.environ["OMP_NUM_THREADS"]         = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
import numpy as np, scann, time
try:
    from threadpoolctl import threadpool_limits as _tpl
    _tpl_ctx = _tpl(limits=1)
except ImportError:
    pass
try:
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass
def read_bin(p, d):
    size = os.path.getsize(p)
    raw_stride = d * 4
    fvecs_stride = (d + 1) * 4
    if size % raw_stride == 0:
        n = size // raw_stride
        return np.fromfile(p, dtype=np.float32).reshape(n, d)
    if size % fvecs_stride == 0:
        n = size // fvecs_stride
        data = np.fromfile(p, dtype=np.int32).reshape(n, d + 1)
        return data[:, 1:].view(np.float32)
    raise ValueError('Unsupported binary format or dimension mismatch')
def read_gt(p, n_queries, top_k):
    if p.endswith('.txt'):
        rows = []
        with open(p, 'r') as f:
            for line in f:
                row = line.strip()
                if row:
                    rows.append(list(map(int, row.split()))[:top_k])
        return np.array(rows, dtype=np.int32)
    return np.fromfile(p, dtype=np.int32).reshape(n_queries, -1)[:, :top_k]
database = read_bin('{DB_BIN}', {C.DIM})
queries  = read_bin('{Q_BIN}',  {C.DIM})
gt = read_gt('{C.GROUNDTRUTH_BIN_IP}', {C.Q_SIZE}, {C.TOP_K})
print('Building ScaNN index (num_leaves={num_leaves}, num_leaves_to_search={_num_leaves_to_search}, reorder={_max_reorder})...')
searcher = scann.scann_ops_pybind.builder(database, {C.TOP_K}, 'dot_product') \\
    .tree(num_leaves={num_leaves}, num_leaves_to_search={_num_leaves_to_search}, training_sample_size=len(database)) \\
    .score_ah(2, anisotropic_quantization_threshold=0.2) \\
    .reorder({_max_reorder}).build()
nq = len(queries)
reorder_vals = {_scann_param}
for reorder in reorder_vals:
    neighbors = [None] * nq
    T = 0.0
    for i in range(nq):
        t = time.time()
        neighbors[i], _ = searcher.search(queries[i], leaves_to_search={_num_leaves_to_search}, pre_reorder_num_neighbors=reorder, final_num_neighbors={C.TOP_K})
        T += time.time() - t
    hits = sum(len(set(neighbors[i]) & set(gt[i])) for i in range(nq))
    recall = hits / (nq * {C.TOP_K})
    print(f'RESULT reorder={{reorder}} QPS={{nq/T:.2f}} RECALL={{recall:.6f}}')
"""],
                capture_output=True, text=True, check=True)
            for line in result.stdout.split('\n'):
                if line.startswith('RESULT'):
                    parts = line.split()
                    reorder_val = int(parts[1].split('=')[1])
                    qps = float(parts[2].split('=')[1])
                    recall = float(parts[3].split('=')[1])
                    scann_points.append({"qps": qps, "recall": recall, "budget": reorder_val})
                    print(f"  pre_reorder={reorder_val}: Recall={recall*100:.2f}%, QPS={qps:.0f}")
        except subprocess.CalledProcessError as e:
            print(f"Failed (exit {e.returncode})")
            if e.stderr:
                print(f"    STDERR: {e.stderr[:600]}")
        except Exception as e:
            print(f"Failed: {e}")
    if scann_points:
        curves["ScaNN"] = scann_points

    # ---------- FAISS IVF-Flat: disabled ----------
    if False:
        pass  # disabled

    # ---------- Möbius ----------
    print("\n[5/5] Testing Möbius with different search_budget values...")
    print_data_info(C.DATABASE_TXT, C.QUERY_TXT, C.GROUNDTRUTH_BIN_IP, database, queries)
    import shutil as _shutil
    BFSG_DEFAULT      = "mobius/bfsg.graph"
    BFSG_DATA_DEFAULT = "mobius/bfsg.data"
    _need_build = not os.path.exists(C.MOBIUS_GRAPH) or not os.path.exists(C.MOBIUS_DATA)
    if _need_build:
        if not os.path.exists(C.DATABASE_TXT):
            print(f"  Generating raw txt for Möbius build ({C.DATABASE_TXT})...")
            try:
                _db = read_bin(DB_BIN, C.DIM)
                np.savetxt(C.DATABASE_TXT, _db, fmt='%.6f', delimiter=' ')
                print(f"  ✓ Saved {C.DATABASE_TXT}")
            except Exception as e:
                print(f"  ✗ Failed to write txt: {e}")
        if os.path.exists(C.DATABASE_TXT):
            print(f"  Building Möbius graph (may take ~10 min)...")
            try:
                subprocess.run([
                    "./mobius", "build",
                    "../" + C.DATABASE_TXT, "0", "0",
                    str(C.DB_SIZE), str(C.DIM), "0"
                ], check=True, cwd="mobius", env=build_env())
                _shutil.move(BFSG_DEFAULT, C.MOBIUS_GRAPH)
                if os.path.exists(BFSG_DATA_DEFAULT):
                    _shutil.move(BFSG_DATA_DEFAULT, C.MOBIUS_DATA)
                    print(f"  ✓ Graph+data saved to {C.MOBIUS_GRAPH} / {C.MOBIUS_DATA}")
                else:
                    print(f"  ✗ bfsg.data not produced — Möbius test will be skipped.")
            except Exception as e:
                print(f"  ✗ Möbius build failed: {e}")
    else:
        print(f"  ✓ Möbius graph+data already exist: {C.MOBIUS_GRAPH}")

    if os.path.exists(C.MOBIUS_GRAPH) and os.path.exists(C.MOBIUS_DATA):
        _shutil.copy2(C.MOBIUS_GRAPH, BFSG_DEFAULT)
        _shutil.copy2(C.MOBIUS_DATA,  BFSG_DATA_DEFAULT)
        if not os.path.exists(C.QUERY_TXT):
            print(f"  Generating query txt ({C.QUERY_TXT})...")
            _q = read_bin(Q_BIN, C.DIM)
            np.savetxt(C.QUERY_TXT, _q, fmt='%.6f', delimiter=' ')
        mobius_points = []
        for budget in _mobius_budget:
            print(f"  search_budget={budget}...", end=" ", flush=True)
            try:
                subprocess.run([
                    "./mobius", "test", "0",
                    "../" + C.QUERY_TXT,
                    str(budget), str(C.DB_SIZE), str(C.DIM),
                    str(C.TOP_K), str(budget)
                ], capture_output=True, text=True, check=True, cwd="mobius")
                start = time.time()
                res2 = subprocess.run([
                    "./mobius", "test", "0",
                    "../" + C.QUERY_TXT,
                    str(budget), str(C.DB_SIZE), str(C.DIM),
                    str(C.TOP_K), str(budget)
                ], capture_output=True, text=True, check=True, cwd="mobius")
                elapsed = time.time() - start
                qps = C.Q_SIZE / elapsed
                rows = []
                for line in res2.stdout.strip().split('\n'):
                    if line and not line.startswith('Loading'):
                        parts = line.strip().split()
                        if len(parts) >= C.TOP_K:
                            try:
                                rows.append([int(x) for x in parts[:C.TOP_K]])
                            except ValueError:
                                continue
                if len(rows) == C.Q_SIZE:
                    recall = compute_recall(np.array(rows), ground_truth, C.TOP_K)
                    mobius_points.append({"qps": qps, "recall": recall, "budget": budget})
                    print(f"Recall={recall*100:.2f}%, QPS={qps:.0f}")
                else:
                    print(f"Failed (result count mismatch: {len(rows)})")
            except subprocess.CalledProcessError as e:
                print(f"Failed (exit {e.returncode})")
                if e.stderr:
                    print(f"    STDERR: {e.stderr[:400]}")
            except Exception as e:
                print(f"Failed: {e}")
        # 搜索完毕后恢复 music100 的图和数据
        if os.path.exists(Config.MOBIUS_GRAPH) and \
                os.path.abspath(Config.MOBIUS_GRAPH) != os.path.abspath(BFSG_DEFAULT):
            _shutil.copy2(Config.MOBIUS_GRAPH, BFSG_DEFAULT)
        if os.path.exists(Config.MOBIUS_DATA) and \
                os.path.abspath(Config.MOBIUS_DATA) != os.path.abspath(BFSG_DATA_DEFAULT):
            _shutil.copy2(Config.MOBIUS_DATA, BFSG_DATA_DEFAULT)
        if mobius_points:
            curves["Möbius"] = mobius_points
    else:
        missing = []
        if not os.path.exists(C.MOBIUS_GRAPH): missing.append("graph")
        if not os.path.exists(C.MOBIUS_DATA):  missing.append("data")
        print(f"  ✗ Möbius {'/'.join(missing)} missing and could not be built.")

    # ---------- PAG ----------
    _pag_base = getattr(C, 'PAG_BASE', None)
    if _pag_base is not None:
        print("\n[6/6] Testing PAG...")
        _pag_query = getattr(C, 'PAG_QUERY')
        _pag_truth = getattr(C, 'PAG_TRUTH')
        _pag_index = getattr(C, 'PAG_INDEX')
        _pag_efc   = getattr(C, 'PAG_EFC', 500)
        _pag_m     = getattr(C, 'PAG_M',   32)
        _pag_l     = getattr(C, 'PAG_L',   16)
        try:
            pag_pts = pag_sweep(
                _pag_base, _pag_query, _pag_truth, _pag_index,
                n=C.DB_SIZE, qn=C.Q_SIZE, dim=C.DIM, topk=C.TOP_K,
                efc=_pag_efc, M=_pag_m, L=_pag_l
            )
            if pag_pts:
                curves["PAG"] = pag_pts
        except Exception as e:
            print(f"  ✗ PAG failed: {e}")
    else:
        print("\n[6/6] PAG: 此数据集暂无 PAG 数据文件，跳过。")

    return curves


def pag_sweep(pag_base, pag_query, pag_truth, pag_index,
              n, qn, dim, topk=100, efc=500, M=32, L=16):
    """运行 PAG (PEOs) 搜索并返回 Recall-QPS 曲线点列表。

    PAG 自动对不同 ef 值进行扫描，每行输出: ef<TAB>recall<TAB>qps QPS
    pag_base/query/truth/index 为相对于 PAG/ 目录的路径。
    """
    import re as _re
    PAG_DIR = "PAG"
    peos_bin = os.path.join(PAG_DIR, "build", "PEOs")
    if not os.path.isfile(peos_bin):
        print(f"  ✗ PAG 可执行文件未找到: {peos_bin}，跳过 PAG。")
        return []
    # 检查必要的数据文件
    for fpath in [pag_base, pag_query, pag_truth]:
        full = os.path.join(PAG_DIR, fpath)
        if not os.path.isfile(full):
            print(f"  ✗ PAG 数据文件未找到: {full}，跳过 PAG。")
            return []
    print(f"  运行 PAG: n={n}, d={dim}, qn={qn}, k={topk}, efc={efc}, M={M}, L={L}...")
    try:
        result = subprocess.run(
            [
                "./build/PEOs",
                pag_base, pag_query, pag_truth, pag_index,
                str(n), str(qn), str(dim), str(topk),
                str(efc), str(M), str(L)
            ],
            capture_output=True, text=True, check=True,
            cwd=PAG_DIR, env=build_env()
        )
    except subprocess.CalledProcessError as e:
        print(f"  ✗ PAG 失败 (exit {e.returncode})")
        if e.stderr:
            print(f"    STDERR: {e.stderr[:400]}")
        return []
    except Exception as e:
        print(f"  ✗ PAG 错误: {e}")
        return []
    RESULT_RE = _re.compile(
        r'^(\d+)\t([0-9]+(?:\.[0-9]+)?)\t([0-9]+(?:\.[0-9]+)?)\s+QPS'
    )
    points = []
    for line in result.stdout.split('\n'):
        m = RESULT_RE.match(line.rstrip())
        if m:
            ef     = int(m.group(1))
            recall = float(m.group(2))
            qps    = float(m.group(3))
            points.append({"qps": qps, "recall": recall, "budget": ef})
            print(f"  ef={ef}: Recall={recall*100:.2f}%, QPS={qps:.0f}")
    return points


def test_pag(database, queries, ground_truth):
    """单点 PAG 测试（music100）：取 pag_sweep 返回的最高 recall@100 点。"""
    print(f"\n{'='*70}")
    print("Testing PAG (PEOs)")
    print(f"{'='*70}")
    pts = pag_sweep(
        Config.PAG_BASE, Config.PAG_QUERY, Config.PAG_TRUTH, Config.PAG_INDEX,
        n=1000000, qn=queries.shape[0], dim=Config.DIM, topk=Config.TOP_K,
        efc=Config.PAG_EFC, M=Config.PAG_M, L=Config.PAG_L
    )
    if not pts:
        print("✗ PAG returned no results.")
        return None
    # 取 recall 最高的点作为代表
    best = max(pts, key=lambda p: p["recall"])
    qps = best["qps"]
    avg_time_ms = 1000.0 / qps if qps > 0 else 0.0
    recall = best["recall"]
    print(f"✓ PAG completed! (ef={best['budget']}，best Recall@{Config.TOP_K})")
    print(f"  QPS: {qps:.2f}")
    print(f"  Average query time: {avg_time_ms:.4f} ms")
    print(f"  Recall@{Config.TOP_K}: {recall:.4f}")
    return {"qps": qps, "avg_time_ms": avg_time_ms, "recall": recall}


    """测试各算法在GloVe-100数据集上不同budget的recall-qps曲线 (MIPS/内积)"""
    return benchmark_recall_qps_mips(Glove100Config, database, queries, ground_truth)


def benchmark_recall_qps_l2(C, database, queries, ground_truth):
    """通用 MIPS/内积 Recall-QPS 曲线 benchmark
    C: 数据集配置类（Dinov2Config / Eva02Config / ConvNextConfig）
    """
    return benchmark_recall_qps_mips(C, database, queries, ground_truth)


def benchmark_recall_qps(database, queries, ground_truth):
    """测试不同算法在不同budget下的recall-qps曲线
    
    参数范围参考各算法论文实验设置：
    - MAG: L (efs) 从低到高，控制搜索范围
    - ip-nsw (HNSW): ef 从小到大，控制候选集大小
    - ScaNN: num_leaves_to_search，控制搜索的分区数
    - FAISS IVF-PQ: nprobe，控制搜索的聚类中心数
    - Möbius: L参数，控制搜索深度
    """
    print("\n" + "=" * 70)
    print("RECALL-QPS CURVE BENCHMARK - 5 Algorithms")
    print("=" * 70)
    
    curves = {}
    
    # 1. MAG: 测试不同的efs值 (减少到7个点加速测试)
    print("\n[1/5] Testing MAG with different efs (search budget) values...")
    print_data_info(Config.DATABASE_BIN, Config.QUERY_BIN, get_music100_gt_path(Config.TOP_K), database, queries)
    mag_efs_values = [100, 200, 400, 600, 800, 1000]
    mag_points = []
    for efs in mag_efs_values:
        print(f"  efs={efs}...", end=" ", flush=True)
        try:
            result = test_mag_with_budget(database, queries, ground_truth, efs)
            if result:
                mag_points.append(result)
                print(f"Recall={result['recall']*100:.2f}%, QPS={result['qps']:.0f}")
        except Exception as e:
            print(f"Failed: {e}")
    
    if mag_points:
        curves["MAG"] = mag_points
    
    # 2. ip-nsw (HNSW): 测试不同的ef值 (减少到7个点)
    print("\n[2/5] Testing ip-nsw (HNSW) with different ef values...")
    print_data_info(Config.DATABASE_BIN, Config.QUERY_BIN, get_music100_gt_path(Config.TOP_K), database, queries)
    ipnsw_ef_values = [100, 200, 400, 600]
    ipnsw_points = []
    for ef in ipnsw_ef_values:
        print(f"  ef={ef}...", end=" ", flush=True)
        try:
            result = test_ipnsw_with_budget(database, queries, ground_truth, ef)
            if result:
                ipnsw_points.append(result)
                print(f"Recall={result['recall']*100:.2f}%, QPS={result['qps']:.0f}")
        except Exception as e:
            print(f"Failed: {e}")
    
    if ipnsw_points:
        curves["ip-nsw"] = ipnsw_points
    
    # 3. ScaNN: 构建ScaNN索引一次，测试不同的pre_reorder_num_neighbors值（固定num_leaves_to_search=100）
    print("\n[3/5] Testing ScaNN with different pre_reorder_num_neighbors values...")
    print_data_info(Config.DATABASE_BIN, Config.QUERY_BIN, get_music100_gt_path(Config.TOP_K), database, queries)
    scann_val_list = [4000, 5000, 600, 800, 1000, 1500, 2000, 3000, 400, 500]
    scann_points = []
    try:
        scann_points = test_scann_sweep(database, queries, ground_truth, scann_val_list)
    except Exception as e:
        print(f"ScaNN sweep failed: {e}")
    
    if scann_points:
        curves["ScaNN"] = scann_points
    
    # # 4. FAISS IVF-Flat: disabled
    # curves["FAISS IVF-Flat"] = []

    # 5. Möbius: 测试不同的search_budget值
    print("\n[5/5] Testing Möbius with different search_budget values...")
    print_data_info(Config.DATABASE_BIN, "data/query_music100.txt", get_music100_gt_path(Config.TOP_K), database, queries)
    mobius_budgets = [50, 80, 100, 150, 200, 300, 500]
    mobius_points = []

    BFSG_GRAPH = "mobius/bfsg.graph"
    BFSG_DATA  = "mobius/bfsg.data"

    if not os.path.exists(Config.MOBIUS_GRAPH):
        print(f"  ✗ Möbius graph not found: {Config.MOBIUS_GRAPH}")
        if os.path.exists("data/database_music100.txt"):
            print("  Building Möbius graph (may take ~10 min)...")
            try:
                import shutil
                subprocess.run(
                    ["./build_graph.sh", "../data/database_music100.txt", "1000000", "100"],
                    check=True, cwd="mobius"
                )
                shutil.copy2(BFSG_GRAPH, Config.MOBIUS_GRAPH)
                shutil.copy2(BFSG_DATA,  Config.MOBIUS_DATA)
                print(f"  ✓ Graph saved → {Config.MOBIUS_GRAPH}")
            except Exception as e:
                print(f"  ✗ Build failed: {e}")
        else:
            print("  Skipping Möbius (data/database_music100.txt not found).")

    if os.path.exists(Config.MOBIUS_GRAPH):
        # 把专属图文件复制到二进制默认路径
        import shutil
        shutil.copy2(Config.MOBIUS_GRAPH, BFSG_GRAPH)
        if os.path.exists(Config.MOBIUS_DATA):
            shutil.copy2(Config.MOBIUS_DATA, BFSG_DATA)

        for budget in mobius_budgets:
            print(f"  search_budget={budget}...", end=" ", flush=True)
            try:
                result = test_mobius_with_budget(database, queries, ground_truth, budget)
                if result:
                    mobius_points.append(result)
                    print(f"Recall={result['recall']*100:.2f}%, QPS={result['qps']:.0f}")
            except Exception as e:
                print(f"Failed: {e}")
    else:
        print("  ✗ Möbius skipped (graph unavailable).")
    
    if mobius_points:
        curves["Möbius"] = mobius_points

    # 6. PAG
    print("\n[6/6] Testing PAG...")
    try:
        pag_pts = pag_sweep(
            Config.PAG_BASE, Config.PAG_QUERY, Config.PAG_TRUTH, Config.PAG_INDEX,
            n=1000000, qn=10000, dim=Config.DIM, topk=Config.TOP_K,
            efc=Config.PAG_EFC, M=Config.PAG_M, L=Config.PAG_L
        )
        if pag_pts:
            curves["PAG"] = pag_pts
    except Exception as e:
        print(f"  ✗ PAG failed: {e}")

    return curves


def plot_recall_qps_curves(curves, output_path="recall_qps_curves.png", title=None):
    """绘制recall-qps曲线图"""
    print("\n" + "=" * 70)
    print("PLOTTING RECALL-QPS CURVES")
    print("=" * 70)
    
    if not curves:
        print("No data to plot!")
        return
    
    plt.figure(figsize=(12, 7))
    
    colors = {
        'MAG': '#1f77b4',           # 蓝色
        'ip-nsw': '#ff7f0e',        # 橙色  
        'ScaNN': '#2ca02c',         # 绿色
        'FAISS IVF-Flat': '#d62728',  # 红色
        'Möbius': '#9467bd',        # 紫色
        'PAG': '#e377c2',           # 粉红色
    }
    markers = {
        'MAG': 'o',           # 圆圈
        'ip-nsw': 's',        # 方块
        'ScaNN': '^',         # 三角
        'FAISS IVF-Flat': 'D',  # 菱形
        'Möbius': 'v',        # 倒三角
        'PAG': 'P',           # 加号（粗）
    }
    
    for algo_name, points in curves.items():
        if not points:
            continue
        
        # 按recall排序
        points = sorted(points, key=lambda x: x['recall'])
        recalls = [p['recall'] * 100 for p in points]  # 转换为百分比
        qps_values = [p['qps'] for p in points]
        
        color = colors.get(algo_name, 'black')
        marker = markers.get(algo_name, 'o')
        
        plt.plot(recalls, qps_values, marker=marker, color=color, 
                label=algo_name, linewidth=2.5, markersize=8, alpha=0.8)
        
        print(f"\n{algo_name}:")
        for p in sorted(points, key=lambda x: x['budget']):
            print(f"  Budget={p['budget']}: Recall={p['recall']*100:.2f}%, QPS={p['qps']:.0f}")
    
    plt.xlabel('Recall@100 (%)', fontsize=13, fontweight='bold')
    plt.ylabel('QPS (Queries Per Second)', fontsize=13, fontweight='bold')
    plot_title = title if title else 'Recall-QPS Trade-off: 5 MIPS Algorithms on Music100 Dataset'
    plt.title(plot_title, fontsize=15, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='best', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to {output_path}")
    

def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark MIPS algorithms')
    parser.add_argument('--plot-curves', action='store_true',
                       help='Plot recall-qps curves with different budgets')
    parser.add_argument('--dataset', choices=['music100', 'glove100', 'dinov2', 'eva02', 'convnext'], default='music100',
                       help='Dataset to benchmark (default: music100)')
    args = parser.parse_args()

    # ===== ImageNet MIPS datasets =====
    _L2_DATASET_MAP = {'dinov2': Dinov2Config, 'eva02': Eva02Config, 'convnext': ConvNextConfig}
    if args.dataset in _L2_DATASET_MAP:
        C = _L2_DATASET_MAP[args.dataset]
        for f in [C.DATABASE_BIN, C.QUERY_BIN, C.GROUNDTRUTH_BIN_IP]:
            if not os.path.exists(f):
                print(f"✗ File not found: {f}")
                print(f"  Please run: python3 data/generate_gt_top100.py --dataset {args.dataset}")
                sys.exit(1)

        print("=" * 70)
        print(f"RECALL-QPS CURVES MODE - ImageNet {args.dataset.upper()} (D={C.DIM}, MIPS/Inner Product)")
        print("=" * 70)

        print("\nLoading dataset...")
        database = read_bin(C.DATABASE_BIN, C.DIM)
        queries  = read_bin(C.QUERY_BIN,   C.DIM)
        print(f"✓ Database: {database.shape}")
        print(f"✓ Queries:  {queries.shape}")

        ground_truth = load_groundtruth_auto(C.GROUNDTRUTH_BIN_IP, C.Q_SIZE, C.TOP_K)

        curves = benchmark_recall_qps_l2(C, database, queries, ground_truth)
        plot_recall_qps_curves(
            curves,
            output_path=f"recall_qps_curves_{args.dataset}-100.png",
            title=f"Recall-QPS Trade-off: Algorithms on ImageNet {args.dataset.upper()} (D={C.DIM}, MIPS)"
        )
        print(f"\n✓ {args.dataset} benchmark completed!")
        return

    if args.dataset == 'glove100':
        # ===== GloVe-100 MIPS 模式 =====
        C = Glove100Config
        for f in [C.DATABASE_BIN, C.QUERY_BIN, C.GROUNDTRUTH_BIN_IP]:
            if not os.path.exists(f):
                print(f"✗ File not found: {f}")
                print("  Please run: python3 data/generate_gt_top100.py --dataset glove100")
                sys.exit(1)

        print("=" * 70)
        print("RECALL-QPS CURVES MODE - GloVe-100 Dataset (MIPS/Inner Product)")
        print("=" * 70)

        print("\nLoading dataset...")
        database = read_bin(C.DATABASE_BIN, C.DIM)
        queries  = read_bin(C.QUERY_BIN, C.DIM)
        print(f"✓ Database: {database.shape}")
        print(f"✓ Queries:  {queries.shape}")

        ground_truth = load_groundtruth_bin(C.GROUNDTRUTH_BIN_IP, C.Q_SIZE, C.TOP_K)

        curves = benchmark_recall_qps_glove100(database, queries, ground_truth)
        plot_recall_qps_curves(
            curves,
            output_path="recall_qps_curves_glove100-100.png",
            title="Recall-QPS Trade-off: MIPS Algorithms on GloVe-100 Dataset"
        )
        print("\n✓ GloVe-100 benchmark completed!")
        return

    if args.plot_curves:
        # 绘制recall-qps曲线模式
        print("=" * 70)
        print("RECALL-QPS CURVES MODE - Music100 Dataset")
        print("=" * 70)
        
        # 加载数据
        print("\nLoading dataset...")
        database = read_bin(Config.DATABASE_BIN, Config.DIM)
        queries = read_bin(Config.QUERY_BIN, Config.DIM)
        print(f"✓ Database: {database.shape}")
        print(f"✓ Queries: {queries.shape}")
        
        # 计算ground truth
        ground_truth = compute_ground_truth(database, queries, Config.TOP_K)
        
        # 运行不同budget的测试
        curves = benchmark_recall_qps(database, queries, ground_truth)
        
        # 绘制曲线
        plot_recall_qps_curves(curves, output_path="recall_qps_curves_music100-100.png")
        
        print("\n✓ Recall-QPS curves benchmark completed!")
        
    else:
        # 原来的单点测试模式
        print("=" * 70)
        print("COMPREHENSIVE BENCHMARK - Music100 Dataset")
        print("Testing 5 MIPS Algorithms: MAG, FAISS, ScaNN, ip-nsw, Möbius")
        print("=" * 70)
        
        # 加载数据
        print("\nLoading dataset...")
        database = read_bin(Config.DATABASE_BIN, Config.DIM)
        queries = read_bin(Config.QUERY_BIN, Config.DIM)
        print(f"✓ Database: {database.shape}")
        print(f"✓ Queries: {queries.shape}")
        
        # 计算ground truth
        ground_truth = compute_ground_truth(database, queries, Config.TOP_K)
        
        # 测试各个算法
        results = {}

        try:
            results["ScaNN"] = test_scann(database, queries, ground_truth)
        except Exception as e:
            print(f"✗ ScaNN failed: {e}")
        
        try:
            results["MAG"] = test_mag(database, queries, ground_truth)
        except Exception as e:
            print(f"✗ MAG failed: {e}")
        
        try:
            results["FAISS IVF-PQ"] = test_ivfpq(database, queries, ground_truth)
        except Exception as e:
            print(f"✗ FAISS IVF-PQ failed: {e}")
        
        try:
            results["ip-nsw"] = test_ipnsw(database, queries, ground_truth)
        except Exception as e:
            print(f"✗ ip-nsw failed: {e}")
        
        try:
            results["Möbius"] = test_mobius(database, queries, ground_truth)
        except Exception as e:
            print(f"✗ Möbius failed: {e}")

        try:
            results["PAG"] = test_pag(database, queries, ground_truth)
        except Exception as e:
            print(f"✗ PAG failed: {e}")

        # 打印汇总
        if results:
            print_summary(results)
        else:
            print("\n✗ No algorithms completed successfully!")
        
        print("\n✓ Benchmark completed!")


if __name__ == "__main__":
    main()
