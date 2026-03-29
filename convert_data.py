#!/usr/bin/env python3
"""
多格式数据转换工具 (hdf5 / bin / txt / fvecs / ivecs)
支持进度条显示和数据校验
适用于 ANN-benchmarks HDF5、ip-nsw bin、Möbius txt、GIST/SIFT fvecs/ivecs
"""
import struct
import sys
import os
import random
import glob
import numpy as np
from tqdm import tqdm
try:
    import h5py
except ImportError:
    h5py = None
try:
    import pandas as pd
except ImportError:
    pd = None


# ─────────────────────────────────────────────────────────
# fvecs / ivecs 读写工具
# ─────────────────────────────────────────────────────────

def read_fvecs(path):
    """
    读取 fvecs 文件，返回 numpy 数组 (n_rows, dim)，dtype=float32
    fvecs 格式：每条向量前有 4 字节 int32 表示维度
    """
    print(f"📖 读取 fvecs: {path}")
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        # 读第一条向量获取维度
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            raise ValueError("文件为空或格式错误")
        dim = struct.unpack('i', dim_bytes)[0]

    vec_size = 4 + dim * 4          # 每条向量字节数
    if file_size % vec_size != 0:
        raise ValueError(f"文件大小 {file_size} 无法被向量大小 {vec_size} 整除，fvecs 格式可能损坏")

    n_rows = file_size // vec_size
    print(f"✅ 检测到: {n_rows:,} 行 × {dim} 维 (float32)")

    data = np.empty((n_rows, dim), dtype=np.float32)
    with open(path, 'rb') as f:
        for i in tqdm(range(n_rows), desc="读取 fvecs", unit="行"):
            d = struct.unpack('i', f.read(4))[0]
            if d != dim:
                raise ValueError(f"第 {i} 行维度 {d} 与预期 {dim} 不符")
            data[i] = struct.unpack(f'{dim}f', f.read(dim * 4))
    return data


def read_ivecs(path):
    """
    读取 ivecs 文件，返回 numpy 数组 (n_rows, dim)，dtype=int32
    ivecs 格式：每条向量前有 4 字节 int32 表示维度
    """
    print(f"📖 读取 ivecs: {path}")
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        dim_bytes = f.read(4)
        if len(dim_bytes) < 4:
            raise ValueError("文件为空或格式错误")
        dim = struct.unpack('i', dim_bytes)[0]

    vec_size = 4 + dim * 4
    if file_size % vec_size != 0:
        raise ValueError(f"文件大小 {file_size} 无法被向量大小 {vec_size} 整除，ivecs 格式可能损坏")

    n_rows = file_size // vec_size
    print(f"✅ 检测到: {n_rows:,} 行 × {dim} 维 (int32)")

    data = np.empty((n_rows, dim), dtype=np.int32)
    with open(path, 'rb') as f:
        for i in tqdm(range(n_rows), desc="读取 ivecs", unit="行"):
            d = struct.unpack('i', f.read(4))[0]
            if d != dim:
                raise ValueError(f"第 {i} 行维度 {d} 与预期 {dim} 不符")
            data[i] = struct.unpack(f'{dim}i', f.read(dim * 4))
    return data


def write_fvecs(data, path):
    """将 numpy 数组 (n_rows, dim) 写为 fvecs 文件"""
    data = data.astype(np.float32)
    n_rows, dim = data.shape
    print(f"💾 写入 fvecs: {path}  ({n_rows:,} 行 × {dim} 维)")
    with open(path, 'wb') as f:
        for i in tqdm(range(n_rows), desc="写入 fvecs", unit="行"):
            f.write(struct.pack('i', dim))
            f.write(data[i].tobytes())
    print(f"✅ 完成: {path} ({os.path.getsize(path)/(1024**2):.1f} MB)")


def write_ivecs(data, path):
    """将 numpy 数组 (n_rows, dim) 写为 ivecs 文件"""
    data = data.astype(np.int32)
    n_rows, dim = data.shape
    print(f"💾 写入 ivecs: {path}  ({n_rows:,} 行 × {dim} 维)")
    with open(path, 'wb') as f:
        for i in tqdm(range(n_rows), desc="写入 ivecs", unit="行"):
            f.write(struct.pack('i', dim))
            f.write(data[i].tobytes())
    print(f"✅ 完成: {path} ({os.path.getsize(path)/(1024**2):.1f} MB)")


# ─────────────────────────────────────────────────────────
# fbin / ibin 读写工具 (DiskANN 全局 header 格式)
# header: [n_rows(int32), dim(int32)] + 裸数据
# ─────────────────────────────────────────────────────────

def read_fbin(path):
    """
    读取 fbin 文件，返回 numpy 数组 (n_rows, dim)，dtype=float32
    fbin 格式：文件头 8 字节 = [n_rows(int32), dim(int32)]，之后为裸 float32 数据
    """
    print(f"📖 读取 fbin: {path}")
    with open(path, 'rb') as f:
        n_rows, dim = struct.unpack('ii', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.float32).reshape(n_rows, dim)
    print(f"✅ 读取完成: {n_rows:,} 行 × {dim} 维 (float32)")
    return data.copy()


def read_ibin(path):
    """
    读取 ibin 文件，返回 numpy 数组 (n_rows, dim)，dtype=int32
    ibin 格式：文件头 8 字节 = [n_rows(int32), dim(int32)]，之后为裸 int32 数据
    """
    print(f"📖 读取 ibin: {path}")
    with open(path, 'rb') as f:
        n_rows, dim = struct.unpack('ii', f.read(8))
        data = np.frombuffer(f.read(), dtype=np.int32).reshape(n_rows, dim)
    print(f"✅ 读取完成: {n_rows:,} 行 × {dim} 维 (int32)")
    return data.copy()


def write_fbin(data, path):
    """
    将 numpy 数组 (n_rows, dim) 写为 fbin 文件
    格式：[n_rows(int32), dim(int32)] + 裸 float32 数据
    """
    data = data.astype(np.float32)
    n_rows, dim = data.shape
    print(f"💾 写入 fbin: {path}  ({n_rows:,} 行 × {dim} 维)")
    with open(path, 'wb') as f:
        f.write(struct.pack('ii', n_rows, dim))
        chunk = 100_000
        for s in tqdm(range(0, n_rows, chunk), desc="写入 fbin", unit="块"):
            f.write(data[s:s+chunk].tobytes())
    print(f"✅ 完成: {path} ({os.path.getsize(path)/(1024**2):.1f} MB)")


def write_ibin(data, path):
    """
    将 numpy 数组 (n_rows, dim) 写为 ibin 文件
    格式：[n_rows(int32), dim(int32)] + 裸 int32 数据
    """
    data = data.astype(np.int32)
    n_rows, dim = data.shape
    print(f"💾 写入 ibin: {path}  ({n_rows:,} 行 × {dim} 维)")
    with open(path, 'wb') as f:
        f.write(struct.pack('ii', n_rows, dim))
        chunk = 100_000
        for s in tqdm(range(0, n_rows, chunk), desc="写入 ibin", unit="块"):
            f.write(data[s:s+chunk].tobytes())
    print(f"✅ 完成: {path} ({os.path.getsize(path)/(1024**2):.1f} MB)")


def inspect_fbin_ibin(path):
    """检查 fbin/ibin 文件头部信息"""
    print(f"\n🔍 检查文件头: {path}")
    file_size = os.path.getsize(path)
    with open(path, 'rb') as f:
        raw_header = f.read(8)
    if len(raw_header) < 8:
        print("  ❌ 文件过小，无法读取 header")
        return
    n_rows, dim = struct.unpack('ii', raw_header)
    ext = os.path.splitext(path)[1].lower()
    dtype_size = 4
    expected = 8 + n_rows * dim * dtype_size
    ok = file_size == expected
    label = 'float32' if ext == '.fbin' else 'int32'
    print(f"  header[0] n_rows = {n_rows:,}")
    print(f"  header[1] dim    = {dim}")
    print(f"  dtype            = {label}")
    print(f"  文件大小          = {file_size:,} 字节")
    print(f"  期望大小 (8+n*d*4)= {expected:,} 字节")
    print(f"  {'✅ 文件大小与 header 一致' if ok else '❌ 文件大小不匹配！'}")
    return ok


def convert_bin_to_fbin(bin_file, fbin_file, n_rows, dim, verify=False):
    """
    将裸 float32 bin 加上全局 header 转为 fbin
    bin 格式: 纯 float32, 无 header
    fbin 格式: [n_rows(int32), dim(int32)] + 裸 float32 数据
    """
    print("=" * 60)
    print(f"🔄 转换方向: bin → fbin")
    print(f"   参数: {n_rows:,} 行 × {dim} 维")
    print("=" * 60)
    with open(bin_file, 'rb') as f:
        raw = f.read()
    actual = len(raw) // 4
    if actual != n_rows * dim:
        print(f"❌ 文件浮点数数量 {actual:,} 与 {n_rows}×{dim}={n_rows*dim:,} 不符")
        return False
    with open(fbin_file, 'wb') as f:
        f.write(struct.pack('ii', n_rows, dim))
        f.write(raw)
    print(f"✅ 完成: {fbin_file} ({os.path.getsize(fbin_file)/(1024**2):.1f} MB)")
    if verify:
        inspect_fbin_ibin(fbin_file)
        data_orig = np.frombuffer(raw, dtype=np.float32)[:10]
        with open(fbin_file, 'rb') as f:
            f.read(8)
            data_back = np.frombuffer(f.read(40), dtype=np.float32)
        if np.allclose(data_orig, data_back):
            print("  ✅ 数据校验通过（前10个浮点数一致）")
        else:
            print("  ❌ 数据校验失败！")
    return True


def convert_bin_to_ibin(bin_file, ibin_file, n_rows, dim, verify=False):
    """
    将裸 int32 bin 加上全局 header 转为 ibin
    bin 格式: 纯 int32, 无 header
    ibin 格式: [n_rows(int32), dim(int32)] + 裸 int32 数据
    """
    print("=" * 60)
    print(f"🔄 转换方向: bin → ibin")
    print(f"   参数: {n_rows:,} 行 × {dim} 维")
    print("=" * 60)
    with open(bin_file, 'rb') as f:
        raw = f.read()
    actual = len(raw) // 4
    if actual != n_rows * dim:
        print(f"❌ 文件 int32 数量 {actual:,} 与 {n_rows}×{dim}={n_rows*dim:,} 不符")
        return False
    with open(ibin_file, 'wb') as f:
        f.write(struct.pack('ii', n_rows, dim))
        f.write(raw)
    print(f"✅ 完成: {ibin_file} ({os.path.getsize(ibin_file)/(1024**2):.1f} MB)")
    if verify:
        inspect_fbin_ibin(ibin_file)
        data_orig = np.frombuffer(raw, dtype=np.int32)[:10]
        with open(ibin_file, 'rb') as f:
            f.read(8)
            data_back = np.frombuffer(f.read(40), dtype=np.int32)
        if np.array_equal(data_orig, data_back):
            print("  ✅ 数据校验通过（前10个整数一致）")
        else:
            print("  ❌ 数据校验失败！")
    return True


# ─────────────────────────────────────────────────────────
# fvecs / ivecs → bin
# ─────────────────────────────────────────────────────────

def convert_fvecs_to_bin(fvecs_file, bin_file, verify=False):
    """
    fvecs → 纯 float32 bin（去除每行的维度前缀）
    bin 文件格式：n_rows × dim 个 float32 连续存储
    """
    print("=" * 60)
    print(f"🔄 转换方向: fvecs → bin")
    print("=" * 60)
    data = read_fvecs(fvecs_file)
    n_rows, dim = data.shape

    print(f"💾 写入 bin: {bin_file}")
    with open(bin_file, 'wb') as f:
        chunk = 10_000
        for s in tqdm(range(0, n_rows, chunk), desc="写入 bin", unit="行"):
            f.write(data[s:s+chunk].astype(np.float32).tobytes())
    print(f"✅ 完成: {bin_file} ({os.path.getsize(bin_file)/(1024**2):.1f} MB)")

    if verify:
        _verify_vecs_to_bin(data, bin_file, dtype=np.float32)

    return data


def convert_ivecs_to_bin(ivecs_file, bin_file, verify=False):
    """
    ivecs → 纯 int32 bin（去除每行的维度前缀）
    bin 文件格式：n_rows × dim 个 int32 连续存储
    """
    print("=" * 60)
    print(f"🔄 转换方向: ivecs → bin")
    print("=" * 60)
    data = read_ivecs(ivecs_file)
    n_rows, dim = data.shape

    print(f"💾 写入 bin: {bin_file}")
    with open(bin_file, 'wb') as f:
        chunk = 10_000
        for s in tqdm(range(0, n_rows, chunk), desc="写入 bin", unit="行"):
            f.write(data[s:s+chunk].astype(np.int32).tobytes())
    print(f"✅ 完成: {bin_file} ({os.path.getsize(bin_file)/(1024**2):.1f} MB)")

    if verify:
        _verify_vecs_to_bin(data, bin_file, dtype=np.int32)

    return data


def _verify_vecs_to_bin(original_data, bin_file, dtype=np.float32, sample_size=2000):
    """
    校验 vecs → bin 转换正确性：从 bin 回读并与原数组对比
    """
    print("\n" + "=" * 60)
    print("🔍 校验 vecs → bin 转换")
    print("=" * 60)

    n_rows, dim = original_data.shape
    file_size = os.path.getsize(bin_file)
    elem_size = 4
    expected_size = n_rows * dim * elem_size

    # 1. 文件大小检查
    if file_size != expected_size:
        print(f"❌ 文件大小不匹配！期望={expected_size} B，实际={file_size} B")
        return False
    print(f"✅ 文件大小正确: {file_size:,} 字节")

    # 2. 回读 bin 数据
    print("🔄 回读 bin 文件...")
    fmt_char = 'f' if dtype == np.float32 else 'i'
    with open(bin_file, 'rb') as f:
        raw = f.read()
    read_back = np.array(struct.unpack(f'{len(raw)//4}{fmt_char}', raw), dtype=dtype)
    read_back = read_back.reshape(n_rows, dim)

    # 3. 随机采样对比
    sample_size = min(sample_size, n_rows)
    sample_rows = random.sample(range(n_rows), sample_size)
    sample_cols = random.sample(range(dim), min(10, dim))

    print(f"🔬 随机采样 {sample_size} 行 × {len(sample_cols)} 列进行对比...")

    max_diff = 0.0
    mismatches = 0
    for r in tqdm(sample_rows, desc="校验进度", unit="行"):
        for c in sample_cols:
            orig = float(original_data[r, c])
            back = float(read_back[r, c])
            diff = abs(orig - back)
            if diff > max_diff:
                max_diff = diff
            if diff > (1e-6 if dtype == np.float32 else 0):
                mismatches += 1
                if mismatches <= 3:
                    print(f"  不匹配[行{r},列{c}]: 原={orig}, 回读={back}, 差={diff:.2e}")

    print(f"\n📊 校验结果:")
    print(f"  - 采样点数: {sample_size * len(sample_cols):,}")
    print(f"  - 最大差异: {max_diff:.4e}")
    print(f"  - 不匹配数: {mismatches}")

    if mismatches == 0:
        print("✅ 校验通过！fvecs/ivecs → bin 转换完全正确！")
    elif dtype == np.float32 and max_diff < 1e-6:
        print("✅ 校验通过（存在可接受的浮点精度误差）")
    else:
        print("❌ 校验失败！转换存在问题！")
        return False
    return True


# ─────────────────────────────────────────────────────────
# bin / txt 互转（原有功能，保持不变）
# ─────────────────────────────────────────────────────────

def convert_bin_to_txt(bin_file, txt_file, dim=None):
    """
    将二进制格式转换为文本格式
    bin文件格式: float32连续存储
    """
    print(f"📖 读取文件: {bin_file}")
    with open(bin_file, 'rb') as f:
        data = f.read()

    file_size = len(data)
    n_floats = file_size // 4

    print(f"📊 文件大小: {file_size / (1024**3):.2f} GB")
    print(f"📊 总浮点数: {n_floats:,}")

    if dim is None:
        possible_dims = [48, 100, 128, 200, 282, 512, 784, 960, 1536]
        for d in possible_dims:
            if n_floats % d == 0:
                dim = d
                break
        if dim is None:
            print("❌ 无法自动检测维度，请使用 --dim 参数指定")
            return None

    if n_floats % dim != 0:
        print(f"❌ 数据无法整除维度 {dim}")
        return None

    n_rows = n_floats // dim
    print(f"✅ 检测到: {n_rows:,} 行 × {dim} 维")

    print("🔄 解包二进制数据...")
    all_data = struct.unpack(f'{n_floats}f', data)

    print(f"💾 写入文本文件: {txt_file}")
    with open(txt_file, 'w') as out:
        for i in tqdm(range(n_rows), desc="转换进度", unit="行"):
            row_data = all_data[i*dim:(i+1)*dim]
            row_str = ' '.join(f'{x:.18e}' for x in row_data)
            out.write(row_str + '\n')

    print(f"✅ 转换完成: {txt_file}")
    return (n_rows, dim, all_data)


def convert_txt_to_bin(txt_file, bin_file):
    """
    将文本格式转换为二进制格式
    txt文件格式: 每行空格分隔的浮点数
    """
    print(f"📖 读取文件: {txt_file}")

    print("📊 统计数据规模...")
    with open(txt_file, 'r') as f:
        n_rows = sum(1 for _ in f)

    print(f"✅ 检测到: {n_rows:,} 行")

    floats = []
    dim = None
    print(f"🔄 读取并解析文本数据...")

    with open(txt_file, 'r') as f:
        for line in tqdm(f, total=n_rows, desc="读取进度", unit="行"):
            values = [float(x) for x in line.strip().split()]
            if dim is None:
                dim = len(values)
            elif len(values) != dim:
                print(f"❌ 警告：维度不一致！期望 {dim}，实际 {len(values)}")
            floats.extend(values)

    print(f"✅ 数据规模: {n_rows:,} 行 × {dim} 维 = {len(floats):,} 个浮点数")

    print(f"💾 写入二进制文件: {bin_file}")
    with open(bin_file, 'wb') as out:
        out.write(struct.pack(f'{len(floats)}f', *floats))

    bin_size = os.path.getsize(bin_file)
    print(f"✅ 转换完成: {bin_file} ({bin_size / (1024**2):.2f} MB)")
    return (n_rows, dim, floats)


def convert_txt_to_int_bin(txt_file, bin_file):
    """
    将文本格式转换为 int32 裸二进制格式
    txt文件格式: 每行空格分隔的数值（会按 int(float(x)) 转为 int32）
    """
    print(f"📖 读取文件: {txt_file}")

    print("📊 统计数据规模...")
    with open(txt_file, 'r') as f:
        n_rows = sum(1 for _ in f)

    print(f"✅ 检测到: {n_rows:,} 行")

    ints = []
    dim = None
    print("🔄 读取并解析文本数据为 int32...")

    with open(txt_file, 'r') as f:
        for line in tqdm(f, total=n_rows, desc="读取进度", unit="行"):
            values = [int(float(x)) for x in line.strip().split()]
            if dim is None:
                dim = len(values)
            elif len(values) != dim:
                print(f"❌ 警告：维度不一致！期望 {dim}，实际 {len(values)}")
            ints.extend(values)

    print(f"✅ 数据规模: {n_rows:,} 行 × {dim} 维 = {len(ints):,} 个 int32")

    print(f"💾 写入 int32 裸二进制文件: {bin_file}")
    with open(bin_file, 'wb') as out:
        out.write(struct.pack(f'{len(ints)}i', *ints))

    bin_size = os.path.getsize(bin_file)
    print(f"✅ 转换完成: {bin_file} ({bin_size / (1024**2):.2f} MB)")
    return (n_rows, dim, ints)


def verify_conversion(data1, data2, sample_size=1000):
    """
    校验两个数据集是否一致
    data1, data2: (n_rows, dim, floats_list)
    """
    n_rows1, dim1, floats1 = data1
    n_rows2, dim2, floats2 = data2

    print("\n" + "=" * 60)
    print("🔍 开始数据校验")
    print("=" * 60)

    print(f"原始数据: {n_rows1:,} 行 × {dim1} 维 = {len(floats1):,} 数据点")
    print(f"转换数据: {n_rows2:,} 行 × {dim2} 维 = {len(floats2):,} 数据点")

    if n_rows1 != n_rows2 or dim1 != dim2:
        print("❌ 数据规模不匹配！")
        return False

    if len(floats1) != len(floats2):
        print("❌ 数据点数量不匹配！")
        return False

    print("✅ 数据规模匹配")

    n_total = len(floats1)
    sample_indices = list(range(min(100, n_total))) + \
                     random.sample(range(n_total), min(sample_size, n_total))
    sample_indices = list(set(sample_indices))

    print(f"🔬 随机采样 {len(sample_indices)} 个数据点进行校验...")

    max_diff = 0
    mismatches = 0

    for i in tqdm(sample_indices, desc="校验进度", unit="点"):
        diff = abs(floats1[i] - floats2[i])
        max_diff = max(max_diff, diff)
        if diff > 1e-6:
            mismatches += 1
            if mismatches <= 5:
                print(f"  不匹配[{i}]: 原={floats1[i]:.10f}, 转={floats2[i]:.10f}, 差={diff:.10e}")

    print(f"\n📊 校验结果:")
    print(f"  - 测试数据点: {len(sample_indices):,}")
    print(f"  - 最大差异: {max_diff:.10e}")
    print(f"  - 不匹配数: {mismatches}")

    if mismatches == 0:
        print("✅ 数据完全一致！转换正确！")
        return True
    elif max_diff < 1e-5:
        print("⚠️  转换基本正确，存在微小浮点精度差异")
        return True
    else:
        print("❌ 转换存在问题，差异较大！")
        return False


# ─────────────────────────────────────────────────────────
# HDF5 转换（原有功能，保持不变）
# ─────────────────────────────────────────────────────────

def _resolve_parquet_inputs(input_spec):
    """
    解析 parquet 输入：
    - 单文件: a.parquet
    - 逗号分隔: a.parquet,b.parquet
    - 通配符: data/*.parquet
    返回去重后的有序文件列表
    """
    parts = [p.strip() for p in input_spec.split(',') if p.strip()]
    if not parts:
        return []

    files = []
    for p in parts:
        if any(ch in p for ch in ['*', '?', '[']):
            files.extend(sorted(glob.glob(p)))
        else:
            files.append(p)

    seen = set()
    uniq = []
    for f in files:
        if f not in seen:
            uniq.append(f)
            seen.add(f)
    return uniq


def _parquet_df_to_matrix(df, file_path):
    """
    将 parquet DataFrame 转为二维 float32 矩阵：
    1) 单列且元素是向量(list/ndarray)时，按向量列展开
    2) 否则选取所有数值列作为特征列
    """
    if df.empty:
        raise ValueError(f"{file_path} 为空表")

    # 优先处理“向量列”场景（例如: idx + emb，其中 emb 为 ndarray/list）
    vector_candidates = []
    for col in df.columns:
        values = df[col].tolist()
        first_valid = next((v for v in values if v is not None), None)
        if isinstance(first_valid, (list, tuple, np.ndarray)):
            vector_candidates.append(col)

    if vector_candidates:
        # 默认使用第一个向量列；常见数据集只有一个向量列（如 emb）
        vec_col = vector_candidates[0]
        arr = np.array(df[vec_col].tolist(), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{file_path} 向量列 '{vec_col}' 形状异常: {arr.shape}")
        return arr

    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] == 0:
        raise ValueError(
            f"{file_path} 不包含可用数值列，也不是单列向量 parquet"
        )
    return numeric_df.to_numpy(dtype=np.float32, copy=True)


def _verify_parquet_to_bin(parquet_files, bin_file, total_rows, dim, sample_size=2000):
    """
    校验 parquet → bin 转换：
    - 文件大小是否正确
    - 随机采样若干行列，比较 parquet 原值与 bin 回读值
    """
    print("\n" + "=" * 60)
    print("🔍 校验 parquet → bin 转换")
    print("=" * 60)

    if pd is None:
        print("❌ 缺少 pandas，无法执行 parquet 校验")
        return False

    expected_size = total_rows * dim * 4
    actual_size = os.path.getsize(bin_file)
    if actual_size != expected_size:
        print(f"❌ 文件大小不匹配！期望={expected_size} B，实际={actual_size} B")
        return False
    print(f"✅ 文件大小正确: {actual_size:,} 字节")

    if total_rows == 0:
        print("✅ 空数据集，无需抽样校验")
        return True

    sample_rows = sorted(random.sample(range(total_rows), min(sample_size, total_rows)))
    sample_cols = random.sample(range(dim), min(10, dim))
    sampled_rows = {}

    row_ptr = 0
    offset = 0
    for pf in parquet_files:
        if row_ptr >= len(sample_rows):
            break
        data = _parquet_df_to_matrix(pd.read_parquet(pf), pf)
        n_rows = data.shape[0]
        local_indices = []
        while row_ptr < len(sample_rows) and sample_rows[row_ptr] < offset + n_rows:
            local_indices.append((sample_rows[row_ptr], sample_rows[row_ptr] - offset))
            row_ptr += 1
        for g_idx, l_idx in local_indices:
            sampled_rows[g_idx] = data[l_idx].copy()
        offset += n_rows

    if len(sampled_rows) != len(sample_rows):
        print("❌ 抽样行回溯失败，样本数量不完整")
        return False

    mmap_data = np.memmap(bin_file, dtype=np.float32, mode='r', shape=(total_rows, dim))

    max_diff = 0.0
    mismatches = 0
    total_points = len(sample_rows) * len(sample_cols)
    print(f"🔬 随机采样 {len(sample_rows)} 行 × {len(sample_cols)} 列进行对比...")
    for r in tqdm(sample_rows, desc="校验进度", unit="行"):
        src = sampled_rows[r]
        for c in sample_cols:
            diff = abs(float(src[c]) - float(mmap_data[r, c]))
            if diff > max_diff:
                max_diff = diff
            if diff > 1e-6:
                mismatches += 1
                if mismatches <= 3:
                    print(
                        f"  不匹配[行{r},列{c}]: 原={float(src[c])}, 回读={float(mmap_data[r, c])}, 差={diff:.2e}"
                    )

    print("\n📊 校验结果:")
    print(f"  - 采样点数: {total_points:,}")
    print(f"  - 最大差异: {max_diff:.4e}")
    print(f"  - 不匹配数: {mismatches}")

    if mismatches == 0:
        print("✅ 校验通过！parquet → bin 转换正确")
        return True

    print("❌ 校验失败！请检查 parquet 数据与转换逻辑")
    return False


def convert_parquet_to_bin(input_spec, bin_file, verify=False):
    """
    parquet → 纯 float32 bin（无 header）
    支持多个 parquet 文件合并写出：
    - 逗号分隔: a.parquet,b.parquet
    - 通配符: data/*.parquet
    """
    if pd is None:
        print("❌ 请先安装 pandas 与 parquet 引擎: pip install pandas pyarrow")
        return False

    parquet_files = _resolve_parquet_inputs(input_spec)
    if not parquet_files:
        print(f"❌ 未找到 parquet 文件: {input_spec}")
        return False

    print("=" * 60)
    print("🔄 转换方向: parquet → bin")
    print("=" * 60)
    print(f"📦 输入 parquet 文件数: {len(parquet_files)}")
    for i, pf in enumerate(parquet_files, 1):
        print(f"  [{i}] {pf}")

    total_rows = 0
    dim = None
    with open(bin_file, 'wb') as out:
        for pf in parquet_files:
            print(f"\n📖 读取 parquet: {pf}")
            df = pd.read_parquet(pf)
            data = _parquet_df_to_matrix(df, pf)
            n_rows, cur_dim = data.shape

            if dim is None:
                dim = cur_dim
            elif cur_dim != dim:
                print(f"❌ 维度不一致: {pf} 为 {cur_dim}，预期 {dim}")
                return False

            print(f"✅ 数据规模: {n_rows:,} 行 × {cur_dim} 维")
            chunk = 100_000
            for s in tqdm(range(0, n_rows, chunk), desc=f"写入 bin [{os.path.basename(pf)}]", unit="行"):
                out.write(data[s:s+chunk].astype(np.float32, copy=False).tobytes())
            total_rows += n_rows

    out_size = os.path.getsize(bin_file)
    print(f"\n✅ 写入完成: {bin_file}")
    print(f"📊 合并结果: {total_rows:,} 行 × {dim} 维")
    print(f"📊 文件大小: {out_size / (1024**2):.2f} MB")

    if verify:
        return _verify_parquet_to_bin(parquet_files, bin_file, total_rows, dim)
    return True

def inspect_hdf5(hdf5_file):
    """查看HDF5文件的结构和内容"""
    if h5py is None:
        print("❌ 请先安装h5py: pip install h5py")
        return
    print(f"📖 HDF5文件结构: {hdf5_file}")
    print("=" * 60)
    with h5py.File(hdf5_file, 'r') as f:
        def print_item(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  📊 [{name}]  shape={obj.shape}  dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"  📁 [{name}/]")
        f.visititems(print_item)
    print("=" * 60)
    print("常见ANN-benchmarks字段说明:")
    print("  train     - 数据库向量 (base)")
    print("  test      - 查询向量 (query)")
    print("  neighbors - 真实近邻索引 (ground truth)")
    print("  distances - 真实近邻距离")


def convert_hdf5(hdf5_file, out_prefix, split=None, verify=False):
    """
    将ANN-benchmarks HDF5格式同时转换为bin和txt
    split: 指定只转换某个字段 (train/test/neighbors/distances)，默认全部转换
    """
    if h5py is None:
        print("❌ 请先安装h5py: pip install h5py")
        return False

    inspect_hdf5(hdf5_file)
    print()

    split_map = {
        'train':     ('_base.bin',        '_base.txt',        np.float32),
        'test':      ('_query.bin',       '_query.txt',       np.float32),
        'neighbors': ('_groundtruth.bin', '_groundtruth.txt', np.int32),
        'distances': ('_distances.bin',   '_distances.txt',   np.float32),
    }

    with h5py.File(hdf5_file, 'r') as f:
        available = list(f.keys())
        targets = [split] if split else [k for k in split_map if k in available]

        for key in targets:
            if key not in f:
                print(f"⚠️  HDF5中不存在 '{key}'，跳过")
                continue

            bin_suffix, txt_suffix, dtype = split_map.get(
                key, (f'_{key}.bin', f'_{key}.txt', np.float32))
            out_bin = out_prefix + bin_suffix
            out_txt = out_prefix + txt_suffix

            data = f[key][:].astype(dtype)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            n_rows, dim = data.shape
            label = {'train': '数据库(base)', 'test': '查询(query)',
                     'neighbors': 'Ground Truth索引', 'distances': 'Ground Truth距离'}.get(key, key)

            print(f"🔄 转换 [{label}]: {n_rows:,}行 × {dim}维")

            # --- 写bin ---
            fmt_char = 'f' if dtype == np.float32 else 'i'
            flat = data.flatten()
            with open(out_bin, 'wb') as bf:
                chunk = 1_000_000
                for s in tqdm(range(0, len(flat), chunk), desc=f"写bin [{key}]", unit="MB"):
                    bf.write(struct.pack(f'{len(flat[s:s+chunk])}{fmt_char}', *flat[s:s+chunk]))
            print(f"  ✅ bin: {out_bin} ({os.path.getsize(out_bin)/(1024**2):.1f} MB)")

            # --- 写txt ---
            with open(out_txt, 'w') as tf:
                for i in tqdm(range(n_rows), desc=f"写txt [{key}]", unit="行"):
                    tf.write(' '.join(f'{v:.10e}' if dtype == np.float32
                                     else str(int(v)) for v in data[i]) + '\n')
            print(f"  ✅ txt: {out_txt} ({os.path.getsize(out_txt)/(1024**2):.1f} MB)")

            # --- 校验 ---
            if verify:
                print(f"  🔍 校验 [{key}]...")
                with open(out_bin, 'rb') as bf:
                    raw = bf.read()
                read_back = struct.unpack(f'{len(raw)//4}{fmt_char}', raw)
                sample = random.sample(range(len(flat)), min(1000, len(flat)))
                max_diff = max(abs(float(flat[i]) - float(read_back[i])) for i in sample)
                if max_diff < 1e-5:
                    print(f"  ✅ 校验通过！最大差异={max_diff:.2e}")
                else:
                    print(f"  ❌ 校验失败！最大差异={max_diff:.2e}")
            print()

    return True


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

def _print_usage():
    print("=" * 60)
    print("多格式数据转换工具 (hdf5 / bin / txt / fvecs / ivecs)")
    print("=" * 60)
    print("\n用法:")
    print("  hdf5 → bin + txt (GloVe/ANN-benchmarks):")
    print("    python convert_data.py <数据.hdf5> <输出前缀> [--split train|test|neighbors|distances] [--verify]")
    print("\n  查看HDF5结构:")
    print("    python convert_data.py <数据.hdf5> inspect")
    print("\n  fvecs → bin  (GIST/SIFT float 向量):")
    print("    python convert_data.py <输入.fvecs> <输出.bin> [--verify]")
    print("\n  ivecs → bin  (GIST/SIFT int 近邻):")
    print("    python convert_data.py <输入.ivecs> <输出.bin> [--verify]")
    print("\n  bin → txt:")
    print("    python convert_data.py <输入.bin> <输出.txt> [--dim <维度>] [--verify]")
    print("\n  txt → bin:")
    print("    python convert_data.py <输入.txt> <输出.bin> [--verify]            # float32 裸bin")
    print("    python convert_data.py <输入.txt> <输出.bin> --int32 [--verify]   # int32 裸bin")
    print("\n  txt → fbin (向量数据):")
    print("    python convert_data.py <输入.txt> <输出.fbin>")
    print("\n  txt → ibin (Ground Truth):")
    print("    python convert_data.py <输入.txt> <输出.ibin>")
    print("\n  parquet → bin (支持多文件合并):")
    print("    python convert_data.py <输入.parquet> <输出.bin> [--verify]")
    print("    python convert_data.py 'a.parquet,b.parquet' <输出.bin> [--verify]")
    print("    python convert_data.py 'data/*.parquet' <输出.bin> [--verify]")
    print("\n  fbin/ibin → txt:")
    print("    python convert_data.py <输入.fbin/.ibin> <输出.txt>")
    print("\n参数:")
    print("  --dim <N>       指定维度（bin→txt 自动检测失败时使用）")
    print("  --split <name>  仅转换 hdf5 的某个字段")
    print("  --verify        转换后进行数据校验")
    print("  --int32         txt→.bin 时按 int32 写出（默认 float32）")
    print("\n示例:")
    print("  python convert_data.py gist_base.fvecs gist_base.bin --verify")
    print("  python convert_data.py gist_groundtruth.ivecs gist_groundtruth.bin --verify")
    print("  python convert_data.py glove-100-angular.hdf5 glove100 --verify")
    print("  python convert_data.py glove-100-angular.hdf5 inspect")
    print("  python convert_data.py data.bin data.txt --dim 960 --verify")
    print("  python convert_data.py data.txt data.bin --verify")
    print("  python convert_data.py gt.txt gt.bin --int32 --verify")


def main():
    if len(sys.argv) < 2:
        _print_usage()
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) >= 3 else None

    # 特殊命令: inspect
    if output_file == 'inspect':
        inspect_hdf5(input_file)
        sys.exit(0)

    if output_file is None:
        print("❌ 请指定输出文件或前缀")
        _print_usage()
        sys.exit(1)

    # 解析可选参数
    dim = None
    verify = False
    split = None
    int32_mode = False
    i = 3
    while i < len(sys.argv):
        if sys.argv[i] == '--dim' and i + 1 < len(sys.argv):
            dim = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--verify':
            verify = True
            i += 1
        elif sys.argv[i] == '--split' and i + 1 < len(sys.argv):
            split = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--int32':
            int32_mode = True
            i += 1
        else:
            i += 1

    parquet_mode = '.parquet' in input_file.lower()
    if parquet_mode:
        parquet_files = _resolve_parquet_inputs(input_file)
        if not parquet_files:
            print(f"❌ 未找到 parquet 文件: {input_file}")
            sys.exit(1)
        missing = [p for p in parquet_files if not os.path.exists(p)]
        if missing:
            print("❌ 以下 parquet 文件不存在:")
            for p in missing:
                print(f"   - {p}")
            sys.exit(1)
    elif not os.path.exists(input_file):
        print(f"❌ 文件不存在: {input_file}")
        sys.exit(1)

    print("=" * 60)
    print("多格式数据转换工具")
    print("=" * 60)
    print()

    in_ext  = os.path.splitext(input_file)[1].lower()
    out_ext = os.path.splitext(output_file)[1].lower()

    # ── fvecs → bin ──────────────────────────────────────
    if in_ext == '.fvecs' and out_ext == '.bin':
        convert_fvecs_to_bin(input_file, output_file, verify=verify)

    # ── txt → fbin ──────────────────────────────────────
    elif in_ext == '.txt' and out_ext == '.fbin':
        print("🔄 转换方向: txt → fbin\n")
        rows = []
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="读取 txt", unit="行"):
                rows.append([float(x) for x in line.strip().split()])
        write_fbin(np.array(rows, dtype=np.float32), output_file)
        if verify:
            inspect_fbin_ibin(output_file)

    # ── txt → ibin ──────────────────────────────────────
    elif in_ext == '.txt' and out_ext == '.ibin':
        print("🔄 转换方向: txt → ibin\n")
        rows = []
        with open(input_file, 'r') as f:
            for line in tqdm(f, desc="读取 txt", unit="行"):
                rows.append([int(float(x)) for x in line.strip().split()])
        write_ibin(np.array(rows, dtype=np.int32), output_file)
        if verify:
            inspect_fbin_ibin(output_file)

    # ── bin → fbin (裸float32加全局header) ───────────────
    elif in_ext == '.bin' and out_ext == '.fbin':
        if dim is None:
            print("❌ 请用 --dim 指定维度")
            sys.exit(1)
        n_rows_auto = os.path.getsize(input_file) // 4 // dim
        convert_bin_to_fbin(input_file, output_file, n_rows_auto, dim, verify=True)

    # ── bin → ibin (裸int32加全局header) ────────────────
    elif in_ext == '.bin' and out_ext == '.ibin':
        if dim is None:
            print("❌ 请用 --dim 指定维度")
            sys.exit(1)
        n_rows_auto = os.path.getsize(input_file) // 4 // dim
        convert_bin_to_ibin(input_file, output_file, n_rows_auto, dim, verify=True)

    # ── ivecs → bin ──────────────────────────────────────
    elif in_ext == '.ivecs' and out_ext == '.bin':
        convert_ivecs_to_bin(input_file, output_file, verify=verify)

    # ── hdf5 → bin + txt ─────────────────────────────────
    elif in_ext in ('.hdf5', '.h5'):
        print("🔄 转换方向: hdf5 → bin + txt\n")
        convert_hdf5(input_file, output_file, split=split, verify=verify)

    # ── parquet → bin (支持多文件合并) ───────────────────
    elif parquet_mode and out_ext == '.bin':
        convert_parquet_to_bin(input_file, output_file, verify=verify)

    # ── bin → txt ─────────────────────────────────────────
    elif in_ext == '.bin' and out_ext == '.txt':
        print("🔄 转换方向: bin → txt\n")
        result1 = convert_bin_to_txt(input_file, output_file, dim)
        if result1 and verify:
            print("\n🔄 执行反向转换以校验数据...")
            with open(output_file, 'r') as f:
                floats2, n_rows2, dim2 = [], 0, None
                for line in tqdm(f, desc="回读文本", unit="行"):
                    values = [float(x) for x in line.strip().split()]
                    if dim2 is None:
                        dim2 = len(values)
                    floats2.extend(values)
                    n_rows2 += 1
            verify_conversion(result1, (n_rows2, dim2, floats2))

    # ── fbin → txt ──────────────────────────────────────
    elif in_ext == '.fbin' and out_ext == '.txt':
        print("🔄 转换方向: fbin → txt\n")
        data = read_fbin(input_file)
        n_rows, dim = data.shape
        with open(output_file, 'w') as out:
            for i in tqdm(range(n_rows), desc="写 txt", unit="行"):
                out.write(' '.join(f'{x:.18e}' for x in data[i]) + '\n')

    # ── ibin → txt ──────────────────────────────────────
    elif in_ext == '.ibin' and out_ext == '.txt':
        print("🔄 转换方向: ibin → txt\n")
        data = read_ibin(input_file)
        n_rows, dim = data.shape
        with open(output_file, 'w') as out:
            for i in tqdm(range(n_rows), desc="写 txt", unit="行"):
                out.write(' '.join(str(int(x)) for x in data[i]) + '\n')

    # ── txt → bin ─────────────────────────────────────────
    elif in_ext == '.txt' and out_ext == '.bin':
        if int32_mode:
            print("🔄 转换方向: txt → int32 bin\n")
            result1 = convert_txt_to_int_bin(input_file, output_file)
            if result1 and verify:
                print("\n🔄 执行反向转换以校验数据...")
                with open(output_file, 'rb') as f:
                    raw = f.read()
                ints2 = struct.unpack(f'{len(raw)//4}i', raw)
                n_rows2 = len(ints2) // result1[1]
                mismatches = 0
                sample_n = min(2000, len(result1[2]))
                for idx in random.sample(range(len(result1[2])), sample_n):
                    if int(result1[2][idx]) != int(ints2[idx]):
                        mismatches += 1
                        if mismatches <= 5:
                            print(f"  不匹配[{idx}]: 原={result1[2][idx]}, 转={ints2[idx]}")
                if mismatches == 0:
                    print(f"✅ int32 校验通过！(rows={n_rows2}, dim={result1[1]})")
                else:
                    print(f"❌ int32 校验失败，不匹配数: {mismatches}")
        else:
            print("🔄 转换方向: txt → bin\n")
            result1 = convert_txt_to_bin(input_file, output_file)
            if result1 and verify:
                print("\n🔄 执行反向转换以校验数据...")
                with open(output_file, 'rb') as f:
                    raw = f.read()
                floats2 = struct.unpack(f'{len(raw)//4}f', raw)
                verify_conversion(result1, (len(floats2)//result1[1], result1[1], floats2))

    else:
        print("❌ 无法识别转换方向！")
        print("支持的转换:")
        print("  .fvecs → .bin")
        print("  .ivecs → .bin")
        print("  .hdf5/.h5 → 前缀 (bin + txt)")
        print("  .parquet(支持多文件) → .bin")
        print("  .bin → .txt")
        print("  .txt → .bin")
        print("  .bin → .fbin  (--dim 必填，自动加全局 header)")
        print("  .bin → .ibin  (--dim 必填，自动加全局 header)")
        print("  .txt → .fbin")
        print("  .txt → .ibin")
        sys.exit(1)


if __name__ == '__main__':
    main()
