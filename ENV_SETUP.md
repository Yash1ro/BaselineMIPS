# BaselineMIPS 环境安装清单

## 1. 系统要求

- Linux (推荐 Ubuntu 22.04/24.04)
- CPU 支持 AVX2；若要直接编译当前 PAG 配置，建议支持 AVX-512
- 建议内存 32GB+（大数据集建图时更高）
- 建议磁盘 200GB+（索引和结果文件较大）

## 2. 系统依赖（必装）

```bash
sudo apt update
sudo apt install -y \
  build-essential g++ make cmake pkg-config \
  libboost-all-dev libgoogle-perftools-dev \
  python3 python3-pip python3-venv
```

说明：
- `build-essential/g++/make/cmake`：编译 ip-nsw、mobius、MAG、PAG
- `libboost-all-dev`：MAG 需要
- `libgoogle-perftools-dev`：MAG 的 `tcmalloc/profiler` 链接需要

## 3. Python 环境（推荐虚拟环境）

```bash
cd /path/to/BaselineMIPS
python3 -m venv exp
source exp/bin/activate
python -m pip install -U pip setuptools wheel
```

## 4. Python 包（按需）

### 4.1 Benchmark 与常用脚本

```bash
pip install numpy matplotlib tqdm requests h5py
```

### 4.2 如需运行 ScaNN 相关脚本

```bash
pip install scann
```

### 4.3 如需运行 Faiss 相关脚本（CPU 版）

```bash
pip install faiss-cpu
```

### 4.4 如需下载/处理部分数据工具脚本

```bash
pip install tensorflow-datasets
```

## 5. 克隆后编译

仓库根目录已提供一键脚本：

```bash
bash build_all.sh
```

可选参数：

```bash
bash build_all.sh --clean
bash build_all.sh --jobs 16
bash build_all.sh --clean --jobs 16
```

## 6. 快速自检

```bash
# 检查编译产物
ls ip-nsw/build/main
ls mobius/mobius
ls MAG/build/test || true
ls PAG/build/PEOs

# 跑一个示例 benchmark（按你已有数据）
python benchmark/benchmark.py --dataset glove200 --algorithms pag
```

## 7. 常见问题

1. PAG 编译报 AVX-512 相关错误
- 原因：当前 `PAG/CMakeLists.txt` 启用了 `-mavx512*`。
- 处理：去掉 AVX-512 编译选项，或换支持 AVX-512 的机器。

2. MAG 链接时报 `tcmalloc` 或 `profiler` 未找到
- 原因：缺少 gperftools 开发库。
- 处理：安装 `libgoogle-perftools-dev` 后重新编译。

3. ScaNN/Faiss pip 安装失败
- 处理：先升级 pip/setuptools/wheel，再安装；必要时固定 Python 版本为 3.10/3.11。
