# BaselineMIPS

MIPS（Maximum Inner Product Search）算法基准测试框架，包含 7 种近似最近邻搜索算法的统一 Benchmark 接口和自动化测试流水线。

## 算法列表

| 算法 | 目录 | 类型 | 论文 |
|------|------|------|------|
| **MAG** | `MAG/` | 内存增强图搜索 | — |
| **ScaNN** | (in-memory) | 各向异性向量量化 | — |
| **ip-NSW** | `ip-nsw/` | 内积导航小世界图 | NeurIPS 2018 |
| **Möbius-Graph** | `mobius/` | Möbius 变换图搜索 | NeurIPS 2019 |
| **PIF-PAG** | `PAG_new/` | PAG 改进版 | — |
| **PAG-only** | `PAG_without_projection/` | PAG 无投影变体 | — |

### 官方实现

- **ip-NSW** 是官方实现，来自 NIPS 2018 论文 *"Non-metric Similarity Graphs for Maximum Inner Product Search"* (Morozov & Babenko)，基于 [hnswlib](https://github.com/nmslib/hnswlib)。
- **Möbius-Graph** 是官方实现，来自 NeurIPS 2019 论文 *"Möbius Transformation for Fast Inner Product Search on Graph"*，底层图搜索基于 [SONG](https://github.com/sunbelbd/song)。

## 环境配置

```bash
cd /home/gu/baseline
source exp/bin/activate
```

## 更新日志

### 2026-03-27

#### Benchmark 结果文件与作图增强

**新功能：**

1. **全量运行自动命名**：在某数据集下运行全部算法（或多个算法）时，结果文件自动命名为
   `benchmark/results/{dataset}_{YYYYMMDD_HHMMSS}.txt`，不再覆盖固定的 `result.txt`。
   随后的作图步骤直接基于本次生成的文件。

2. **单算法原地更新**：若 `--algorithms` 只指定一个算法，benchmark 会找到该数据集下最新的
   结果文件（`results/{dataset}_*.txt`，按修改时间排序），仅替换文件中该算法的行，其他
   算法的数据与元信息头部保持不变。若找不到已有文件则创建新的时间戳命名文件。

3. **结果文件头部记录数据集信息**：每个结果文件开头以 `#` 注释行保存：
   - 数据集名称 (`dataset`)
   - 数据库向量条数 (`db_size`)
   - 向量维度 (`dim`)
   - 查询条数 (`query_size`)
   - 运行时间戳 (`timestamp`)

4. **结果文件同时保存算法参数配置**：每次运行的算法的 sweep 参数（如 `mag_efs`、
   `scann_num_leaves`、`ipnsw_ef_values` 等）也以 `#` 注释行写入文件头部，便于复现。

5. **`result_plot.py` 兼容新格式**：`load_results` 自动跳过所有 `#` 注释行，同时保持对
   旧格式（无注释头）文件的向后兼容。`save_results`、`plot_results`、CLI 接口均无变化。

**新结果文件格式示例：**

```
# dataset: music100
# db_size: 1000000
# dim: 100
# query_size: 10000
# timestamp: 2026-03-27T14:30:22
# --- params:mag ---
# mag_efs: [100, 200, 400, 600, 800, 1000]
# --- params:scann ---
# scann_distance: dot_product
# scann_mode: reorder
# scann_num_leaves: 2000
# scann_reorder_values: [400, 500, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000]
# ...
algorithm	budget	recall	qps
mag	100	0.66331100	79.942188
mag	200	0.77252200	79.537111
...
```

---

## Benchmark 使用方法

### 全量运行（所有算法，自动生成时间戳文件）

```bash
cd /home/gu/baseline
source exp/bin/activate

# 运行所有算法，结果自动保存为 benchmark/results/music100_YYYYMMDD_HHMMSS.txt
python benchmark/benchmark.py --dataset music100

# 指定数据集
python benchmark/benchmark.py --dataset glove100
python benchmark/benchmark.py --dataset glove200
python benchmark/benchmark.py --dataset dinov2
python benchmark/benchmark.py --dataset book_corpus
```

### 单算法运行（更新最新结果文件中的对应部分）

```bash
# 仅跑 mag，自动找到 results/music100_*.txt 中最新的文件并更新 mag 相关行
python benchmark/benchmark.py --dataset music100 --algorithms mag

# 仅跑 scann
python benchmark/benchmark.py --dataset music100 --algorithms scann

# 仅跑 ipnsw
python benchmark/benchmark.py --dataset music100 --algorithms ipnsw
```

### 多算法部分运行（创建新时间戳文件）

```bash
# 跑 mag 和 ipnsw，生成新的时间戳文件
python benchmark/benchmark.py --dataset music100 --algorithms mag,ipnsw
```

### 手动指定结果文件路径

```bash
# 使用 --result-txt 覆盖自动路径逻辑
python benchmark/benchmark.py --dataset music100 --result-txt /path/to/my_result.txt
```

### 综合基准测试（全数据集 × 全算法 × 多 top-K）

`run_full_benchmark.py` 可一键运行所有数据集、所有算法、多个 top-K 的完整 Recall-QPS 基准测试，
同时统计 **peak memory**、**索引构建时间**，并带时间戳保存到 `statistics.log`。

```bash
source exp/bin/activate

# 1. 运行全部测试（5 数据集 × 5 算法 × top-10/100/500）
python benchmark/run_full_benchmark.py

# 2. 只测试指定数据集
python benchmark/run_full_benchmark.py --datasets music100 glove100

# 3. 只测试指定算法
python benchmark/run_full_benchmark.py --algorithms mag ipnsw scann

# 4. 只测试指定 top-K
python benchmark/run_full_benchmark.py --top-ks 10 100

# 5. 组合使用
python benchmark/run_full_benchmark.py --datasets music100 --algorithms mag ipnsw --top-ks 10 100

# 6. 跳过 top-500 groundtruth 自动生成（若无预计算 GT 则跳过该组合）
python benchmark/run_full_benchmark.py --skip-gt-gen

# 7. ScaNN 使用 leaves 模式
python benchmark/run_full_benchmark.py --scann-mode leaves
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | 全部 5 个 | 要测试的数据集（music100 glove100 glove200 dinov2 book_corpus） |
| `--algorithms` | 全部算法 | 要测试的算法（mag scann ipnsw mobius pag_new pag_without_projection） |
| `--top-ks` | `10 100 500` | 要测试的 top-K 值 |
| `--skip-gt-gen` | 关闭 | 缺少 top-K > 100 的 groundtruth 时跳过而非暴力生成 |
| `--scann-mode` | `reorder` | ScaNN 参数扫描模式（reorder / leaves） |

**输出文件：**

| 文件 | 说明 |
|------|------|
| `statistics.log` | 主结果文件，每组测试一个表格块 + 末尾汇总表（含原始 JSON） |
| `benchmark/results/{dataset}_top{K}_{timestamp}.txt` | 每组 (数据集, top-K) 的 TSV 结果 |
| `benchmark/imgs/{dataset}_top{K}.png` | 对应的 Recall-QPS 曲线图 |

**`statistics.log` 格式示例：**

```
┌── music100  top-10  @  2026-04-07T22:39:10  (9m43s) ──────────────────────────┐
│
│  Algorithm   Build Time  Build Peak  Query Peak  │  Recall-QPS Sweep
│  ────────── ─────────── ─────────── ───────────  │  ──────────────────────────
│  mag           (cached)           -      980 MB  │  R=0.9342@100  R=0.9927@1000
│  scann             9.1s     1352 MB      907 MB  │  R=0.9711@400  R=0.9804@5000
│  ipnsw         (cached)           -     1192 MB  │  R=0.9905@100  R=0.9999@2000
│  mobius        (cached)           -     1451 MB  │  R=0.9626@50   R=0.9987@1000
│  pag_new            9.9s     2423 MB     1366 MB  │  R=0.0007@10   R=0.0006@990
│
└──────────────────────────────────────────────────────────────────────┘
```

> **注意：**
> - 首次运行时索引会自动构建，后续运行复用已有索引（Build Time 显示为 `(cached)`）。
> - top-500 的 groundtruth 若不存在会自动暴力计算（大数据集耗时较长），可用 `--skip-gt-gen` 跳过。
> - QPS 测量在单线程环境下进行，确保公平比较。
> - 完整运行（5×5×3 = 75 组）预计耗时较长，建议先用小范围参数测试。

---

### 对已有结果文件单独作图

```bash
# 对指定结果文件作图（支持带 # 注释头的新格式和旧格式）
python benchmark/tools/result_plot.py \
    --input benchmark/results/music100_20260327_143022.txt \
    --dataset music100 \
    --top-k 100 \
    --title "music100 Recall-QPS"
```

---

## benchmark 目录脚本说明（作用 + 参数）

下面按“可直接执行脚本”和“内部算法模块”两类整理。参数默认值以当前代码为准。

### 1) 可直接执行脚本（CLI）

#### `benchmark/benchmark.py`

- 作用：单数据集基准测试主入口。可运行一个或多个算法，生成/更新结果文件并作图。
- 典型命令：`python benchmark/benchmark.py --dataset music100 --algorithms mag,scann,ipnsw,mobius,pag_new,pag_without_projection`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `music100` | 数据集名称。可选：`music100`/`glove100`/`glove200`/`dinov2`/`book_corpus`。 |
| `--scann-mode` | `reorder` | ScaNN 扫描模式。可选：`reorder`（扫 `reorder`）/`leaves`（扫 `leaves_to_search`）。 |
| `--algorithms` | `mag,scann,ipnsw,mobius,pag_new,pag_without_projection` | 逗号分隔算法列表。`mag`=MAG, `scann`=ScaNN, `ipnsw`=ip-NSW（官方）, `mobius`=Möbius-Graph（官方）, `pag_new`=PIF-PAG, `pag_without_projection`=PAG-only。 |
| `--result-txt` | `None` | 结果文件路径。为空时自动选择路径：单算法优先更新最新文件，多算法/全量会创建带时间戳新文件。 |
| `--plot` | `None` | 输出图片路径。为空时默认到 `benchmark/imgs/{dataset}_top{K}.png`。 |
| `--title` | `None` | 图标题。为空时使用默认标题。 |
| `--top-k` | `None` | 覆盖数据集默认 top-K（如 `500`）。 |

#### `benchmark/run_full_benchmark.py`

- 作用：全量基准入口。支持“多数据集 × 多算法 × 多 top-K”，记录构建时间/峰值内存/查询结果并写 `statistics.log`。
- 典型命令：`python benchmark/run_full_benchmark.py --datasets music100 glove100 --algorithms mag scann ipnsw --top-ks 10 100`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | 全部数据集 | 要测试的数据集列表。当前集合：`music100 glove100 glove200 dinov2 book_corpus gist1m ir101 openai1536`。 |
| `--algorithms` | 全部算法 | 要测试的算法列表。当前集合：`mag`=MAG, `scann`=ScaNN, `ipnsw`=ip-NSW（官方）, `mobius`=Möbius-Graph（官方）, `pag_new`=PIF-PAG, `pag_without_projection`=PAG-only。 |
| `--top-ks` | `10 100 500` | 要测试的 top-K 列表。 |
| `--skip-gt-gen` | 关闭 | 当 top-K>100 且缺少预计算 GT 时，跳过该组合（不做暴力 GT 生成）。 |
| `--scann-mode` | `reorder` | ScaNN 模式：`reorder` 或 `leaves`。 |

#### `benchmark/run_pag_comparison.py`

- 作用：比较 PAG 变体（`pag_new`/`pag_without_projection`），并更新对比图与结果。（注意：基础 PAG 未上传，仅支持 PIF-PAG 和 PAG-only。）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | 全部数据集 | 参与对比的数据集列表。 |
| `--top-ks` | `10 100` | 参与对比的 top-K 列表。 |
| `--variants` | `pag_new pag_without_projection` | 选择要跑的 PAG 变体。 |
| `--skip-index-build` | 关闭 | 仅执行 query；若某变体索引不存在则跳过，不触发建索引。 |

#### `benchmark/rerun_pag_queries.py`

- 作用：对指定数据集重新执行 PAG 变体 query（默认 top-10），可在已有索引上快速复测并输出结果文件。（注意：基础 PAG 未上传，仅支持 PIF-PAG 和 PAG-only。）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--datasets` | 全部数据集 | 要重跑 PAG 变体 query 的数据集列表。 |

#### `benchmark/generate_top10_plots.py`

- 作用：从已有结果文件自动生成 top-10 的单数据集图、多数据集对比图和内存柱状图。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--result-rank` | `1` | 按时间倒序选择第 N 新的结果文件（`1`=最新，`2`=次新）。 |
| `--prefer-algorithm` | `None` | 只优先选择包含该算法的结果文件（例如 `pag_new`）；找不到时可回退。 |

#### `benchmark/plot_statistics_metrics.py`

- 作用：从 `statistics.log` 提取指标，按数据集生成 4 类柱状图（建索引时间、建索引峰值内存、查询时间、查询峰值内存）。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--top-k` | `10` | 只绘制该 top-K 的统计记录。 |

#### `benchmark/run_large_datasets.py`

- 作用：对大数据集（当前固定 `ir101` 与 `book_corpus`）执行 PAG 变体（PIF-PAG / PAG-only）测试，并在每轮后清理索引以节省磁盘。
- 参数：无 CLI 参数（当前通过脚本常量控制：`DATASETS`、`TOP_K`、`VARIANTS`）。

#### `benchmark/run_pag_and_plot.py`

- 作用：对所有数据集重跑 PAG 变体（要求索引已存在），写入新结果文件并自动调用 `generate_top10_plots.py` 生成图。
- 参数：无 CLI 参数（当前通过脚本常量控制：`ALL_DATASETS`，以及固定 `top10` 流程）。

### 2) 内部算法模块（通常由主脚本调用）

这些脚本通常不直接从命令行调用，而是由 `benchmark.py` / `run_full_benchmark.py` 动态导入并执行 `run(...)`。

| 脚本 | 作用 | 入口函数参数 |
|------|------|--------------|
| `benchmark/benchmark_mag.py` | MAG 参数扫（`efs`） | `run(config, ground_truth)` |
| `benchmark/benchmark_scann.py` | ScaNN 参数扫（`reorder` 或 `leaves`） | `run(config, database, queries, ground_truth)` |
| `benchmark/benchmark_ipnsw.py` | **ip-NSW**（官方实现）参数扫（`efSearch`） | `run(config, ground_truth)` |
| `benchmark/benchmark_mobius.py` | **Möbius-Graph**（官方实现）参数扫（`search_budget`） | `run(config, ground_truth)` |
| `benchmark/benchmark_pag_new.py` | PIF-PAG 脚本执行与输出解析 | `run(config)` |
| `benchmark/benchmark_pag_without_projection.py` | PAG-only 脚本执行与输出解析 | `run(config)` |

### 3) 这些模块的“参数从哪里来”

内部模块主要通过 `benchmark/common.py` 里的 `DatasetConfig` 读取参数，不走 CLI。常用字段如下：

| 参数组 | 关键字段 | 说明 |
|--------|----------|------|
| 数据与 GT | `name` `dim` `top_k` `db_size` `query_size` `database_bin` `query_bin` `groundtruth_bin_top100/top500` | 数据规模、维度、输入路径与 GT 路径。 |
| MAG | `mag_efs` | MAG 的 `efs` 扫描列表。 |
| ScaNN | `scann_distance` `scann_mode` `scann_num_leaves` `scann_leaves_to_search` `scann_reorder_values` `scann_leaves_values` | ScaNN 建索引与查询扫描参数。 |
| ip-NSW | `ipnsw_m` `ipnsw_ef_construction` `ipnsw_ef_values` | 图构建与查询参数。 |
| Mobius | `mobius_budget_values` | Mobius 查询预算扫描列表。 |
| PAG 变体 | `pag_new_run_script` `pag_new_hnsw_efc` `pag_new_hnsw_M` `pag_new_hnsw_L` `pag_without_proj_run_script` | PIF-PAG / PAG-only 的运行脚本与图参数。 |

> 说明：不同数据集在 `DatasetConfig.__post_init__` 中会覆盖默认值（如维度、数据路径、参数扫描范围）。

---

### 2026-03-25

- 在 benchmark 配置中新增了 `book_corpus` 数据集支持（`dim=1024`、`top_k=100`、数据路径与各算法结果输出路径）。
- 将 benchmark 侧 `ipnsw` 的建图参数在 `dinov2` 与 `book_corpus` 上统一为 `efConstruction=500` 与 `M=32`。
- 修复了 `ip-nsw/main.cpp` 在大规模数据集上的内存问题：
	- 分配大小计算改为 64 位安全（`size_t`），避免大数据下整数溢出；
	- 数组释放由 `delete` 修正为 `delete[]`。
- 修复了 `MAG/include/util.h` 的内存处理问题：
	- 当维度已对齐时增加安全的提前返回；
	- 分配大小计算改为 `size_t`；
	- 增加 `memalign` 空指针检查；
	- 修复分配与释放方式不匹配，统一使用 `delete[]` 释放对齐输入。
- 当前已知运行问题：
	- `book_corpus` 全流程 benchmark 在 Mobius 阶段可能因生成 txt 文件触发 `OSError: [Errno 28] No space left on device`（磁盘空间不足）。
