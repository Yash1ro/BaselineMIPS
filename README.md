# BaselineMIPS

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
