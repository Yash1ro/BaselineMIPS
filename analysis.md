# ScaNN 参数对比分析（ann-benchmarks vs 你的 benchmark / Iceberg / test_single.py）

## 1. ann-benchmarks 里 ScaNN 的参数是怎么设的

基于 ann-benchmarks 仓库中的 ScaNN wrapper 与配置（`ann_benchmarks/algorithms/scann/module.py` 与 `ann_benchmarks/algorithms/scann/config.yml`），其思路是：

- 构建参数（build-time）由 `args` 给出：
  - `n_leaves`
  - `avq_threshold`
  - `dims_per_block`
  - `dist`（`dot_product` 或 `squared_l2`）
- 查询参数（search-time）由 `query_args` 给出：
  - `(leaves_to_search, reorder)` 成对 sweep。

对应 wrapper 逻辑：

- `fit()`：
  - `builder(X, 10, dist)`
  - `.tree(n_leaves, 1, training_sample_size=len(X), spherical=..., quantize_centroids=True)`
  - `.score_ah(dims_per_block, anisotropic_quantization_threshold=avq_threshold)`
  - `.reorder(1)`
- `set_query_arguments(leaves_reorder)`：解包为 `self.leaves_to_search, self.reorder`
- `query(v, n)`：`search(v, n, self.reorder, self.leaves_to_search)`

### 1.1 ann-benchmarks 的 ScaNN 网格特点

### Angular（dot_product）

配置里有 4 组 run groups（`scann1~scann4`），典型是：

- 不同 build 参数组合：
  - `n_leaves` 大致在 `1000~2000`
  - `avq_threshold` 有 `0.15 / 0.2 / 0.55`
  - `dims_per_block` 有 `1 / 2 / 3`
- 每组都给一串 `(leaves_to_search, reorder)` 二元组进行查询 sweep（从非常小到较大预算）。

### Euclidean（squared_l2）

同样是多组 build 参数 + 多组 query 二元组 sweep：

- `n_leaves` 例如 `100/600/2000`
- `avq_threshold` 用 `.nan`（对应 L2 配置）
- `dims_per_block` 例如 `2/4`
- `query_args` 仍是 `(leaves_to_search, reorder)` 联合 sweep。

一句话概括：ann-benchmarks 的 ScaNN 是标准的“build 参数 + query 参数”双层网格搜索，不是单点。

## 2. 与你当前 benchmark 的异同

参考文件：

- `benchmark/common.py`
- `benchmark/benchmark_scann.py`
- `benchmark/benchmark.py`
- `test_scann.py`

### 2.1 相同点

- 都用 `dot_product`（你当前 benchmark 是 IP 路线）。
- 都使用 AH 量化（`score_ah`）并依赖 `leaves_to_search` 与 `reorder` 控制召回-速度。
- 都是多点 sweep，不是单点跑一次。

### 2.2 不同点

- ann-benchmarks：
  - 同时 sweep build 参数（`n_leaves/avq_threshold/dims_per_block`）和 query 参数（`leaves_to_search/reorder`）。
- 你的 benchmark：
  - build 参数基本固定（`num_leaves=2000`，`score_ah(2, aqt=0.2)`）。
  - 通过 `--scann-mode` 切换两种单维 sweep：
    - `reorder` 模式：固定 `leaves_to_search=100`，扫 `reorder` 列表。
    - `leaves` 模式：固定 `reorder=max(reorder_values)`，扫 `leaves_to_search` 列表。
  - 不是二元联合网格（不是 `(leaf, reorder)` 笛卡尔积）。

### 2.3 线程与公平性差异

- 你的 benchmark 在 `benchmark/benchmark.py` 全局限制单线程（`OMP_NUM_THREADS=1` 等），更贴近 ANN-Benchmarks 的“单核比较”原则。
- ann-benchmarks 框架也是以单核公平比较为目标（README principles）。

## 3. 与 Iceberg 的异同

参考文件：

- `Iceberg/config/algorithm.yaml`
- `Iceberg/run.py`
- `Iceberg/test/benchmark_scann.py`

### 3.1 Iceberg 的 ScaNN默认参数

- `algorithm.yaml` 中仅给：
  - `num_leaves: 2000`
  - `num_leaves_to_search: 1000`

### 3.2 Iceberg 的实际测试流程

`benchmark_scann.py` 中：

- build：`score_ah(2, aqt=0.2)`，`reorder(top_k)`。
- search：
  - `base_group = [1, 2, 5, 8, 12, 16, 20, 30, 40, 50, 60, 80, 100, 120, 200, 300, 500, 1000, 2000, 3000]`
  - 对每个 `base` 调 `search_batched_parallel(..., leaves_to_search=base, pre_reorder_num_neighbors=top_k*10)`。

### 3.3 Iceberg 和你 benchmark 的关键差异

- Iceberg 更接近“只扫 leaves_to_search”，而且 `pre_reorder_num_neighbors` 固定为 `top_k*10`。
- 你的 benchmark 可扫 `reorder`（或扫 `leaves`），两种模式更清晰可切换。
- Iceberg 用 `search_batched_parallel`（并行批查），你的 benchmark/test_scann 主要是单 query 循环统计 QPS；二者 QPS 绝对值不可直接横比。
- Iceberg 脚本里 `mode` 参数传入但 ScaNN脚本内部并未真正实现严格的 build/search 逻辑分离（仍是单次调用中 build 后扫 base_group）。

## 4. 与 test_single.py 的异同

参考文件：`test_single.py`

### 4.1 test_single.py 在做什么

- 固定数据集：`glove100`（`nd=1183514, nq=10000, dim=100`）。
- 固定 build 核心：`num_leaves=2000`，`score_ah(2, aqt=0.2)`，`distance=dot_product`。
- 分别对 `topk=1/10/100` 建 searcher，并做双层嵌套 sweep：
  - 外层 `efs`（其实是 `leaves_to_search`）
  - 内层 `val`（`pre_reorder_num_neighbors`）
- 计算 recall + qps，并画图。

### 4.2 与 ann-benchmarks 的关系

- 和 ann-benchmarks 相同：都在扫 `(leaves_to_search, reorder)` 这对查询参数。
- 和 ann-benchmarks 不同：
  - test_single.py 几乎不扫 build 参数（只固定一套）。
  - ann-benchmarks 会在多个 build 参数配置上重复 query sweep。

### 4.3 与你 benchmark 的关系

- test_single.py 是“二维网格 sweep（leaf × reorder）”。
- 你 benchmark 是“单维 sweep（leaf 或 reorder）”，更快、更工程化，但覆盖面比 test_single.py 和 ann-benchmarks 小。

## 5. 四者对比总表

| 维度 | ann-benchmarks | 你的 benchmark | Iceberg | test_single.py |
|---|---|---|---|---|
| Build 参数 sweep | 是（多组 n_leaves/aqt/dpb/dist） | 否（基本固定） | 否（固定） | 否（固定） |
| Query 参数 sweep | 是（leaf,reorder 联合） | 是（但按 mode 单维） | 是（主要 leaf） | 是（leaf,reorder 联合） |
| ScaNN 距离 | dot_product + squared_l2 | dot_product | dot_product | dot_product |
| 线程策略 | 倾向单核公平 | 明确单线程环境 | batched_parallel（并行） | 单 query 循环 |
| 结果可比性（与ANNB） | 基准 | 高（但网格更稀） | 中（模式与线程差异大） | 中高（网格像，但框架不同） |

## 6. 结论（针对“异同点”）

- 如果按“参数覆盖完整度”排序：
  - ann-benchmarks > test_single.py > 你的 benchmark ≈ Iceberg（ScaNN部分）。
- 如果按“工程运行成本/速度”排序：
  - 你的 benchmark（单维 sweep）更省时。
- Iceberg 当前 ScaNN更像固定 build + leaves sweep，且并行批查会让 QPS 口径与单线程流程不一致。
- 你现在的 benchmark 已经比 Iceberg 更接近可控对比；若要进一步贴近 ann-benchmarks，可新增一个 `--scann-mode grid`：对 `(leaves_to_search, reorder)` 做二维笛卡尔扫，并可选再扫 `n_leaves/aqt/dpb`。