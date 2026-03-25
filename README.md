# BaselineMIPS

## 更新日志

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
