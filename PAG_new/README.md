# PIF-PAG

## Requirements

- CMake 3.10 or newer
- A C++20 compiler such as GCC or Clang
- OpenMP
- CPU/compiler support for AVX-512 flags used in `CMakeLists.txt`


## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

After compilation, the executable is:

```bash
./build/PEOs
```

## Input Format

The current program reads raw binary files without headers:

- `data_path`: base vectors stored as contiguous `float32`, shape `[data_size, dim]`
- `query_path`: query vectors stored as contiguous `float32`, shape `[query_size, dim]`
- `truth_path`: ground-truth neighbor ids stored as contiguous `uint32`; the code assumes 100 ids per query

## Usage

```bash
./build/PEOs <data_path> <query_path> <truth_path> <index_path> \
  <data_size> <query_size> <dim> <topk> <efConstruction> <M> <L>
```

Example:

```bash
./build/PEOs ../data/music100/music100_base.bin \
  ../data/music100/music100_query.bin \
  ../data/music100/music100_truth.bin \
  ./music100/index_top100 \
  1000000 10000 100 100 1000 64 32
```

## Parameters

| Parameter | Meaning |
| --- | --- |
| `data_path` | Path to the base vector binary file |
| `query_path` | Path to the query vector binary file |
| `truth_path` | Path to the ground-truth id binary file |
| `index_path` | Directory used to save or load the index |
| `data_size` | Number of base vectors |
| `query_size` | Number of query vectors |
| `dim` | Vector dimension |
| `topk` | Number of nearest neighbors evaluated; use `topk <= 100` with the default truth format |
| `efConstruction` | HNSW construction/search candidate parameter; larger values usually improve recall and increase cost |
| `M` | HNSW graph degree parameter; must be a multiple of 8 |
| `L` | Number of projection levels/subspaces; must be a multiple of 8 |

## Index Behavior

If `index_path` does not exist, the program creates the directory, builds the index, and writes files such as `index.bin`, `info.bin`, `permutation.bin`, and `table.bin`.

If `index_path` already exists, the program loads the saved index and runs search/evaluation directly.

For a fresh rebuild, use a new index directory or remove the old generated index directory first.