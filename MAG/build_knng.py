#!/usr/bin/env python3
"""
Build an approximate kNN graph (L2 metric) for MAG index using FAISS IVFFlat.

MAG loads this as the seed for its L2 subgraph (final_graph_), then refines
it with Link() using L2 distances before separately building the IP subgraph.
IVFFlat is used instead of IVFPQ: no quantization loss gives substantially
better neighbor accuracy, which directly improves MAG index quality.

Binary format expected by MAG's Load_nn_graph():
  For each of N points: [k: uint32][id_0: uint32] ... [id_{k-1}: uint32]

Usage:
  python3 MAG/build_knng.py [db_path] [out_path] [dim] [k]
  (defaults to Music100 with k=100)
"""
import sys
import os
import numpy as np


def main():
    if len(sys.argv) >= 5:
        db_path  = sys.argv[1]
        out_path = sys.argv[2]
        dim      = int(sys.argv[3])
        k        = int(sys.argv[4])
    else:
        db_path  = "data/database_music100.bin"
        out_path = "MAG/music100.knng"
        dim      = 100
        k        = 100

    print(f"Building kNN graph: {db_path} → {out_path}, dim={dim}, k={k}", flush=True)

    # Load database (raw float32, no header)
    n = os.path.getsize(db_path) // (dim * 4)
    data = np.fromfile(db_path, dtype=np.float32).reshape(n, dim)
    print(f"Loaded {n} vectors of dim {dim}", flush=True)

    import faiss

    nthreads = int(os.environ.get("BUILD_NTHREADS", os.cpu_count() or 4))
    faiss.omp_set_num_threads(nthreads)
    print(f"Using {nthreads} threads", flush=True)

    # IVFFlat: exact distances within each cluster (no quantization loss).
    # L2 metric matches MAG's internal L2 subgraph construction.
    nlist = min(4096, max(256, int(np.sqrt(n))))
    train_n = min(n, max(nlist * 40, 200_000))
    print(f"Training IVFFlat (nlist={nlist}, train_n={train_n})...", flush=True)
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(data[:train_n])
    index.add(data)
    # Search ~25% of clusters: good recall/speed trade-off for graph building.
    index.nprobe = max(64, nlist // 4)
    print(f"Index built, nprobe={index.nprobe}", flush=True)

    # Precompute the per-row k header (same for every row)
    k_header = np.array([k], dtype=np.uint32).tobytes()

    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)

    BATCH = 50_000
    print(f"Searching {k+1}-NN for all {n} vectors...", flush=True)
    with open(out_path, 'wb') as f:
        for start in range(0, n, BATCH):
            end = min(start + BATCH, n)
            _, I = index.search(data[start:end], k + 1)
            for bi in range(end - start):
                self_id = start + bi
                nn = [int(j) for j in I[bi] if j != self_id and j >= 0][:k]
                while len(nn) < k:
                    nn.append(0)
                f.write(k_header)
                f.write(np.array(nn, dtype=np.uint32).tobytes())
            if start % (BATCH * 10) == 0:
                print(f"  {end}/{n}", flush=True)

    print(f"Saved {out_path}  ({n} pts, k={k})", flush=True)


if __name__ == '__main__':
    main()
