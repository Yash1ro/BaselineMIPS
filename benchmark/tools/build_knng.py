#!/usr/bin/env python3
"""
Build an approximate kNN graph for MAG index using FAISS.
Saves in the binary format expected by MAG's Load_nn_graph():
  For each point: [k: uint32, n0: uint32, ..., n_{k-1}: uint32]

Usage:
  python3 benchmark/tools/build_knng.py [database_path] [output_path] [dim] [k]
  (defaults to Music100 with k=100)
"""
import sys
import os
import struct
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

    # Allow multiple threads for FAISS (override baseline.py's OMP=1 restriction)
    nthreads = int(os.environ.get("BUILD_NTHREADS", os.cpu_count() or 4))
    faiss.omp_set_num_threads(nthreads)
    print(f"Using {nthreads} threads for FAISS", flush=True)

    # Build IVFPQ index for approximate kNN (as described in MAG paper stage-1)
    nlist = min(4096, max(256, int(np.sqrt(n))))
    train_n = min(n, max(nlist * 40, 100_000))
    # Choose PQ subquantizers m that divides dim for IndexIVFPQ.
    m = next((cand for cand in (16, 12, 10, 8, 6, 5, 4, 3, 2, 1) if dim % cand == 0), 1)
    nbits = 8
    print(f"Training IVFPQ (nlist={nlist}, m={m}, nbits={nbits}, train_n={train_n})...", flush=True)
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
    index.train(data[:train_n])
    index.add(data)
    index.nprobe = min(128, max(32, nlist // 8))
    print(f"Index built, nprobe={index.nprobe}", flush=True)

    # Search in batches: k+1 neighbors, then drop self
    BATCH = 50_000
    neighbors = np.zeros((n, k), dtype=np.uint32)
    print(f"Searching {k+1}-NN for all {n} vectors...", flush=True)
    for start in range(0, n, BATCH):
        end = min(start + BATCH, n)
        _, I = index.search(data[start:end], k + 1)
        for bi in range(end - start):
            nn = [int(j) for j in I[bi] if j != start + bi and j >= 0][:k]
            while len(nn) < k:
                nn.append(0)
            neighbors[start + bi] = nn
        if start % (BATCH * 10) == 0:
            print(f"  {end}/{n}", flush=True)
    print(f"Search done.", flush=True)

    # Write binary .knng file
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    k_bytes = struct.pack('<I', k)
    fmt = f'<{k}I'
    with open(out_path, 'wb') as f:
        for i in range(n):
            f.write(k_bytes)
            f.write(struct.pack(fmt, *neighbors[i]))

    print(f"Saved {out_path}  ({n} pts, k={k})", flush=True)

if __name__ == '__main__':
    main()
