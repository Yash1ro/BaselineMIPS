#!/usr/bin/env python3
import os
import sys
import requests
import h5py
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

DEFAULT_URL = "http://ann-benchmarks.com/glove-200-angular.hdf5"
DEFAULT_HDF5 = "glove-200-angular.hdf5"


def download_file(url: str, out_path: str) -> None:
    if os.path.exists(out_path):
        print(f"[skip] HDF5 already exists: {out_path}")
        return

    print(f"[download] {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with open(out_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"[saved] {out_path}")


def save_bin(array: np.ndarray, out_path: str, dtype) -> None:
    arr = np.asarray(array, dtype=dtype)
    arr.tofile(out_path)


def compute_truth_from_base_query(
    base: np.ndarray,
    query: np.ndarray,
    top_k: int = 100,
    metric: str = "ip",
) -> np.ndarray:
    """Compute exact top-k neighbors from base/query, return int32 ids."""
    if base.ndim != 2 or query.ndim != 2:
        raise ValueError("base/query must be 2D arrays")
    if base.shape[1] != query.shape[1]:
        raise ValueError(
            f"dim mismatch: base dim={base.shape[1]}, query dim={query.shape[1]}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    top_k = min(top_k, base.shape[0])
    base = np.asarray(base, dtype=np.float32, order="C")
    query = np.asarray(query, dtype=np.float32, order="C")

    if faiss is not None:
        if metric == "ip":
            index = faiss.IndexFlatIP(base.shape[1])
        elif metric == "l2":
            index = faiss.IndexFlatL2(base.shape[1])
        else:
            raise ValueError("metric must be 'ip' or 'l2'")
        index.add(base)
        _, ids = index.search(query, top_k)
        return ids.astype(np.int32, copy=False)

    raise RuntimeError(
        "faiss is required to compute ground truth in this script. "
        "Please install faiss-cpu/faiss-gpu."
    )


def main():
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "."
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 100
    metric = sys.argv[3].lower() if len(sys.argv) >= 4 else "ip"
    os.makedirs(out_dir, exist_ok=True)

    hdf5_path = os.path.join(out_dir, DEFAULT_HDF5)
    data_out = os.path.join(out_dir, "glove200_base.bin")
    query_out = os.path.join(out_dir, "glove200_query.bin")
    truth_out = os.path.join(out_dir, "glove200_truth.bin")

    if metric not in ("ip", "l2"):
        raise ValueError("metric must be 'ip' or 'l2'")

    download_file(DEFAULT_URL, hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        train = f["train"][:].astype(np.float32, copy=False)
        test = f["test"][:].astype(np.float32, copy=False)

    print(
        f"[truth] computing exact top-{top_k} with metric={metric} "
        f"from base/query..."
    )
    truth = compute_truth_from_base_query(train, test, top_k=top_k, metric=metric)

    save_bin(train, data_out, np.float32)
    save_bin(test, query_out, np.float32)
    save_bin(truth, truth_out, np.int32)

    print("Done.")
    print(f"train     -> {data_out}   shape={train.shape} dtype={train.dtype}")
    print(f"test      -> {query_out}  shape={test.shape} dtype={test.dtype}")
    print(f"truth     -> {truth_out}  shape={truth.shape} dtype={truth.dtype}")

    print("\nUsage:")
    print("  python benchmark/tools/generate_glove.py [out_dir] [top_k] [metric]")
    print("  metric: ip (default) or l2")


if __name__ == "__main__":
    main()