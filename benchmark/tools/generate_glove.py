#!/usr/bin/env python3
import os
import sys
import requests
import h5py
import numpy as np

DEFAULT_URL = "http://ann-benchmarks.com/glove-100-angular.hdf5"
DEFAULT_HDF5 = "glove-100-angular.hdf5"


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


def main():
    out_dir = sys.argv[1] if len(sys.argv) >= 2 else "."
    os.makedirs(out_dir, exist_ok=True)

    hdf5_path = os.path.join(out_dir, DEFAULT_HDF5)
    data_out = os.path.join(out_dir, "glove100_data.bin")
    query_out = os.path.join(out_dir, "glove100_query.bin")
    truth_out = os.path.join(out_dir, "glove100_truth.bin")

    download_file(DEFAULT_URL, hdf5_path)

    with h5py.File(hdf5_path, "r") as f:
        train = f["train"][:].astype(np.float32, copy=False)
        test = f["test"][:].astype(np.float32, copy=False)
        neighbors = f["neighbors"][:].astype(np.int32, copy=False)

    save_bin(train, data_out, np.float32)
    save_bin(test, query_out, np.float32)
    save_bin(neighbors, truth_out, np.int32)

    print("Done.")
    print(f"train     -> {data_out}   shape={train.shape} dtype={train.dtype}")
    print(f"test      -> {query_out}  shape={test.shape} dtype={test.dtype}")
    print(f"neighbors -> {truth_out}  shape={neighbors.shape} dtype={neighbors.dtype}")


if __name__ == "__main__":
    main()