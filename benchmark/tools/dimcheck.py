#!/usr/bin/env python3
import argparse
import os
import subprocess
import re
import struct
import tempfile
from pathlib import Path


DATASET_DIM_HINTS = {
	"book_corpus": 1024,
	"dinov2": 768,
	"glove": 100,
	"music": 100,
}


def guess_dim_from_filename(path: str) -> int | None:
	name = os.path.basename(path)
	m = re.search(r"(\d+)(?=_[^.]+\.bin$)", name)
	return int(m.group(1)) if m else None


def guess_dim_from_path(path: str) -> int | None:
	lower = path.lower()
	for key, dim in DATASET_DIM_HINTS.items():
		if key in lower:
			return dim
	return None


def guess_dim_from_num(size: int, num: int | None) -> int | None:
	if num is None or num <= 0:
		return None
	den = num * 4
	if size % den != 0:
		return None
	dim = size // den
	return dim if dim > 0 else None


def strip_label_in_place(path: str, dim: int) -> tuple[int, int, str]:
	"""Convert labeled records to raw float32 vectors and replace original file."""
	tool_dir = Path(__file__).resolve().parent
	convert_py = tool_dir / "convert.py"
	if not convert_py.exists():
		raise FileNotFoundError(f"convert.py not found: {convert_py}")

	fd, tmp_path = tempfile.mkstemp(prefix="nolabel_", suffix=".bin", dir=os.path.dirname(path) or ".")
	os.close(fd)
	try:
		cmd = [
			"python3",
			str(convert_py),
			path,
			tmp_path,
			"--out-dim",
			str(dim),
			"--label-cols",
			"1",
			"--header",
			"auto",
		]
		subprocess.run(cmd, check=True)

		# Keep the original filename: replace source with converted tmp file.
		os.replace(tmp_path, path)
		size = os.path.getsize(path)
		row_bytes = dim * 4
		if size % row_bytes != 0:
			raise ValueError(
				f"converted file size mismatch: size={size}, dim={dim}, row_bytes={row_bytes}"
			)
		return size // row_bytes, dim, "raw-float32"
	finally:
		if os.path.exists(tmp_path):
			os.remove(tmp_path)


def infer_num_dim(path: str, raw_dim: int | None, num: int | None) -> tuple[int, int, str]:
	size = os.path.getsize(path)
	if size < 4:
		raise ValueError(f"file too small: {size} bytes")

	# Try fbin first: 8-byte header of uint32 (rows, cols).
	if size >= 8:
		with open(path, "rb") as f:
			hdr = f.read(8)
		rows, cols = struct.unpack("II", hdr)
		if rows > 0 and cols > 0 and size == 8 + rows * cols * 4:
			return rows, cols, "fbin"

	dim = raw_dim
	if dim is None:
		dim = guess_dim_from_num(size, num)
	if dim is None:
		dim = guess_dim_from_path(path)
	if dim is None:
		dim = guess_dim_from_filename(path)
	if dim is None or dim <= 0:
		raise ValueError(
			"cannot infer dim for raw float32 bin. Please provide --dim or --num, "
			"or use a filename like xxx100_base.bin"
		)

	row_bytes = dim * 4
	if size % row_bytes != 0:
		# Some datasets are stored as labeled records:
		# [uint32 label][float32 * dim] per vector.
		labeled_row_bytes = (dim + 1) * 4
		if size % labeled_row_bytes == 0:
			return strip_label_in_place(path, dim)
		raise ValueError(
			f"raw float32 size mismatch: size={size}, dim={dim}, row_bytes={row_bytes}"
		)

	return size // row_bytes, dim, "raw-float32"


def main() -> None:
	parser = argparse.ArgumentParser(description="Print dataset num and dim from .bin file")
	parser.add_argument("path", help="Path to dataset .bin file")
	parser.add_argument(
		"--dim",
		type=int,
		default=None,
		help="Dimension for raw float32 bin without header (optional if filename contains dim)",
	)
	parser.add_argument(
		"--num",
		type=int,
		default=None,
		help="Vector count for raw float32 bin; can infer dim when --dim is not set",
	)
	args = parser.parse_args()

	num, dim, mode = infer_num_dim(args.path, args.dim, args.num)
	print(f"path: {args.path}")
	print(f"mode: {mode}")
	print(f"num: {num}")
	print(f"dim: {dim}")


if __name__ == "__main__":
	main()