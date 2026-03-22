#!/usr/bin/env python3
import argparse
import os
import re
import struct


DATASET_DIM_HINTS = {
	"book_corpus": 1024,
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
		dim = guess_dim_from_filename(path)
	if dim is None:
		dim = guess_dim_from_path(path)
	if dim is None or dim <= 0:
		raise ValueError(
			"cannot infer dim for raw float32 bin. Please provide --dim or --num, "
			"or use a filename like xxx100_base.bin"
		)

	row_bytes = dim * 4
	if size % row_bytes != 0:
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