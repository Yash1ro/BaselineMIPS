#!/usr/bin/env python3
import argparse
import os
import struct
import sys

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        # Fallback when tqdm is unavailable.
        return iterable if iterable is not None else range(total or 0)


def infer_header_mode(path: str, mode: str, count: int | None, in_dim: int) -> tuple[str, int]:
    size = os.path.getsize(path)
    row_bytes = in_dim * 4

    if mode == "none":
        if count is None:
            if size % row_bytes != 0:
                raise ValueError(f"{path}: size {size} is not divisible by row_bytes {row_bytes}")
            count = size // row_bytes
        return "none", count

    if mode == "fbin":
        if count is None:
            with open(path, "rb") as f:
                hdr = f.read(8)
            if len(hdr) != 8:
                raise ValueError(f"{path}: cannot read 8-byte fbin header")
            rows, cols = struct.unpack("II", hdr)
            if cols != in_dim:
                raise ValueError(f"{path}: fbin cols={cols}, expected in_dim={in_dim}")
            count = rows
        expected = 8 + count * row_bytes
        if size != expected:
            raise ValueError(f"{path}: size {size}, expected {expected} for fbin")
        return "fbin", count

    # auto
    if count is not None:
        expected_no_header = count * row_bytes
        expected_fbin = 8 + expected_no_header
        if size == expected_no_header:
            return "none", count
        if size == expected_fbin:
            return "fbin", count
        raise ValueError(
            f"{path}: size {size} does not match none ({expected_no_header}) or fbin ({expected_fbin})"
        )

    # count unknown: try fbin first
    header_mode = "none"
    inferred_count = None
    if size >= 8:
        with open(path, "rb") as f:
            hdr = f.read(8)
        rows, cols = struct.unpack("II", hdr)
        if cols == in_dim and size == 8 + rows * row_bytes:
            header_mode = "fbin"
            inferred_count = rows

    if inferred_count is None:
        if size % row_bytes != 0:
            raise ValueError(f"{path}: size {size} is not divisible by row_bytes {row_bytes}")
        inferred_count = size // row_bytes

    return header_mode, inferred_count


def convert(path_in: str, path_out: str, out_dim: int, label_cols: int, count: int | None, mode: str, batch_rows: int) -> int:
    if out_dim <= 0:
        raise ValueError("out_dim must be > 0")
    if label_cols < 0:
        raise ValueError("label_cols must be >= 0")
    if batch_rows <= 0:
        raise ValueError("batch_rows must be > 0")

    in_dim = out_dim + label_cols
    row_bytes_in = in_dim * 4
    row_bytes_out = out_dim * 4
    skip_bytes = label_cols * 4

    header_mode, count = infer_header_mode(path_in, mode, count, in_dim)

    expected_out_size = count * row_bytes_out

    need_convert = True
    if os.path.exists(path_out):
        out_size = os.path.getsize(path_out)
        if out_size == expected_out_size:
            print(f"[convert.py] output exists, skip conversion: {path_out}")
            need_convert = False
        else:
            print(
                f"[convert.py] output exists but size mismatch, rebuilding: "
                f"{out_size} != {expected_out_size}"
            )

    if need_convert:
        with open(path_in, "rb") as fin, open(path_out, "wb") as fout:
            if header_mode == "fbin":
                fin.seek(8)

            rows_left = count
            pbar = tqdm(total=count, desc="Converting", unit="rows")
            while rows_left > 0:
                cur_rows = min(rows_left, batch_rows)
                chunk = fin.read(cur_rows * row_bytes_in)
                if len(chunk) != cur_rows * row_bytes_in:
                    raise RuntimeError(
                        f"short read: got {len(chunk)} bytes, expected {cur_rows * row_bytes_in}"
                    )

                # Slice each row to remove leading label columns.
                for i in range(cur_rows):
                    st = i * row_bytes_in + skip_bytes
                    ed = st + row_bytes_out
                    fout.write(chunk[st:ed])

                rows_left -= cur_rows
                pbar.update(cur_rows)
            pbar.close()

            extra = fin.read(1)
            if extra:
                raise RuntimeError("trailing bytes detected after expected rows")

        out_size = os.path.getsize(path_out)
        if out_size != expected_out_size:
            raise RuntimeError(f"output size mismatch: got {out_size}, expected {expected_out_size}")

    # Always verify output against source rows after dropping labels.
    with open(path_in, "rb") as fin, open(path_out, "rb") as fout:
        if header_mode == "fbin":
            fin.seek(8)

        rows_left = count
        checked = 0
        pbar = tqdm(total=count, desc="Verifying", unit="rows")
        while rows_left > 0:
            cur_rows = min(rows_left, batch_rows)
            src_chunk = fin.read(cur_rows * row_bytes_in)
            out_chunk = fout.read(cur_rows * row_bytes_out)

            if len(src_chunk) != cur_rows * row_bytes_in:
                raise RuntimeError(
                    f"short read in source during verify: got {len(src_chunk)} bytes, "
                    f"expected {cur_rows * row_bytes_in}"
                )
            if len(out_chunk) != cur_rows * row_bytes_out:
                raise RuntimeError(
                    f"short read in output during verify: got {len(out_chunk)} bytes, "
                    f"expected {cur_rows * row_bytes_out}"
                )

            for i in range(cur_rows):
                st = i * row_bytes_in + skip_bytes
                ed = st + row_bytes_out
                src_slice = src_chunk[st:ed]
                out_st = i * row_bytes_out
                out_ed = out_st + row_bytes_out
                out_slice = out_chunk[out_st:out_ed]
                if src_slice != out_slice:
                    row_id = checked + i
                    raise RuntimeError(
                        f"verification failed at row {row_id}: output differs from source after label drop"
                    )

            rows_left -= cur_rows
            checked += cur_rows
            pbar.update(cur_rows)
        pbar.close()

        extra_out = fout.read(1)
        if extra_out:
            raise RuntimeError("output has trailing bytes after expected rows")

    print(f"[convert.py] verification passed: {count} rows checked")

    return count


def infer_out_dim_from_path(path_in: str, path_out: str, label_cols: int) -> int | None:
    """Infer output dimension from common dataset naming conventions."""
    p = f"{path_in} {path_out}".lower()
    if "dinov2" in p:
        return 768
    if "book_corpus" in p or "stella" in p:
        return 1024
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert labeled float32 vectors by dropping leading label columns."
    )
    parser.add_argument("input", help="Input vector file")
    parser.add_argument("output", help="Output vector file")
    parser.add_argument(
        "--out-dim",
        type=int,
        default=None,
        help="Dimension after removing labels. If omitted, infer from path for known datasets.",
    )
    parser.add_argument("--label-cols", type=int, default=1, help="Number of leading float32/int32 columns to drop")
    parser.add_argument("--count", type=int, default=None, help="Number of vectors; infer from file if omitted")
    parser.add_argument(
        "--header",
        choices=["auto", "none", "fbin"],
        default="auto",
        help="Input header mode",
    )
    parser.add_argument("--batch-rows", type=int, default=4096, help="Rows per IO batch")
    args = parser.parse_args()

    out_dim = args.out_dim
    if out_dim is None:
        out_dim = infer_out_dim_from_path(args.input, args.output, args.label_cols)
        if out_dim is None:
            raise ValueError(
                "cannot infer --out-dim from paths; provide --out-dim explicitly "
                "(known auto-infer: dinov2->768, book_corpus/stella->1024)"
            )

    rows = convert(
        path_in=args.input,
        path_out=args.output,
        out_dim=out_dim,
        label_cols=args.label_cols,
        count=args.count,
        mode=args.header,
        batch_rows=args.batch_rows,
    )

    print(
        f"Converted {rows} rows: {args.input} -> {args.output} "
        f"(drop {args.label_cols} leading cols, out_dim={out_dim})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[convert.py] ERROR: {exc}", file=sys.stderr)
        raise
