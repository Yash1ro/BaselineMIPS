#!/usr/bin/env bash
set -euo pipefail

# One-click build script for baseline algorithms.
# Usage:
#   bash build_all.sh
#   bash build_all.sh --clean
#   bash build_all.sh --jobs 16
#   bash build_all.sh --clean --jobs 16

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOBS="$(nproc)"
CLEAN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean)
      CLEAN=1
      shift
      ;;
    --jobs)
      if [[ $# -lt 2 ]]; then
        echo "[ERROR] --jobs requires a value" >&2
        exit 1
      fi
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash build_all.sh [--clean] [--jobs N]

Options:
  --clean     Remove build outputs before compiling
  --jobs N    Number of parallel jobs (default: nproc)
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

build_cmake_project() {
  local name="$1"
  local dir="$2"
  local build_dir="$dir/build"

  if [[ ! -d "$dir" ]]; then
    echo "[WARN] Skip $name: directory not found: $dir"
    return 0
  fi

  if [[ "$CLEAN" -eq 1 && -d "$build_dir" ]]; then
    echo "[INFO] Cleaning $name build directory"
    rm -rf "$build_dir"
  fi

  echo "[INFO] Configuring $name"
  cmake -S "$dir" -B "$build_dir" -DCMAKE_BUILD_TYPE=Release

  echo "[INFO] Building $name with -j$JOBS"
  cmake --build "$build_dir" -j"$JOBS"
}

build_mobius() {
  local dir="$ROOT_DIR/mobius"
  if [[ ! -d "$dir" ]]; then
    echo "[WARN] Skip mobius: directory not found: $dir"
    return 0
  fi

  if [[ "$CLEAN" -eq 1 ]]; then
    echo "[INFO] Cleaning mobius outputs"
    rm -f "$dir/mobius" "$dir/mobius.so" "$dir/mobius_single"
  fi

  echo "[INFO] Building mobius with -j$JOBS"
  make -C "$dir" -j"$JOBS"
}

echo "[INFO] Root: $ROOT_DIR"
echo "[INFO] Jobs: $JOBS"
echo "[INFO] Clean: $CLEAN"

build_cmake_project "ip-nsw" "$ROOT_DIR/ip-nsw"
build_mobius
build_cmake_project "MAG" "$ROOT_DIR/MAG"
build_cmake_project "PAG" "$ROOT_DIR/PAG"

echo "[INFO] Build completed successfully"
