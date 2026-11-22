#!/usr/bin/env bash
# CI: build act_ort against ONNX Runtime 1.30.0 and require C++ == PyTorch on a seeded random
# policy and random inputs (checkpoints and data/ are not in the repo). Headers: conda-forge
# onnxruntime-cpp 1.30.0 (sha256 pinned). Library: the onnxruntime 1.30.0 pip wheel.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${PY:-python}"
work="${WORK:-$(mktemp -d)}"
mkdir -p "$work"
pkg=onnxruntime-cpp-1.30.0-h987d1f6_1_cpu
sha=6f042fb15ce643fe954c1eefef3bc3c02a2f302a1b2c76d4bbea8eedaca75e3e

if [ ! -f "$work/$pkg.conda" ]; then
  curl -sSfL -o "$work/$pkg.conda" "https://conda.anaconda.org/conda-forge/linux-64/$pkg.conda"
fi
echo "$sha  $work/$pkg.conda" | sha256sum -c -
mkdir -p "$work/x" "$work/ort"
"$py" -m zipfile -e "$work/$pkg.conda" "$work/x/"
zstd -dc "$work/x/pkg-$pkg.tar.zst" | tar -x -C "$work/ort" include
lib="$("$py" -c 'import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), "capi"))')"
mkdir -p "$work/lib"
ln -sf "$lib/libonnxruntime.so.1.30.0" "$work/lib/libonnxruntime.so.1"

ORT_INCLUDE="$work/ort/include/onnxruntime/core/session" ORT_LIB="$work/lib" "$here/build.sh"
"$py" "$here/bench.py" --synthetic --n 16 --n-time 100 --rounds 1 --work "$work/act"
