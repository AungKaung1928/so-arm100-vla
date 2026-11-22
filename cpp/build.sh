#!/usr/bin/env bash
# Build act_ort (policy step) against an ONNX Runtime with the same version as the Python wheel.
#   ORT_INCLUDE  dir holding onnxruntime_cxx_api.h
#   ORT_LIB      dir holding libonnxruntime.so.1 (the wheel's onnxruntime/capi, symlinked)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
: "${ORT_INCLUDE:?set ORT_INCLUDE}" "${ORT_LIB:?set ORT_LIB}"
g++ -O2 -std=c++17 -Wall -Wextra -Werror -I "$ORT_INCLUDE" "$here/act_ort.cpp" \
  -L "$ORT_LIB" -l:libonnxruntime.so.1 -Wl,-rpath,"$ORT_LIB" -o "$here/act_ort"
echo "built $here/act_ort"
