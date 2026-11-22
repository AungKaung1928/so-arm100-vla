// The policy step (act_step.hpp) checked against PyTorch reference chunks and timed.
// Inputs written by cpp/bench.py into <dir>: images.u8 (N x 96 x 128 x 3), states.f32 (N x 15),
// lang.f32 (N x 384), stats.f32 (state mean, state std, action mean, action std),
// ref.f32 (N x 10 x 6, denormalised chunks from PyTorch).
// Usage: act_ort <policy.onnx> <dir> [n_time]. Prints one JSON line.
#include "act_step.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

using act::kH, act::kW, act::kState, act::kEmbed, act::kK, act::kAct;

template <typename T>
std::vector<T> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
  std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

double pct(std::vector<double> v, double q) {
  std::sort(v.begin(), v.end());
  const double r = q / 100.0 * static_cast<double>(v.size() - 1);  // numpy 'linear'
  const size_t lo = static_cast<size_t>(r), hi = std::min(lo + 1, v.size() - 1);
  return v[lo] + (v[hi] - v[lo]) * (r - static_cast<double>(lo));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) { std::fprintf(stderr, "usage: %s policy.onnx dir [n_time]\n", argv[0]); return 2; }
  const std::string dir = argv[2];
  const int n_time = argc > 3 ? std::atoi(argv[3]) : 500;
  const auto imgs = read_all<uint8_t>(dir + "/images.u8");
  const auto states = read_all<float>(dir + "/states.f32");
  const auto langs = read_all<float>(dir + "/lang.f32");
  const auto ref = read_all<float>(dir + "/ref.f32");
  const auto st = read_all<float>(dir + "/stats.f32");
  const size_t n = states.size() / kState;
  if (st.size() != act::kStatsLen || imgs.size() != n * kH * kW * 3 ||
      langs.size() != n * kEmbed || ref.size() != n * kK * kAct || n == 0 || n_time < 1) {
    std::fprintf(stderr, "input sizes do not match N = %zu\n", n);
    return 2;
  }
  act::Step step(argv[1], act::Stats::from_flat(st.data()));

  std::vector<float> act(kK * kAct);
  double max_diff = 0.0;
  for (size_t i = 0; i < n; ++i) {
    step.run(&imgs[i * kH * kW * 3], &states[i * kState], &langs[i * kEmbed], act.data());
    for (int j = 0; j < kK * kAct; ++j) {
      const double d = std::fabs(act[j] - ref[i * kK * kAct + j]);
      if (!std::isfinite(d)) {  // std::max would drop a NaN and report parity
        std::fprintf(stderr, "non-finite action or reference at sample %zu, index %d\n", i, j);
        return 1;
      }
      max_diff = std::max(max_diff, d);
    }
  }

  std::vector<double> ms(n_time);
  for (int t = 0; t < n_time; ++t) {
    const size_t i = static_cast<size_t>(t) % n;
    const auto t0 = std::chrono::steady_clock::now();
    step.run(&imgs[i * kH * kW * 3], &states[i * kState], &langs[i * kEmbed], act.data());
    ms[t] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
  std::printf("{\"n\": %zu, \"max_abs_diff\": %.3g, \"step_ms_p50\": %.4f, \"step_ms_p99\": %.4f, \"ort_version\": \"%s\"}\n",
              n, max_diff, pct(ms, 50), pct(ms, 99), OrtGetApiBase()->GetVersionString());
  return 0;
}
