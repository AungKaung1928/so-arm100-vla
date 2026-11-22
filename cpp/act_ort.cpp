// One LangPolicy step in C++, as LangRunner.act does it minus the ensembler:
// uint8 HWC image -> [0,1] CHW float, state -> (s - mean) / std, ONNX Runtime (1 intra-op
// thread, batch 1), chunk * std + mean. Checked against PyTorch reference chunks.
// Inputs written by cpp/bench.py into <dir>: images.u8 (N x 96 x 128 x 3), states.f32 (N x 15),
// lang.f32 (N x 384), stats.f32 (state mean, state std, action mean, action std),
// ref.f32 (N x 10 x 6, denormalised chunks from PyTorch).
// Usage: act_ort <policy.onnx> <dir> [n_time]. Prints one JSON line.
#include <onnxruntime_cxx_api.h>

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

constexpr int kH = 96, kW = 128, kState = 15, kEmbed = 384, kK = 10, kAct = 6;

template <typename T>
std::vector<T> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
  std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

struct Stats { const float *s_mean, *s_std, *a_mean, *a_std; };

class Step {
 public:
  Step(const char* model, const Stats& st) : env_(ORT_LOGGING_LEVEL_WARNING, "act"), st_(st) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetInterOpNumThreads(1);
    sess_ = Ort::Session(env_, model, so);
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    in_[0] = Ort::Value::CreateTensor<float>(mem, img_.data(), img_.size(), kImgShape.data(), 4);
    in_[1] = Ort::Value::CreateTensor<float>(mem, state_.data(), state_.size(), kStateShape.data(), 2);
    in_[2] = Ort::Value::CreateTensor<float>(mem, lang_.data(), lang_.size(), kLangShape.data(), 2);
    out_ = Ort::Value::CreateTensor<float>(mem, chunk_.data(), chunk_.size(), kChunkShape.data(), 3);
  }

  // Writes kK x kAct denormalised joint targets into act.
  void run(const uint8_t* img, const float* state, const float* lang, float* act) {
    constexpr int hw = kH * kW;
    for (int p = 0; p < hw; ++p)
      for (int c = 0; c < 3; ++c) img_[c * hw + p] = img[p * 3 + c] / 255.0f;
    for (int i = 0; i < kState; ++i) state_[i] = (state[i] - st_.s_mean[i]) / st_.s_std[i];
    std::copy(lang, lang + kEmbed, lang_.begin());
    sess_.Run(Ort::RunOptions{nullptr}, kIn, in_.data(), 3, kOut, &out_, 1);
    for (int i = 0; i < kK * kAct; ++i) act[i] = chunk_[i] * st_.a_std[i % kAct] + st_.a_mean[i % kAct];
  }

 private:
  static constexpr std::array<int64_t, 4> kImgShape{1, 3, kH, kW};
  static constexpr std::array<int64_t, 2> kStateShape{1, kState};
  static constexpr std::array<int64_t, 2> kLangShape{1, kEmbed};
  static constexpr std::array<int64_t, 3> kChunkShape{1, kK, kAct};
  static constexpr const char* kIn[3] = {"image", "state", "lang"};
  static constexpr const char* kOut[1] = {"chunk"};
  Ort::Env env_;
  Stats st_;
  Ort::Session sess_{nullptr};
  std::vector<float> img_ = std::vector<float>(3 * kH * kW), state_ = std::vector<float>(kState),
                     lang_ = std::vector<float>(kEmbed), chunk_ = std::vector<float>(kK * kAct);
  std::array<Ort::Value, 3> in_{Ort::Value{nullptr}, Ort::Value{nullptr}, Ort::Value{nullptr}};
  Ort::Value out_{nullptr};
};

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
  if (st.size() != 2 * kState + 2 * kAct || imgs.size() != n * kH * kW * 3 ||
      langs.size() != n * kEmbed || ref.size() != n * kK * kAct || n == 0 || n_time < 1) {
    std::fprintf(stderr, "input sizes do not match N = %zu\n", n);
    return 2;
  }
  Step step(argv[1], {st.data(), st.data() + kState, st.data() + 2 * kState, st.data() + 2 * kState + kAct});

  std::vector<float> act(kK * kAct);
  double max_diff = 0.0;
  for (size_t i = 0; i < n; ++i) {
    step.run(&imgs[i * kH * kW * 3], &states[i * kState], &langs[i * kEmbed], act.data());
    for (int j = 0; j < kK * kAct; ++j)
      max_diff = std::max(max_diff, static_cast<double>(std::fabs(act[j] - ref[i * kK * kAct + j])));
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
