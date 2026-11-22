// One LangPolicy step on ONNX Runtime, shared by cpp/act_ort.cpp and the ROS 2 node:
// uint8 HWC RGB image -> [0,1] CHW float, 15-d proprio state -> (s - mean) / std, the exported
// graph (1 intra-op thread, batch 1), chunk * std + mean. Same steps as LangRunner.act
// (so_arm100_vla/policy.py) without the ensembler.
#pragma once
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

namespace act {

constexpr int kH = 96, kW = 128, kState = 15, kEmbed = 384, kK = 10, kAct = 6;

// Normalisation stats from the checkpoint, flat as cpp/bench.py writes stats.f32:
// state mean, state std, action mean, action std.
struct Stats {
  std::array<float, kState> s_mean, s_std;
  std::array<float, kAct> a_mean, a_std;
  static Stats from_flat(const float* p) {
    Stats s;
    std::copy(p, p + kState, s.s_mean.begin());
    std::copy(p + kState, p + 2 * kState, s.s_std.begin());
    std::copy(p + 2 * kState, p + 2 * kState + kAct, s.a_mean.begin());
    std::copy(p + 2 * kState + kAct, p + 2 * kState + 2 * kAct, s.a_std.begin());
    return s;
  }
};
constexpr size_t kStatsLen = 2 * kState + 2 * kAct;

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

}  // namespace act
