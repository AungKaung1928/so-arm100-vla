// The node against PyTorch: recorded frames, states and reference chunks from cpp/bench.py go
// in through real topics and TF; the published trajectory must equal the reference chunk.
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/transform_stamped.hpp"
#include "so_arm100_vla_ros/act_policy_node.hpp"
#include "tf2_ros/static_transform_broadcaster.h"

using namespace std::chrono_literals;
using so_arm100_vla_ros::ActPolicyNode;

namespace {

const std::string kData = ACT_TEST_DATA;

template <typename T>
std::vector<T> read_all(const std::string& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  std::vector<T> v(static_cast<size_t>(f.tellg()) / sizeof(T));
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(T)));
  return v;
}

class NodeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (kData.empty()) GTEST_SKIP() << "ACT_TEST_DATA not set (cpp/bench.py work dir)";
    rclcpp::init(0, nullptr);
    rclcpp::NodeOptions o;
    o.parameter_overrides({{"model_path", kData + "/policy.onnx"},
                           {"stats_path", kData + "/stats.f32"},
                           {"lang_path", kData + "/lang0.f32"}});
    // lang0.f32: the first sample's embedding, the instruction fixed for the test episode
    auto lang = read_all<float>(kData + "/lang.f32");
    std::ofstream(kData + "/lang0.f32", std::ios::binary)
      .write(reinterpret_cast<const char*>(lang.data()), act::kEmbed * sizeof(float));
    node_ = std::make_shared<ActPolicyNode>(o);
    io_ = std::make_shared<rclcpp::Node>("io");
    img_pub_ = io_->create_publisher<sensor_msgs::msg::Image>("image", rclcpp::SensorDataQoS());
    js_pub_ = io_->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    tf_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(io_);
    sub_ = io_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
      "joint_trajectory", 10, [this](trajectory_msgs::msg::JointTrajectory::ConstSharedPtr m) { got_.push_back(*m); });
    // created after rclcpp::init: an executor needs the default context
    exec_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    exec_->add_node(node_);
    exec_->add_node(io_);
    imgs_ = read_all<uint8_t>(kData + "/images.u8");
    states_ = read_all<float>(kData + "/states.f32");
    ref_ = read_all<float>(kData + "/ref.f32");
  }
  void TearDown() override {
    exec_.reset();
    sub_.reset();
    tf_.reset();
    js_pub_.reset();
    img_pub_.reset();
    io_.reset();
    node_.reset();
    if (rclcpp::ok()) rclcpp::shutdown();
  }

  void spin_for(std::chrono::milliseconds d) {
    const auto end = std::chrono::steady_clock::now() + d;
    while (std::chrono::steady_clock::now() < end) exec_->spin_some(10ms);
  }

  // Joint states + ee TF of sample i (joint names shuffled: the node must map by name).
  // nan_velocity: what joint_state_broadcaster sends for a joint without a velocity interface.
  void send_state(size_t i, bool nan_velocity = false) {
    const float* s = &states_[i * act::kState];
    sensor_msgs::msg::JointState js;
    const std::vector<std::string> names{"Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"};
    for (int j = act::kAct - 1; j >= 0; --j) {
      js.name.push_back(names[j]);
      js.position.push_back(s[j]);
      js.velocity.push_back(nan_velocity ? std::nan("") : s[act::kAct + j]);
    }
    js_pub_->publish(js);
    geometry_msgs::msg::TransformStamped t;
    t.header.stamp = io_->now();
    t.header.frame_id = "world";
    t.child_frame_id = "ee";
    t.transform.translation.x = s[12];
    t.transform.translation.y = s[13];
    t.transform.translation.z = s[14];
    t.transform.rotation.w = 1.0;
    tf_->sendTransform(t);
  }

  sensor_msgs::msg::Image image(size_t i, const std::string& enc = "rgb8") {
    sensor_msgs::msg::Image m;
    m.header.stamp = rclcpp::Time(42, static_cast<uint32_t>(i));
    m.encoding = enc;
    m.width = act::kW;
    m.height = act::kH;
    m.step = act::kW * 3 + 4;  // padded rows: the node must drop the padding
    m.data.assign(static_cast<size_t>(m.step) * act::kH, 0);
    const size_t row = act::kW * 3;
    for (int y = 0; y < act::kH; ++y)
      std::copy_n(&imgs_[i * act::kH * row + y * row], row, &m.data[y * m.step]);
    return m;
  }

  // Publishes image i until one trajectory arrives (discovery takes a moment).
  bool publish_until_reply(const sensor_msgs::msg::Image& m, std::chrono::seconds timeout) {
    const size_t before = got_.size();
    const auto end = std::chrono::steady_clock::now() + timeout;
    while (got_.size() == before && std::chrono::steady_clock::now() < end) {
      img_pub_->publish(m);
      spin_for(100ms);
    }
    return got_.size() > before;
  }

  std::shared_ptr<ActPolicyNode> node_;
  std::shared_ptr<rclcpp::Node> io_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr js_pub_;
  std::shared_ptr<tf2_ros::StaticTransformBroadcaster> tf_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr sub_;
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> exec_;
  std::vector<trajectory_msgs::msg::JointTrajectory> got_;
  std::vector<uint8_t> imgs_;
  std::vector<float> states_, ref_;
};

TEST_F(NodeTest, NoStateNoTrajectory) {
  EXPECT_FALSE(publish_until_reply(image(0), 2s));
}

TEST_F(NodeTest, ChunkMatchesPyTorch) {
  // sample 0 only: lang0.f32 is sample 0's instruction
  for (int k = 0; k < 5; ++k) { send_state(0); spin_for(100ms); }
  ASSERT_TRUE(publish_until_reply(image(0), 20s));
  const auto& tr = got_.back();
  EXPECT_EQ(tr.header.stamp.sec, 42);
  EXPECT_EQ(tr.header.stamp.nanosec, 0u);
  ASSERT_EQ(tr.joint_names.size(), 6u);
  EXPECT_EQ(tr.joint_names[0], "Rotation");
  ASSERT_EQ(tr.points.size(), static_cast<size_t>(act::kK));
  double max_diff = 0.0;
  for (int i = 0; i < act::kK; ++i) {
    ASSERT_EQ(tr.points[i].positions.size(), static_cast<size_t>(act::kAct));
    EXPECT_NEAR(rclcpp::Duration(tr.points[i].time_from_start).seconds(), 0.05 * (i + 1), 1e-6);
    for (int j = 0; j < act::kAct; ++j) {
      const double d = std::fabs(tr.points[i].positions[j] - ref_[i * act::kAct + j]);
      EXPECT_LT(d, 1e-4) << "point " << i << " joint " << j;  // also fails on NaN
      max_diff = std::max(max_diff, d);
    }
  }
  std::printf("max |node - PyTorch| = %.3g rad\n", max_diff);
}

TEST_F(NodeTest, NonFiniteStateNoTrajectory) {
  for (int k = 0; k < 5; ++k) { send_state(0, true); spin_for(100ms); }
  EXPECT_FALSE(publish_until_reply(image(0), 2s));
  for (int k = 0; k < 5; ++k) { send_state(0); spin_for(100ms); }
  EXPECT_TRUE(publish_until_reply(image(0), 20s));  // same setup, finite state: published
}

TEST_F(NodeTest, WrongEncodingIgnored) {
  for (int k = 0; k < 5; ++k) { send_state(0); spin_for(100ms); }
  EXPECT_FALSE(publish_until_reply(image(0, "bgr8"), 2s));
  EXPECT_TRUE(publish_until_reply(image(0), 20s));  // same setup, right encoding: published
}

}  // namespace
