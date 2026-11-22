#include "so_arm100_vla_ros/act_policy_node.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include "tf2/exceptions.h"

namespace so_arm100_vla_ros {
namespace {

std::vector<float> read_floats(const std::string& path, size_t expected) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open " + path);
  const auto bytes = static_cast<size_t>(f.tellg());
  if (bytes != expected * sizeof(float)) {
    throw std::runtime_error(path + ": " + std::to_string(bytes) + " bytes, expected " +
                             std::to_string(expected * sizeof(float)));
  }
  std::vector<float> v(expected);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
  return v;
}

}  // namespace

ActPolicyNode::ActPolicyNode(const rclcpp::NodeOptions& options)
: Node("act_policy_node", options) {
  // Defaults: the simulator's joint order (so_arm100_sim.scene.JOINT_NAMES) and 20 Hz control.
  joint_names_ = declare_parameter<std::vector<std::string>>(
    "joint_names",
    std::vector<std::string>{"Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"});
  base_frame_ = declare_parameter<std::string>("base_frame", "world");
  ee_frame_ = declare_parameter<std::string>("ee_frame", "ee");
  control_period_ = declare_parameter<double>("control_period", 0.05);
  const auto model = declare_parameter<std::string>("model_path", "");
  const auto stats = declare_parameter<std::string>("stats_path", "");
  const auto lang = declare_parameter<std::string>("lang_path", "");
  if (joint_names_.size() != static_cast<size_t>(act::kAct)) {
    throw std::invalid_argument("joint_names needs " + std::to_string(act::kAct) + " names");
  }
  if (!(control_period_ > 0.0)) throw std::invalid_argument("control_period must be > 0");

  const auto lang_v = read_floats(lang, act::kEmbed);
  std::copy(lang_v.begin(), lang_v.end(), lang_.begin());
  step_ = std::make_unique<act::Step>(model.c_str(),
                                      act::Stats::from_flat(read_floats(stats, act::kStatsLen).data()));
  image_.resize(static_cast<size_t>(act::kH) * act::kW * 3);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  traj_pub_ = create_publisher<trajectory_msgs::msg::JointTrajectory>("joint_trajectory", 10);
  joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
    "joint_states", 10, [this](sensor_msgs::msg::JointState::ConstSharedPtr m) { on_joint_state(m); });
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    "image", rclcpp::SensorDataQoS(), [this](sensor_msgs::msg::Image::ConstSharedPtr m) { on_image(m); });
  RCLCPP_INFO(get_logger(), "policy %s loaded, 1 ORT thread", model.c_str());
}

void ActPolicyNode::on_joint_state(sensor_msgs::msg::JointState::ConstSharedPtr msg) {
  joints_ = std::move(msg);
}

bool ActPolicyNode::build_state(std::array<float, act::kState>& state) {
  if (!joints_) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no joint_states yet");
    return false;
  }
  const auto& js = *joints_;
  if (js.position.size() != js.name.size() || js.velocity.size() != js.name.size()) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "joint_states needs one position and one velocity per name");
    return false;
  }
  for (int i = 0; i < act::kAct; ++i) {
    const auto it = std::find(js.name.begin(), js.name.end(), joint_names_[i]);
    if (it == js.name.end()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "joint %s missing from joint_states",
                           joint_names_[i].c_str());
      return false;
    }
    const auto k = static_cast<size_t>(it - js.name.begin());
    state[i] = static_cast<float>(js.position[k]);
    state[act::kAct + i] = static_cast<float>(js.velocity[k]);
  }
  try {
    const auto t = tf_buffer_->lookupTransform(base_frame_, ee_frame_, tf2::TimePointZero);
    state[12] = static_cast<float>(t.transform.translation.x);
    state[13] = static_cast<float>(t.transform.translation.y);
    state[14] = static_cast<float>(t.transform.translation.z);
  } catch (const tf2::TransformException& e) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s", e.what());
    return false;
  }
  return true;
}

void ActPolicyNode::on_image(sensor_msgs::msg::Image::ConstSharedPtr msg) {
  constexpr size_t row = static_cast<size_t>(act::kW) * 3;
  if (msg->encoding != "rgb8" || msg->width != static_cast<uint32_t>(act::kW) ||
      msg->height != static_cast<uint32_t>(act::kH) || msg->step < row ||
      msg->data.size() < static_cast<size_t>(msg->step) * act::kH) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                         "image must be rgb8 %dx%d, got %s %ux%u", act::kW, act::kH,
                         msg->encoding.c_str(), msg->width, msg->height);
    return;
  }
  std::array<float, act::kState> state{};
  if (!build_state(state)) return;
  for (int y = 0; y < act::kH; ++y) {  // drop row padding (step > width * 3)
    std::copy_n(msg->data.begin() + static_cast<std::ptrdiff_t>(y * msg->step), row,
                image_.begin() + static_cast<std::ptrdiff_t>(y * row));
  }
  std::array<float, act::kK * act::kAct> chunk{};
  step_->run(image_.data(), state.data(), lang_.data(), chunk.data());

  trajectory_msgs::msg::JointTrajectory traj;
  traj.header.stamp = msg->header.stamp;
  traj.header.frame_id = base_frame_;
  traj.joint_names = joint_names_;
  traj.points.resize(act::kK);
  for (int i = 0; i < act::kK; ++i) {
    auto& p = traj.points[i];
    p.positions.assign(chunk.begin() + i * act::kAct, chunk.begin() + (i + 1) * act::kAct);
    p.time_from_start = rclcpp::Duration::from_seconds((i + 1) * control_period_);
  }
  traj_pub_->publish(traj);
}

}  // namespace so_arm100_vla_ros
