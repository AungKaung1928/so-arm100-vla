// ActPolicyNode: the so-arm100-vla language policy step (act_step.hpp) behind ROS 2 topics.
//   in:  image (sensor_msgs/Image, rgb8, 128 x 96), joint_states (sensor_msgs/JointState,
//        position + velocity of the 6 joints in `joint_names`), TF base_frame -> ee_frame
//   out: joint_trajectory (trajectory_msgs/JointTrajectory): the 10-step chunk, point i at
//        (i + 1) * control_period, stamped with the image's stamp
// The instruction is a 384-d embedding file (lang_path) made offline by the Python text
// encoder: the encoder runs once per instruction, not per step (as LangRunner does).
// One chunk per image; temporal ensembling is left to the consumer.
#pragma once

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "act_step.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace so_arm100_vla_ros {

class ActPolicyNode : public rclcpp::Node {
 public:
  explicit ActPolicyNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  void on_joint_state(sensor_msgs::msg::JointState::ConstSharedPtr msg);
  void on_image(sensor_msgs::msg::Image::ConstSharedPtr msg);
  // Fills the 15-d proprio state (qpos, qvel, ee position); false (and a log line) when an
  // input is missing or non-finite.
  bool build_state(std::array<float, act::kState>& state);

  std::vector<std::string> joint_names_;
  std::string base_frame_, ee_frame_;
  double control_period_;
  std::array<float, act::kEmbed> lang_{};
  std::unique_ptr<act::Step> step_;
  std::vector<uint8_t> image_;
  sensor_msgs::msg::JointState::ConstSharedPtr joints_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr traj_pub_;
};

}  // namespace so_arm100_vla_ros
