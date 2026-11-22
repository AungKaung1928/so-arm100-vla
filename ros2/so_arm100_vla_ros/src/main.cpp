#include "so_arm100_vla_ros/act_policy_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<so_arm100_vla_ros::ActPolicyNode>());
  rclcpp::shutdown();
  return 0;
}
