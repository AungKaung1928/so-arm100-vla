cmake_minimum_required(VERSION 3.16)
project(so_arm100_vla_ros CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Werror)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# ONNX Runtime is not a ROS package: ORT_INCLUDE holds onnxruntime_cxx_api.h, ORT_LIB holds
# libonnxruntime.so.1 (same version as the Python wheel the graph was checked with, 1.30.0).
set(ORT_INCLUDE "" CACHE PATH "dir with onnxruntime_cxx_api.h")
set(ORT_LIB "" CACHE PATH "dir with libonnxruntime.so.1")
find_library(ONNXRUNTIME NAMES libonnxruntime.so.1 onnxruntime PATHS ${ORT_LIB} NO_DEFAULT_PATH REQUIRED)

# The policy step is shared with the standalone runner in the repo's cpp/ directory.
set(ACT_STEP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp)

add_library(act_policy_node_lib src/act_policy_node.cpp)
target_include_directories(act_policy_node_lib PUBLIC include ${ACT_STEP_DIR})
target_include_directories(act_policy_node_lib SYSTEM PUBLIC ${ORT_INCLUDE})
target_link_libraries(act_policy_node_lib ${ONNXRUNTIME})
ament_target_dependencies(act_policy_node_lib rclcpp sensor_msgs trajectory_msgs tf2_ros)

add_executable(act_policy_node src/main.cpp)
target_link_libraries(act_policy_node act_policy_node_lib)
set_target_properties(act_policy_node PROPERTIES INSTALL_RPATH ${ORT_LIB})

install(TARGETS act_policy_node_lib act_policy_node
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(geometry_msgs REQUIRED)
  # ACT_TEST_DATA: a cpp/bench.py work dir (policy.onnx, images.u8, states.f32, lang.f32,
  # stats.f32, ref.f32). The test is skipped, not failed, when it is unset.
  set(ACT_TEST_DATA "" CACHE PATH "cpp/bench.py work dir for the node test")
  ament_add_gtest(test_act_policy_node test/test_act_policy_node.cpp TIMEOUT 120)
  target_link_libraries(test_act_policy_node act_policy_node_lib)
  ament_target_dependencies(test_act_policy_node rclcpp sensor_msgs trajectory_msgs tf2_ros geometry_msgs)
  target_compile_definitions(test_act_policy_node PRIVATE ACT_TEST_DATA="${ACT_TEST_DATA}")
endif()

ament_package()
