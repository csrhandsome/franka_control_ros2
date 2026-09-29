#pragma once

#include <array>
#include <mutex>
#include <string>

#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace data_collect_franka {

class JointPositionTargetController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr int kNumJoints = 7;

  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr target_joints_sub_;
  std::mutex target_mutex_;
  std::array<double, kNumJoints> command_q_{};
  std::array<double, kNumJoints> target_q_{};
  bool has_target_{false};
  bool initialized_{false};

  double max_joint_velocity_{0.8};
  std::string arm_prefix_;
  std::string robot_type_{"fr3"};
};

}  // namespace data_collect_franka
