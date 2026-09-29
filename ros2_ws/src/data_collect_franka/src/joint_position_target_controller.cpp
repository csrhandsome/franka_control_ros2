#include "data_collect_franka/joint_position_target_controller.hpp"

#include <algorithm>
#include <cmath>

namespace data_collect_franka {

controller_interface::InterfaceConfiguration
JointPositionTargetController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                           "/position");
  }
  return config;
}

controller_interface::InterfaceConfiguration
JointPositionTargetController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= kNumJoints; ++i) {
    config.names.push_back(arm_prefix_ + robot_type_ + "_joint" + std::to_string(i) +
                           "/position");
  }
  return config;
}

controller_interface::return_type JointPositionTargetController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& period) {
  if (!initialized_) {
    for (int i = 0; i < kNumJoints; ++i) {
      command_q_.at(static_cast<size_t>(i)) = state_interfaces_[i].get_value();
      target_q_.at(static_cast<size_t>(i)) = command_q_.at(static_cast<size_t>(i));
    }
    initialized_ = true;
  }

  std::array<double, kNumJoints> target{};
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    target = has_target_ ? target_q_ : command_q_;
  }

  const double dt = std::max(period.seconds(), 1e-4);
  const double max_step = std::max(max_joint_velocity_, 0.0) * dt;
  for (int i = 0; i < kNumJoints; ++i) {
    const double error = target.at(static_cast<size_t>(i)) - command_q_.at(static_cast<size_t>(i));
    const double step = std::clamp(error, -max_step, max_step);
    command_q_.at(static_cast<size_t>(i)) += step;
    command_interfaces_[i].set_value(command_q_.at(static_cast<size_t>(i)));
  }
  return controller_interface::return_type::OK;
}

CallbackReturn JointPositionTargetController::on_init() {
  auto_declare<std::string>("arm_prefix", "");
  auto_declare<std::string>("robot_type", "fr3");
  auto_declare<double>("max_joint_velocity", 0.8);
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointPositionTargetController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
  robot_type_ = get_node()->get_parameter("robot_type").as_string();
  if (robot_type_.empty()) {
    robot_type_ = "fr3";
  }
  max_joint_velocity_ = get_node()->get_parameter("max_joint_velocity").as_double();

  target_joints_sub_ = get_node()->create_subscription<std_msgs::msg::Float64MultiArray>(
      "~/target_joints", rclcpp::SystemDefaultsQoS(),
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() != static_cast<size_t>(kNumJoints)) {
          RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                               "Ignoring joint target with %zu values; expected %d.",
                               msg->data.size(), kNumJoints);
          return;
        }
        std::lock_guard<std::mutex> lock(target_mutex_);
        for (int i = 0; i < kNumJoints; ++i) {
          target_q_.at(static_cast<size_t>(i)) = msg->data[static_cast<size_t>(i)];
        }
        has_target_ = true;
      });
  return CallbackReturn::SUCCESS;
}

CallbackReturn JointPositionTargetController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  initialized_ = false;
  has_target_ = false;
  return CallbackReturn::SUCCESS;
}

}  // namespace data_collect_franka

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(data_collect_franka::JointPositionTargetController,
                       controller_interface::ControllerInterface)
