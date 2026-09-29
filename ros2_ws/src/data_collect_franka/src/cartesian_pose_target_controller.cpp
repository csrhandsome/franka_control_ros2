#include "data_collect_franka/cartesian_pose_target_controller.hpp"

#include <algorithm>
#include <chrono>
#include <tuple>
#include <franka_msgs/srv/set_full_collision_behavior.hpp>

namespace data_collect_franka {

controller_interface::InterfaceConfiguration
CartesianPoseTargetController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_command_interface_names();
  return config;
}

controller_interface::InterfaceConfiguration
CartesianPoseTargetController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  config.names = franka_cartesian_pose_->get_state_interface_names();
  return config;
}

controller_interface::return_type CartesianPoseTargetController::update(
    const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {
  Eigen::Quaterniond target_orientation;
  Eigen::Vector3d target_position;
  bool has_target = false;
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    has_target = has_target_;
    target_orientation = target_orientation_;
    target_position = target_position_;
  }

  if (has_target) {
    const double alpha = std::clamp(filter_coeff_, 0.0, 1.0);
    position_ = (1.0 - alpha) * position_ + alpha * target_position;
    if (orientation_.dot(target_orientation) < 0.0) {
      target_orientation.coeffs() = -target_orientation.coeffs();
    }
    orientation_ = orientation_.slerp(alpha, target_orientation);
  }

  if (!franka_cartesian_pose_->setCommand(orientation_, position_)) {
    RCLCPP_ERROR_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                          "setCommand failed; is the elbow command interface accidentally active?");
    return controller_interface::return_type::ERROR;
  }
  return controller_interface::return_type::OK;
}

CallbackReturn CartesianPoseTargetController::on_init() {
  auto_declare<std::string>("arm_prefix", "");
  auto_declare<double>("filter_coeff", 0.35);
  auto_declare<double>("collision_service_timeout_s", 1.0);
  return CallbackReturn::SUCCESS;
}

void CartesianPoseTargetController::maybe_set_collision_behavior() {
  const double timeout_s = get_node()->get_parameter("collision_service_timeout_s").as_double();
  auto client = get_node()->create_client<franka_msgs::srv::SetFullCollisionBehavior>(
      "service_server/set_full_collision_behavior");
  if (!client->wait_for_service(std::chrono::duration<double>(timeout_s))) {
    RCLCPP_WARN(get_node()->get_logger(),
                "Collision behavior service not available within %.2fs; continuing.", timeout_s);
    return;
  }

  auto request = std::make_shared<franka_msgs::srv::SetFullCollisionBehavior::Request>();
  request->lower_torque_thresholds_nominal = {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0};
  request->upper_torque_thresholds_nominal = {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0};
  request->lower_torque_thresholds_acceleration = {25.0, 25.0, 22.0, 20.0, 19.0, 17.0, 14.0};
  request->upper_torque_thresholds_acceleration = {35.0, 35.0, 32.0, 30.0, 29.0, 27.0, 24.0};
  request->lower_force_thresholds_nominal = {30.0, 30.0, 30.0, 25.0, 25.0, 25.0};
  request->upper_force_thresholds_nominal = {40.0, 40.0, 40.0, 35.0, 35.0, 35.0};
  request->lower_force_thresholds_acceleration = {30.0, 30.0, 30.0, 25.0, 25.0, 25.0};
  request->upper_force_thresholds_acceleration = {40.0, 40.0, 40.0, 35.0, 35.0, 35.0};

  auto future = client->async_send_request(request);
  if (future.wait_for(std::chrono::duration<double>(timeout_s)) != std::future_status::ready) {
    RCLCPP_WARN(get_node()->get_logger(), "Collision behavior service timed out; continuing.");
    return;
  }
  RCLCPP_INFO(get_node()->get_logger(), "Default collision behavior requested.");
}

CallbackReturn CartesianPoseTargetController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_prefix_ = get_node()->get_parameter("arm_prefix").as_string();
  arm_prefix_ = arm_prefix_.empty() ? "" : arm_prefix_ + "_";
  filter_coeff_ = get_node()->get_parameter("filter_coeff").as_double();

  franka_cartesian_pose_ =
      std::make_unique<franka_semantic_components::FrankaCartesianPoseInterface>(
          arm_prefix_, k_elbow_activated_);

  maybe_set_collision_behavior();

  target_pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/target_pose", rclcpp::SensorDataQoS(),
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        Eigen::Quaterniond orientation(msg->pose.orientation.w, msg->pose.orientation.x,
                                       msg->pose.orientation.y, msg->pose.orientation.z);
        if (orientation.norm() < 1e-8) {
          orientation = Eigen::Quaterniond::Identity();
        } else {
          orientation.normalize();
        }
        std::lock_guard<std::mutex> lock(target_mutex_);
        target_orientation_ = orientation;
        target_position_ = Eigen::Vector3d(msg->pose.position.x, msg->pose.position.y,
                                           msg->pose.position.z);
        has_target_ = true;
      });

  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseTargetController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->assign_loaned_command_interfaces(command_interfaces_);
  franka_cartesian_pose_->assign_loaned_state_interfaces(state_interfaces_);

  try {
    std::tie(orientation_, position_) =
        franka_cartesian_pose_->getCurrentOrientationAndTranslation();
  } catch (const std::exception& exc) {
    RCLCPP_WARN(get_node()->get_logger(),
                "No cartesian pose state (%s); holding identity until a target arrives.",
                exc.what());
    orientation_ = Eigen::Quaterniond::Identity();
    position_ = Eigen::Vector3d::Zero();
  }

  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    target_orientation_ = orientation_;
    target_position_ = position_;
    has_target_ = false;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseTargetController::on_deactivate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_cartesian_pose_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

}  // namespace data_collect_franka

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(data_collect_franka::CartesianPoseTargetController,
                       controller_interface::ControllerInterface)
