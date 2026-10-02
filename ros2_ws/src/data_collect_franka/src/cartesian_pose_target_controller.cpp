#include "data_collect_franka/cartesian_pose_target_controller.hpp"

#include <algorithm>
#include <cmath>

namespace data_collect_franka {
namespace {
controller_interface::InterfaceConfiguration pose_interfaces() {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  // LCAS Panda driver: column-major O_T_EE, exported as 00 ... 15.
  for (int i = 0; i < 16; ++i) {
    config.names.push_back("ee_cartesian_position/" +
                           (i < 10 ? std::string("0") : std::string()) + std::to_string(i));
  }
  return config;
}
}  // namespace

controller_interface::InterfaceConfiguration
CartesianPoseTargetController::command_interface_configuration() const {
  return pose_interfaces();
}

controller_interface::InterfaceConfiguration
CartesianPoseTargetController::state_interface_configuration() const {
  return pose_interfaces();
}

controller_interface::return_type CartesianPoseTargetController::update(
    const rclcpp::Time&, const rclcpp::Duration& period) {
  Eigen::Quaterniond target_orientation;
  Eigen::Vector3d target_position;
  {
    std::lock_guard<std::mutex> lock(target_mutex_);
    target_orientation = has_target_ ? target_orientation_ : orientation_;
    target_position = has_target_ ? target_position_ : position_;
  }
  const double dt = std::clamp(period.seconds(), 0.0, 0.01);
  const double alpha = 1.0 - std::pow(1.0 - filter_coeff_, dt / 0.001);
  reference_position_ += alpha * (target_position - reference_position_);
  if (reference_orientation_.dot(target_orientation) < 0.0) {
    target_orientation.coeffs() = -target_orientation.coeffs();
  }
  reference_orientation_ = reference_orientation_.slerp(alpha, target_orientation).normalized();
  const auto bounded = [](Eigen::Vector3d value, double limit) -> Eigen::Vector3d {
    const double norm = value.norm();
    return norm > limit ? Eigen::Vector3d(value * (limit / norm)) : value;
  };
  const double omega = smoothing_frequency_;
  // A critically damped third-order follower starts with zero velocity and
  // acceleration. Target steps change jerk, never jump directly to a velocity.
  const auto advance = [&](const Eigen::Vector3d& error, Eigen::Vector3d& velocity,
                           Eigen::Vector3d& acceleration, double velocity_limit,
                           double acceleration_limit, double jerk_limit) {
    const Eigen::Vector3d jerk = bounded(
        omega * omega * omega * error - 3.0 * omega * omega * velocity -
        3.0 * omega * acceleration, jerk_limit);
    acceleration = bounded(acceleration + dt * jerk, acceleration_limit);
    velocity = bounded(velocity + dt * acceleration, velocity_limit);
  };
  advance(reference_position_ - position_, velocity_, acceleration_,
          max_translation_velocity_, max_translation_acceleration_, max_translation_jerk_);
  position_ += dt * velocity_;
  Eigen::Quaterniond difference = orientation_.conjugate() * reference_orientation_;
  if (difference.w() < 0.0) {
    difference.coeffs() = -difference.coeffs();
  }
  const Eigen::AngleAxisd error_rotation(difference.normalized());
  advance(error_rotation.axis() * error_rotation.angle(), angular_velocity_, angular_acceleration_,
          max_rotation_velocity_, max_rotation_acceleration_, max_rotation_jerk_);
  const double angle_step = dt * angular_velocity_.norm();
  if (angle_step > 1e-12) {
    orientation_ = (orientation_ * Eigen::Quaterniond(
        Eigen::AngleAxisd(angle_step, angular_velocity_.normalized()))).normalized();
  }
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) = orientation_.toRotationMatrix();
  pose.block<3, 1>(0, 3) = position_;
  for (int i = 0; i < 16; ++i) {
    command_interfaces_[i].set_value(pose.data()[i]);
  }
  return controller_interface::return_type::OK;
}

CallbackReturn CartesianPoseTargetController::on_init() {
  auto_declare<std::string>("base_frame", "panda_link0");
  auto_declare<double>("filter_coeff", 0.35);
  auto_declare<double>("max_translation_velocity", 0.05);
  auto_declare<double>("max_rotation_velocity", 0.2);
  auto_declare<double>("max_translation_acceleration", 0.1);
  auto_declare<double>("max_translation_jerk", 1.0);
  auto_declare<double>("max_rotation_acceleration", 0.2);
  auto_declare<double>("max_rotation_jerk", 2.0);
  auto_declare<double>("smoothing_frequency", 10.0);
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseTargetController::on_configure(const rclcpp_lifecycle::State&) {
  base_frame_ = get_node()->get_parameter("base_frame").as_string();
  filter_coeff_ = get_node()->get_parameter("filter_coeff").as_double();
  max_translation_velocity_ = get_node()->get_parameter("max_translation_velocity").as_double();
  max_rotation_velocity_ = get_node()->get_parameter("max_rotation_velocity").as_double();
  max_translation_acceleration_ = get_node()->get_parameter("max_translation_acceleration").as_double();
  max_translation_jerk_ = get_node()->get_parameter("max_translation_jerk").as_double();
  max_rotation_acceleration_ = get_node()->get_parameter("max_rotation_acceleration").as_double();
  max_rotation_jerk_ = get_node()->get_parameter("max_rotation_jerk").as_double();
  smoothing_frequency_ = get_node()->get_parameter("smoothing_frequency").as_double();
  for (double limit : {max_translation_acceleration_, max_translation_jerk_,
                       max_rotation_acceleration_, max_rotation_jerk_, smoothing_frequency_}) {
    if (!std::isfinite(limit) || limit <= 0.0) {
      return CallbackReturn::ERROR;
    }
  }
  if (!std::isfinite(filter_coeff_) || filter_coeff_ <= 0.0 || filter_coeff_ > 1.0 ||
      !std::isfinite(max_translation_velocity_) || max_translation_velocity_ <= 0.0 ||
      !std::isfinite(max_rotation_velocity_) || max_rotation_velocity_ <= 0.0) {
    RCLCPP_ERROR(get_node()->get_logger(), "Invalid Cartesian filter or velocity limits");
    return CallbackReturn::ERROR;
  }
  target_pose_sub_ = get_node()->create_subscription<geometry_msgs::msg::PoseStamped>(
      "~/target_pose", rclcpp::SensorDataQoS(),
      [this](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        Eigen::Quaterniond orientation(msg->pose.orientation.w, msg->pose.orientation.x,
                                       msg->pose.orientation.y, msg->pose.orientation.z);
        Eigen::Vector3d position(msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
        if ((!msg->header.frame_id.empty() && msg->header.frame_id != base_frame_) ||
            !position.allFinite() || !orientation.coeffs().allFinite() ||
            orientation.norm() < 1e-8) {
          RCLCPP_WARN_THROTTLE(get_node()->get_logger(), *get_node()->get_clock(), 2000,
                              "Ignoring invalid Cartesian target or incorrect base frame");
          return;
        }
        orientation.normalize();
        std::lock_guard<std::mutex> lock(target_mutex_);
        target_orientation_ = orientation;
        target_position_ = position;
        has_target_ = true;
      });
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseTargetController::on_activate(const rclcpp_lifecycle::State&) {
  if (state_interfaces_.size() != 16 || command_interfaces_.size() != 16) {
    return CallbackReturn::ERROR;
  }
  Eigen::Matrix4d pose;
  for (int i = 0; i < 16; ++i) {
    pose.data()[i] = state_interfaces_[i].get_value();
  }
  // Calibrated Panda transforms are not exactly orthogonal (about 2e-5 in
  // measured feedback). Accept only a near-rigid transform, then normalize.
  const Eigen::Matrix3d rotation = pose.block<3, 3>(0, 0);
  if (!pose.allFinite() ||
      !pose.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-6) ||
      !(rotation.transpose() * rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-3) ||
      std::abs(rotation.determinant() - 1.0) > 1e-3) {
    RCLCPP_ERROR(get_node()->get_logger(), "No valid measured Cartesian pose; refusing activation");
    return CallbackReturn::ERROR;
  }
  // The driver seeds command interfaces from the last desired pose before
  // starting libfranka. Keep that pose at activation, then approach measured
  // Python targets through the velocity-limited filter, avoiding a first-step jump.
  Eigen::Matrix4d commanded;
  for (int i = 0; i < 16; ++i) {
    commanded.data()[i] = command_interfaces_[i].get_value();
  }
  if (commanded.allFinite() &&
      commanded.row(3).isApprox(Eigen::RowVector4d(0, 0, 0, 1), 1e-6)) {
    const Eigen::Matrix3d command_rotation = commanded.block<3, 3>(0, 0);
    if (!(command_rotation.transpose() * command_rotation).isApprox(Eigen::Matrix3d::Identity(), 1e-3) ||
        std::abs(command_rotation.determinant() - 1.0) > 1e-3) {
      return CallbackReturn::ERROR;
    }
    pose = commanded;
  }
  position_ = pose.block<3, 1>(0, 3);
  orientation_ = Eigen::Quaterniond(pose.block<3, 3>(0, 0)).normalized();
  velocity_.setZero();
  acceleration_.setZero();
  angular_velocity_.setZero();
  angular_acceleration_.setZero();
  reference_position_ = position_;
  reference_orientation_ = orientation_;
  std::lock_guard<std::mutex> lock(target_mutex_);
  target_orientation_ = orientation_;
  target_position_ = position_;
  has_target_ = false;
  return CallbackReturn::SUCCESS;
}

CallbackReturn CartesianPoseTargetController::on_deactivate(const rclcpp_lifecycle::State&) {
  std::lock_guard<std::mutex> lock(target_mutex_);
  has_target_ = false;
  return CallbackReturn::SUCCESS;
}
}  // namespace data_collect_franka

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(data_collect_franka::CartesianPoseTargetController,
                       controller_interface::ControllerInterface)
