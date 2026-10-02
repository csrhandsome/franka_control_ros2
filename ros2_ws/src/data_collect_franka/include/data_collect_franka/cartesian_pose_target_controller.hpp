#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace data_collect_franka {

class CartesianPoseTargetController : public controller_interface::ControllerInterface {
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
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;

  std::mutex target_mutex_;
  Eigen::Quaterniond orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond target_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d target_position_{Eigen::Vector3d::Zero()};
  bool has_target_{false};
  Eigen::Vector3d velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d acceleration_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_velocity_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_acceleration_{Eigen::Vector3d::Zero()};
  Eigen::Vector3d reference_position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond reference_orientation_{Eigen::Quaterniond::Identity()};

  double filter_coeff_{0.35};
  double max_translation_velocity_{0.05};
  double max_rotation_velocity_{0.2};
  double max_translation_acceleration_{0.1};
  double max_translation_jerk_{1.0};
  double max_rotation_acceleration_{0.2};
  double max_rotation_jerk_{2.0};
  double smoothing_frequency_{10.0};
  std::string base_frame_{"panda_link0"};
};

}  // namespace data_collect_franka
