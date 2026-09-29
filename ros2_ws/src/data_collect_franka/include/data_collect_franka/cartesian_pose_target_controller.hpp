#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_cartesian_pose_interface.hpp>
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
  void maybe_set_collision_behavior();

  std::unique_ptr<franka_semantic_components::FrankaCartesianPoseInterface> franka_cartesian_pose_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_pose_sub_;

  std::mutex target_mutex_;
  Eigen::Quaterniond orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d position_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond target_orientation_{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d target_position_{Eigen::Vector3d::Zero()};
  bool has_target_{false};

  const bool k_elbow_activated_{false};
  double filter_coeff_{0.35};
  std::string arm_prefix_;
};

}  // namespace data_collect_franka
