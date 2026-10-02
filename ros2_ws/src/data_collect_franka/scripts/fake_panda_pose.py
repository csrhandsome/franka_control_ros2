#!/usr/bin/env python3
"""Expose GenericSystem Cartesian GPIO feedback as a pose, for fake hardware only."""
import numpy as np
import rclpy
from control_msgs.msg import DynamicJointState
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation


def main():
    rclpy.init()
    node = Node("fake_panda_pose")
    publisher = node.create_publisher(
        PoseStamped, "franka_robot_state_broadcaster/current_pose", qos_profile_sensor_data
    )

    def on_state(msg):
        if "ee_cartesian_position" not in msg.joint_names:
            return
        interfaces = msg.interface_values[msg.joint_names.index("ee_cartesian_position")]
        values = dict(zip(interfaces.interface_names, interfaces.values))
        if any(f"{index:02d}" not in values for index in range(16)):
            return
        pose = np.array([values[f"{index:02d}"] for index in range(16)]).reshape(4, 4, order="F")
        if not np.isfinite(pose).all():
            return
        result = PoseStamped()
        result.header = msg.header
        result.header.frame_id = "panda_link0"
        result.pose.position.x, result.pose.position.y, result.pose.position.z = map(float, pose[:3, 3])
        quat = Rotation.from_matrix(pose[:3, :3]).as_quat()
        (result.pose.orientation.x, result.pose.orientation.y,
         result.pose.orientation.z, result.pose.orientation.w) = map(float, quat)
        publisher.publish(result)

    node.create_subscription(DynamicJointState, "dynamic_joint_states", on_state, qos_profile_sensor_data)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
