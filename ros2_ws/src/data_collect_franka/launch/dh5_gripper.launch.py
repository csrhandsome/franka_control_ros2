# DH5 soft gripper: serial driver node + two fingertip UVC cameras.
# Cameras publish Image topics. The gripper node owns the Modbus serial port.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
            DeclareLaunchArgument("use_fake", default_value="false"),
            DeclareLaunchArgument("enable_cameras", default_value="true"),
            DeclareLaunchArgument("left_device", default_value="/dev/video2"),
            DeclareLaunchArgument("right_device", default_value="/dev/video1"),
            DeclareLaunchArgument("camera_width", default_value="640"),
            DeclareLaunchArgument("camera_height", default_value="480"),
            DeclareLaunchArgument("camera_fps", default_value="30"),
            DeclareLaunchArgument("force", default_value="50"),
            DeclareLaunchArgument("velocity", default_value="100"),
            Node(
                package="data_collect_franka",
                executable="dh5_gripper_node.py",
                name="dh5_gripper",
                namespace="dh5_gripper",
                output="screen",
                parameters=[
                    {
                        "port": LaunchConfiguration("port"),
                        "use_fake": ParameterValue(
                            LaunchConfiguration("use_fake"), value_type=bool
                        ),
                        "force": ParameterValue(
                            LaunchConfiguration("force"), value_type=int
                        ),
                        "velocity": ParameterValue(
                            LaunchConfiguration("velocity"), value_type=int
                        ),
                    }
                ],
            ),
            Node(
                package="data_collect_franka",
                executable="usb_camera_node.py",
                namespace="gripper_left",
                name="camera",
                output="screen",
                parameters=[
                    {
                        "device": LaunchConfiguration("left_device"),
                        "width": ParameterValue(
                            LaunchConfiguration("camera_width"), value_type=int
                        ),
                        "height": ParameterValue(
                            LaunchConfiguration("camera_height"), value_type=int
                        ),
                        "fps": ParameterValue(
                            LaunchConfiguration("camera_fps"), value_type=int
                        ),
                        "frame_id": "gripper_left_optical_frame",
                    }
                ],
                condition=IfCondition(LaunchConfiguration("enable_cameras")),
            ),
            Node(
                package="data_collect_franka",
                executable="usb_camera_node.py",
                namespace="gripper_right",
                name="camera",
                output="screen",
                parameters=[
                    {
                        "device": LaunchConfiguration("right_device"),
                        "width": ParameterValue(
                            LaunchConfiguration("camera_width"), value_type=int
                        ),
                        "height": ParameterValue(
                            LaunchConfiguration("camera_height"), value_type=int
                        ),
                        "fps": ParameterValue(
                            LaunchConfiguration("camera_fps"), value_type=int
                        ),
                        "frame_id": "gripper_right_optical_frame",
                    }
                ],
                condition=IfCondition(LaunchConfiguration("enable_cameras")),
            ),
        ]
    )
