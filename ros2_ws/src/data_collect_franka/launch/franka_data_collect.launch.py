# Overlay bringup: official franka.launch.py + custom controllers.yaml.
# Fake hardware only for this stage. Do not launch real-arm motion from here.

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_spawners(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    return [
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=[
                "cartesian_pose_target_controller",
                "--inactive",
                "--controller-manager-timeout",
                "60",
            ],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=[
                "joint_position_target_controller",
                "--inactive",
                "--controller-manager-timeout",
                "60",
            ],
            output="screen",
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    PathJoinSubstitution(
                        [
                            FindPackageShare("data_collect_franka"),
                            "launch",
                            "realsense_dual.launch.py",
                        ]
                    )
                ]
            ),
            launch_arguments={
                "external_serial": LaunchConfiguration("external_serial"),
                "wrist_serial": LaunchConfiguration("wrist_serial"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("start_cameras")),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    PathJoinSubstitution(
                        [
                            FindPackageShare("data_collect_franka"),
                            "launch",
                            "dh5_gripper.launch.py",
                        ]
                    )
                ]
            ),
            launch_arguments={
                "port": LaunchConfiguration("dh5_port"),
                "use_fake": LaunchConfiguration("dh5_use_fake"),
                "enable_cameras": LaunchConfiguration("start_dh5_cameras"),
                "left_device": LaunchConfiguration("dh5_left_device"),
                "right_device": LaunchConfiguration("dh5_right_device"),
            }.items(),
            condition=IfCondition(LaunchConfiguration("start_dh5_gripper")),
        ),
    ]


def generate_launch_description():
    controllers_yaml = PathJoinSubstitution(
        [FindPackageShare("data_collect_franka"), "config", "controllers.yaml"]
    )

    launch_args = [
        DeclareLaunchArgument("robot_type", default_value="fr3"),
        DeclareLaunchArgument("arm_prefix", default_value=""),
        DeclareLaunchArgument("namespace", default_value=""),
        DeclareLaunchArgument("robot_ip", default_value="dont-care"),
        DeclareLaunchArgument("load_gripper", default_value="true"),
        DeclareLaunchArgument("use_fake_hardware", default_value="true"),
        DeclareLaunchArgument("fake_sensor_commands", default_value="false"),
        DeclareLaunchArgument("joint_state_rate", default_value="30"),
        DeclareLaunchArgument("start_cameras", default_value="false"),
        DeclareLaunchArgument("external_serial", default_value=""),
        DeclareLaunchArgument("wrist_serial", default_value=""),
        DeclareLaunchArgument("start_dh5_gripper", default_value="false"),
        DeclareLaunchArgument("start_dh5_cameras", default_value="true"),
        DeclareLaunchArgument("dh5_use_fake", default_value="false"),
        DeclareLaunchArgument("dh5_port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("dh5_left_device", default_value="/dev/video2"),
        DeclareLaunchArgument("dh5_right_device", default_value="/dev/video1"),
    ]

    franka = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                PathJoinSubstitution(
                    [FindPackageShare("franka_bringup"), "launch", "franka.launch.py"]
                )
            ]
        ),
        launch_arguments={
            "robot_type": LaunchConfiguration("robot_type"),
            "arm_prefix": LaunchConfiguration("arm_prefix"),
            "namespace": LaunchConfiguration("namespace"),
            "robot_ip": LaunchConfiguration("robot_ip"),
            "load_gripper": LaunchConfiguration("load_gripper"),
            "use_fake_hardware": LaunchConfiguration("use_fake_hardware"),
            "fake_sensor_commands": LaunchConfiguration("fake_sensor_commands"),
            "joint_state_rate": LaunchConfiguration("joint_state_rate"),
            "controllers_yaml": controllers_yaml,
        }.items(),
    )

    return LaunchDescription(
        launch_args + [franka, OpaqueFunction(function=generate_spawners)]
    )
