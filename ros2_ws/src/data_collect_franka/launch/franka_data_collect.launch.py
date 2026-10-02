# Panda bringup using the pinned LCAS driver and compatible project controllers.

import os
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import xacro
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    Shutdown,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_spawners(context):
    namespace = LaunchConfiguration("namespace").perform(context)
    return [
        Node(
            package="controller_manager",
            executable="spawner",
            namespace=namespace,
            arguments=[
                "joint_trajectory_controller",
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
    launch_args = [
        DeclareLaunchArgument("robot_type", default_value="panda"),
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

    return LaunchDescription(
        launch_args + [OpaqueFunction(function=generate_panda_bringup)]
    )


def generate_panda_bringup(context):
    def value(name):
        return LaunchConfiguration(name).perform(context)

    if value("robot_type") != "panda" or value("arm_prefix"):
        raise ValueError("The LCAS Panda runtime requires robot_type:=panda and an empty arm_prefix")
    fake = value("use_fake_hardware").lower() == "true"
    namespace = value("namespace")
    description_file = os.path.join(
        get_package_share_directory("franka_description"), "robots", "panda_arm.urdf.xacro"
    )
    description = xacro.process_file(description_file, mappings={
        "robot_ip": value("robot_ip"),
        "hand": value("load_gripper"),
        "use_fake_hardware": value("use_fake_hardware"),
        "fake_sensor_commands": value("fake_sensor_commands"),
    }).toxml()
    if fake:
        # GenericSystem mirrors Cartesian GPIO commands as state. This tests ROS
        # wiring and lifecycle only; it does not simulate Panda dynamics or IK.
        root = ET.fromstring(description)
        system = root.find("ros2_control")
        for joint in system.findall("joint"):
            initial = joint.find("param[@name='initial_position']").text
            position = joint.find("state_interface[@name='position']")
            ET.SubElement(position, "param", name="initial_value").text = initial
        gpio = ET.SubElement(system, "gpio", name="ee_cartesian_position")
        pose = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0.3, 0, 0.5, 1]
        for index, initial in enumerate(pose):
            ET.SubElement(gpio, "command_interface", name=f"{index:02d}")
            interface = ET.SubElement(gpio, "state_interface", name=f"{index:02d}")
            ET.SubElement(interface, "param", name="initial_value").text = str(initial)
        description = ET.tostring(root, encoding="unicode")
    description_parameter = {"robot_description": ParameterValue(description, value_type=str)}
    config_dir = os.path.join(get_package_share_directory("data_collect_franka"), "config")
    manager = Node(
        package="franka_control2", executable="franka_control2_node", namespace=namespace,
        parameters=[description_parameter, os.path.join(config_dir, "controllers.yaml"),
                    os.path.join(config_dir, "panda_trajectory_controller.yaml")],
        output="screen", on_exit=Shutdown(),
    )
    actions = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             namespace=namespace, parameters=[description_parameter], output="screen"),
        manager,
        Node(package="controller_manager", executable="spawner", namespace=namespace,
             arguments=["joint_state_broadcaster", "--controller-manager-timeout", "60"],
             output="screen"),
    ]
    if fake:
        actions.append(Node(package="data_collect_franka", executable="fake_panda_pose.py",
                            namespace=namespace, output="screen"))
    else:
        actions.append(Node(
            package="controller_manager", executable="spawner", namespace=namespace,
            arguments=["franka_robot_state_broadcaster", "--controller-manager-timeout", "60"],
            output="screen",
        ))
        if value("load_gripper").lower() == "true":
            actions.append(IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(
                    get_package_share_directory("franka_gripper"), "launch", "gripper.launch.py"
                )), launch_arguments={"robot_ip": value("robot_ip")}.items(),
            ))
    return actions + generate_spawners(context)
