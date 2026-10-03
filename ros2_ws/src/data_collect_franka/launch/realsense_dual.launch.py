from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _camera_node(namespace, serial):
    parameters = {
        "camera_name": namespace,
        "enable_color": True,
        "enable_depth": False,
        "enable_infra1": False,
        "enable_infra2": False,
        "enable_gyro": False,
        "enable_accel": False,
        "rgb_camera.profile": "640x480x30",
    }
    if serial:
        parameters["serial_no"] = serial
    return Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        # The driver publishes below its node name. Keep the topics aligned
        # with config/collect/franka.yaml: /external/color/image_raw and
        # /wrist/color/image_raw. Override the driver's default /camera namespace.
        name=namespace,
        namespace="/",
        parameters=[parameters],
        output="screen",
    )


def generate_camera_nodes(context):
    return [
        _camera_node(
            "external", LaunchConfiguration("external_serial").perform(context)
        ),
        _camera_node("wrist", LaunchConfiguration("wrist_serial").perform(context)),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("external_serial", default_value=""),
            DeclareLaunchArgument("wrist_serial", default_value=""),
            OpaqueFunction(function=generate_camera_nodes),
        ]
    )
