from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_auvs", default_value="3"),
        DeclareLaunchArgument("checkpoint_path", default_value=""),
        Node(
            package="damage_detection",
            executable="infer_node",
            name="damage_inference_node",
            output="screen",
            parameters=[{
                "num_auvs": LaunchConfiguration("num_auvs"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
            }],
        ),
    ])
