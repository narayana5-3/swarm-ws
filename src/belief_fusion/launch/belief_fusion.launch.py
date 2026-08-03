from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_auvs", default_value="3"),
        DeclareLaunchArgument("run_dir", default_value="~/swarm_ws/live_run"),
        Node(
            package="belief_fusion",
            executable="belief_fusion_node",
            name="belief_fusion_node",
            output="screen",
            parameters=[{
                "num_auvs": LaunchConfiguration("num_auvs"),
                "run_dir": LaunchConfiguration("run_dir"),
            }],
        ),
    ])
