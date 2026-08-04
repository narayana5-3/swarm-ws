from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_auvs", default_value="3"),
        DeclareLaunchArgument("run_dir", default_value="~/swarm_ws/live_run"),
        # See cbba_node.py's own declare_parameter comment for why this
        # defaults to 300s and not the more "adaptive-looking" 15s.
        DeclareLaunchArgument("allocation_period_sec", default_value="300.0"),
        Node(
            package="cbba_allocator",
            executable="cbba_node",
            name="cbba_allocator_node",
            output="screen",
            parameters=[{
                "num_auvs": LaunchConfiguration("num_auvs"),
                "run_dir": LaunchConfiguration("run_dir"),
                "allocation_period_sec": LaunchConfiguration("allocation_period_sec"),
            }],
        ),
    ])
