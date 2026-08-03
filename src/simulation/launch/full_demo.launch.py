"""
Single-command bringup for the full live demo: Gazebo world + AUVs, then
every M2 perception/fusion/allocation/control/dashboard node, in the
dependency order each one actually needs (occupancy before belief_fusion,
belief_fusion before cbba_allocator, cbba_allocator before swarm_control --
each stage reads files/topics the previous stage produces).

Usage:
    ros2 launch simulation full_demo.launch.py num_auvs:=3
    ros2 launch simulation full_demo.launch.py num_auvs:=3 headless:=true  # CI/verification, no GUI

checkpoint_path (damage_detection) defaults to empty -- an untrained model
runs by default (see damage_detection/infer_node.py's own warning); pass a
real train.py checkpoint before the actual judged demo.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def _include(package, launch_file, launch_arguments, delay_sec):
    path = os.path.join(get_package_share_directory(package), "launch", launch_file)
    action = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path), launch_arguments=launch_arguments.items())
    return TimerAction(period=delay_sec, actions=[action])


def generate_launch_description():
    num_auvs = LaunchConfiguration("num_auvs")
    run_dir = LaunchConfiguration("run_dir")
    headless = LaunchConfiguration("headless")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    port = LaunchConfiguration("port")

    return LaunchDescription([
        DeclareLaunchArgument("num_auvs", default_value="3"),
        DeclareLaunchArgument("run_dir", default_value="~/swarm_ws/live_run"),
        DeclareLaunchArgument("headless", default_value="false"),
        DeclareLaunchArgument("checkpoint_path", default_value=""),
        DeclareLaunchArgument("port", default_value="8080"),

        # Sim needs to be up and spawning before anything else has topics to
        # subscribe to; the other stages stagger in behind it so each one's
        # first tick already has real data from the stage before it.
        _include("simulation", "simulation.launch.py",
                 {"num_auvs": num_auvs, "headless": headless}, 0.0),
        _include("occupancy_mapping", "occupancy_mapping.launch.py",
                 {"num_auvs": num_auvs, "run_dir": run_dir}, 8.0),
        _include("damage_detection", "damage_detection.launch.py",
                 {"num_auvs": num_auvs, "checkpoint_path": checkpoint_path}, 8.0),
        _include("belief_fusion", "belief_fusion.launch.py",
                 {"num_auvs": num_auvs, "run_dir": run_dir}, 12.0),
        _include("cbba_allocator", "cbba_allocator.launch.py",
                 {"num_auvs": num_auvs, "run_dir": run_dir}, 20.0),
        _include("swarm_control", "swarm_control.launch.py",
                 {"num_auvs": num_auvs, "run_dir": run_dir}, 20.0),
        _include("dashboard", "dashboard.launch.py",
                 {"num_auvs": num_auvs, "run_dir": run_dir, "port": port}, 12.0),
    ])
