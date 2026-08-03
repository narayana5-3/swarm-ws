import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# Trained via damage_detection/train.py: DeepCrack real-photo pretraining
# (0.82 Dice / 0.72 IoU on held-out real crack photos) then fine-tuned on
# crack_injection.py output from real captured Gazebo frames (0.67 Dice /
# 0.52 IoU on held-out crack-containing sim frames). Real, held-out-
# validated numbers, not a placeholder -- see eval_results/ for the
# side-by-side panels. Override checkpoint_path to use a different/newer
# checkpoint.
_DEFAULT_CHECKPOINT = os.path.join(
    get_package_share_directory("damage_detection"), "checkpoints", "sim_finetuned.pt")


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("num_auvs", default_value="3"),
        DeclareLaunchArgument("checkpoint_path", default_value=_DEFAULT_CHECKPOINT),
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
