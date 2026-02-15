"""Launch file for camera + perception + reachability checker."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('bringup')

    reachability_params_file = os.path.join(
        bringup_dir, 'config', 'reachability_params.yaml'
    )

    return LaunchDescription([
        # Include perception launch
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_dir, 'launch', 'bringup_perception.launch.py')
            ),
        ),

        # --- Reachability Checker ---
        Node(
            package='px100_integration',
            executable='reachability_checker',
            name='reachability_checker',
            parameters=[reachability_params_file],
            output='screen',
        ),
    ])
