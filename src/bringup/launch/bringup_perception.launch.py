"""Launch file for camera + perception pipeline only."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_dir = get_package_share_directory('bringup')

    oak_params_file = os.path.join(bringup_dir, 'config', 'oak_params.yaml')
    perception_params_file = os.path.join(bringup_dir, 'config', 'perception_params.yaml')
    tf_params_file = os.path.join(bringup_dir, 'config', 'tf_params.yaml')

    return LaunchDescription([
        # --- Static TF: oak_camera_frame -> oak_rgb_optical_frame ---
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='oak_rgb_optical_tf',
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '0.0',
                '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
                '--frame-id', 'oak_camera_frame',
                '--child-frame-id', 'oak_rgb_optical_frame',
            ],
        ),

        # --- Static TF: oak_camera_frame -> oak_depth_optical_frame ---
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='oak_depth_optical_tf',
            arguments=[
                '--x', '0.075', '--y', '0.0', '--z', '0.0',
                '--roll', '-1.5708', '--pitch', '0.0', '--yaw', '-1.5708',
                '--frame-id', 'oak_camera_frame',
                '--child-frame-id', 'oak_depth_optical_frame',
            ],
        ),

        # --- Static TF: oak_camera_frame -> px100/base_link (placeholder) ---
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='camera_to_px100_tf',
            arguments=[
                '--x', '0.0', '--y', '0.0', '--z', '-0.20',
                '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
                '--frame-id', 'oak_camera_frame',
                '--child-frame-id', 'px100/base_link',
            ],
        ),

        # --- OAK Camera Node ---
        Node(
            package='oak_depthai_wrapper',
            executable='oak_camera_node',
            name='oak_camera_node',
            parameters=[oak_params_file],
            output='screen',
        ),

        # --- Perception Node ---
        Node(
            package='plant_perception',
            executable='perception_node',
            name='perception_node',
            parameters=[perception_params_file],
            output='screen',
        ),
    ])
