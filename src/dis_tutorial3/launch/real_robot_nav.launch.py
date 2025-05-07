#!/usr/bin/env python3

import math
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node

# Initial position of the robot on the map (adjust to match real robot start)
px = 264
py = 314
yaw_deg = 90

# Map image info (adjust to match your actual map.pgm)
image_height = 426
resolution = 0.01
origin_x = -2.99
origin_y = -1.36

def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')

    # Declare args
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false', description='Use simulation (bag) clock'
    )
    map_file_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'map.yaml']),
        description='Path to the map yaml file'
    )

    # Compute pose for /initialpose
    x = origin_x + resolution * px
    y = origin_y + resolution * (image_height - py)
    yaw_rad = math.radians(yaw_deg)
    qz = math.sin(yaw_rad / 2)
    qw = math.cos(yaw_rad / 2)

    pose_msg = (
        f'{{"header": {{"frame_id": "map"}}, '
        f'"pose": {{"pose": {{"position": {{"x": {x:.2f}, "y": {y:.2f}, "z": 0.0}}, '
        f'"orientation": {{"z": {qz:.4f}, "w": {qw:.4f}}}}}, '
        f'"covariance": [0.25, 0, 0, 0, 0, 0, '
        f'0, 0.25, 0, 0, 0, 0, '
        f'0, 0, 0.25, 0, 0, 0, '
        f'0, 0, 0, 0.0685, 0, 0, '
        f'0, 0, 0, 0, 0.0685, 0, '
        f'0, 0, 0, 0, 0, 0.07]}}}}'
    )

    initial_pose_cmd = [
        'ros2', 'topic', 'pub', '--once',
        '/initialpose', 'geometry_msgs/msg/PoseWithCovarianceStamped',
        pose_msg
    ]

    # Launch Nav2 first — must run before AMCL
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={'use_sim_time': LaunchConfiguration('use_sim_time')}.items()
    )

    # Delay AMCL until Nav2 is up
    delayed_localization_launch = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'localization.launch.py'])
                ),
                launch_arguments={
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    'map': LaunchConfiguration('map')
                }.items()
            )
        ]
    )

    # Publish /initialpose only after AMCL is running
    initial_pose_pub = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=initial_pose_cmd,
                output='screen'
            )
        ]
    )

    # RViz
    rviz_config_path = PathJoinSubstitution([pkg_dis_tutorial3, 'rviz', 'exported_config.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return LaunchDescription([
        use_sim_time_arg,
        map_file_arg,
        nav2_launch,
        delayed_localization_launch,
        initial_pose_pub,
        rviz_node,
    ])
