from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.substitutions import ThisLaunchFileDir
import os
from ament_index_python.packages import get_package_share_directory


import math

#README - launches the navigation and localization stack from bag and localizes based of the starting position

#bag dir
bag_dir_name = 'bag_giga'

#start position of robot on map (when the bag was recorded)
px = 264 
py = 314
yaw_deg = 90

#map info from map.pgm
image_height = 426
resolution = 0.01
origin_x = -2.99
origin_y = -1.36

def generate_launch_description():
    global px, py, yaw_deg, image_height, resolution, origin_x, origin_y, bag_dir_name

    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')

    default_bag_path = os.path.join(pkg_dis_tutorial3, '..', '..', '..', '..', 'bags', bag_dir_name)
    default_bag_path = os.path.abspath(default_bag_path)

    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=default_bag_path,
        description='Path to the bag directory'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='true',
        description='Use simulation (bag) clock'
    )

    map_file_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'map.yaml']),
        description='Path to the map yaml file'
    )

    # Bag playback
    bag_play = ExecuteProcess(
        cmd=['ros2', 'bag', 'play', LaunchConfiguration('bag_path'), '--clock'],
        output='screen'
    )

    # Localization
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'localization.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'map': LaunchConfiguration('map')
        }.items()
    )

    # Navigation
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'nav2.launch.py'])
        ),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time')
        }.items()
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

    
    #AMCL initial pose init from bag

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

    initial_pose_pub = TimerAction(
        period=1.0,
        actions=[
            ExecuteProcess(
                cmd=initial_pose_cmd,
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        bag_path_arg,
        use_sim_time_arg,
        map_file_arg,
        bag_play,
        localization_launch,
        nav2_launch,
        initial_pose_pub,
        rviz_node,
        
    ])
