from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    pkg_dis_tutorial3 = get_package_share_directory('dis_tutorial3')

    # Arguments
    arguments = [
        DeclareLaunchArgument('rviz', default_value='true', description='Launch RViz2'),
        DeclareLaunchArgument('world', default_value='task1', description='Simulation world'),
        DeclareLaunchArgument('model', default_value='standard', description='Turtlebot4 model'),
        DeclareLaunchArgument('map', default_value=PathJoinSubstitution([pkg_dis_tutorial3, 'maps', 'map.yaml']), description='Map YAML')
    ]

    # Step 0: Base simulation and nav2 (disable default RViz inside it)
    sim_base = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([pkg_dis_tutorial3, 'launch', 'sim_turtlebot_nav.launch.py'])
        ),
        launch_arguments={'rviz': 'false'}.items()  # 👈 prevent launching built-in RViz
    )

    # RViz2 node with your custom config
    rviz_config_path = PathJoinSubstitution([pkg_dis_tutorial3, 'rviz', 'exported_config.rviz'])
    rviz_node = TimerAction(
    period=2.0,  # Wait until all nodes are running
    actions=[Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )]
)

    # Step 1: Launch planner.py
    planner_node = TimerAction(
        period=10.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='planner.py',
            name='inspection_planner',
            output='screen'
        )]
    )

    # Step 2: detect_people.py
    detect_people_node = TimerAction(
        period=12.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='detect_people.py',
            name='detect_people',
            output='screen'
        )]
    )

    # Step 3: detect_rings.py
    detect_rings_node = TimerAction(
        period=14.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='detect_rings.py',
            name='detect_rings',
            output='screen'
        )]
    )

    # Step 4: face_search.py
    face_search_node = TimerAction(
        period=16.0,
        actions=[Node(
            package='dis_tutorial3',
            executable='face_search.py',
            name='face_search',
            output='screen'
        )]
    )

    return LaunchDescription(arguments + [
        sim_base,
        rviz_node,
        planner_node,
        detect_people_node,
        detect_rings_node,
        face_search_node
    ])
