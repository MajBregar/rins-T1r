# DONE:
- Face detection and position estimation
- Navigation between markers

# TODO:
- Autonomous exploration
- Ring detection
- Ring color recognition
- Distinguish between ring and non-ring and 3D rings and 2D rings



# Tips and tricks
- [Official documentation](https://docs.ros.org/en/humble/index.html)
- [ROS2 Cookbook](https://github.com/mikeferguson/ros2_cookbook)
- You can create packages and services. Packages are used to group related nodes, topics, services, etc. Services are used to call functions in other packages.
- You can use `ros2 launch <package_name> <launch_file_name>` to launch a package.
- Run `colcon build --symlink-install`, this creates links to your python source files in the build folder, so changing your code should work without building the package again.
- Run `ros2 bag record` to record messages from all topics and be able to replay them later. You can also record specific topics.
- 


## ROS2 commands
- Run `ros2 launch <package_name> <launch_file_name>` to launch a package.
- Run `ros2 topic list` to list all topics.
- Run `ros2 topic echo <topic_name>` to print messages from a topic.
- Run `ros2 topic pub <topic_name> <message_type> '<message_content>'` to publish a message to a topic.
- Run `ros2 topic pub <topic_name> <message_type> '<message_content>'` to publish a message to a topic.


## Testing
- Test implementation on worlds with different face positions: There are three worlds in this repository: dis.sdf, demo1.sdf and demo2.sdf

## Tutorials
- [dis_tutorial1](https://github.com/vicoslab/dis_tutorial1)
  - ros2 commands
  - writing a **package**
  - writing a **service**
- [dis_tutorial2](https://github.com/vicoslab/dis_tutorial2)
  - writing and running **Launch files**
  - recording and replaying **bags**
- [dis_tutorial3](https://github.com/vicoslab/dis_tutorial3)
  - running the simulation with **Gazebo**
  - building and launching a **map**
  - **face detection** and **localization**
  - Sending **movement goals** from a node
- [dis_tutorial4](https://github.com/vicoslab/dis_tutorial4)
  - **Coordintate transformations**
- [dis_tutorial5](https://github.com/vicoslab/dis_tutorial5)
  - **Plane segmentation**
  - **Cylinder segmentation**
  - **Ring detection**