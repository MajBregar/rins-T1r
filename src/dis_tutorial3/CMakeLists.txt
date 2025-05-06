cmake_minimum_required(VERSION 3.5)
project(dis_tutorial3)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

set(msg_files
  "msg/DetectedFace.msg"
  "msg/DetectedRing.msg"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs
)

install(DIRECTORY
  launch
  config
  worlds
  maps
  rviz
  DESTINATION share/${PROJECT_NAME}/
)

install(PROGRAMS
  scripts/detect_people.py
  scripts/robot_commander.py
  scripts/planner.py
  scripts/face_search.py
  scripts/detect_rings.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

