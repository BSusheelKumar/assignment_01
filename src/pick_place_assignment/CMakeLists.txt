cmake_minimum_required(VERSION 3.8)
project(pick_place_assignment)

find_package(ament_cmake REQUIRED)
find_package(ignition-cmake2 REQUIRED)
find_package(ignition-plugin1 REQUIRED COMPONENTS register)
set(IGN_PLUGIN_VER ${ignition-plugin1_VERSION_MAJOR})
find_package(ignition-gazebo6 REQUIRED)
set(IGN_GAZEBO_VER ${ignition-gazebo6_VERSION_MAJOR})

add_library(vacuum_gripper SHARED src/vacuum_gripper.cc)
target_link_libraries(vacuum_gripper PRIVATE
  ignition-plugin${IGN_PLUGIN_VER}::ignition-plugin${IGN_PLUGIN_VER}
  ignition-gazebo${IGN_GAZEBO_VER}::ignition-gazebo${IGN_GAZEBO_VER})
install(TARGETS vacuum_gripper LIBRARY DESTINATION lib)

find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common filters)
add_executable(carton_detector_node src/carton_detector_node.cpp)
target_include_directories(carton_detector_node PUBLIC ${PCL_INCLUDE_DIRS})
ament_target_dependencies(carton_detector_node rclcpp sensor_msgs geometry_msgs tf2 tf2_ros pcl_conversions visualization_msgs)
target_link_libraries(carton_detector_node ${PCL_LIBRARIES})
install(TARGETS carton_detector_node DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY worlds launch urdf config DESTINATION share/${PROJECT_NAME})
install(PROGRAMS
  scripts/box_detector.py
  scripts/scene_manager.py
  scripts/pick_place_carton_safe.py
  scripts/add_cols_to.py
  DESTINATION lib/${PROJECT_NAME})
ament_package()
