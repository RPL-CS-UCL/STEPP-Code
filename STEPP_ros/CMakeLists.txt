cmake_minimum_required(VERSION 3.0.2)
project(STEPP_ros)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  octomap_ros
  octomap_msgs
  pcl_conversions
  pcl_ros
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  grid_map_ros
  image_transport
  message_generation
)

add_message_files(
  FILES
  Float32Stamped.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)


find_package(PCL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp rospy sensor_msgs std_msgs nav_msgs cv_bridge message_runtime
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Add your C++ source files here
add_executable(depth_projection_synchronized src/depth_projection_synchronized.cpp) # src/utils.cpp)
target_link_libraries(depth_projection_synchronized ${catkin_LIBRARIES} ${PCL_LIBRARIES})
add_dependencies(depth_projection_synchronized ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Make the Python script executable
catkin_install_python(PROGRAMS scripts/inference_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
