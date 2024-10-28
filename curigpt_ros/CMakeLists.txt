cmake_minimum_required(VERSION 3.0.2)
project(curigpt_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  trajectory_msgs
  geometry_msgs
  message_generation
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  ExecuteGroupPose.srv
)
catkin_python_setup()


generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  trajectory_msgs# Or other dependencies
)