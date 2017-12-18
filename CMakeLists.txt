cmake_minimum_required(VERSION 2.8.3)
project(product_counter)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
set(CMAKE_AUTOUIC ON)

find_package(catkin REQUIRED COMPONENTS refills_msgs cv_bridge message_runtime image_transport
    tf tf_conversions)

find_package(OpenCV REQUIRED)

include_directories(
 include
 src/refills_volume_estimator
 ${OpenCV_INCLUDE_DIRS}
 ${catkin_INCLUDE_DIRS}
)

catkin_package(
  INCLUDE_DIRS include src/refills_volume_estimator
  
#  LIBRARIES refills_counting
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)


#hack for qt creator
file(GLOB_RECURSE FILTERING_HEADERS *.h)

add_library(volume_counter src/refills_volume_estimator/counter.cpp
			   src/refills_volume_estimator/counterUtils.cpp)
target_link_libraries(volume_counter ${OpenCV_LIBRARIES})

add_executable(counter_node 
		src/counter_node.cpp 
		src/RealSenseBridge.cpp ${FILTERING_HEADERS})
add_dependencies(counter_node refills_msgs_generate_messages_cpp)
target_link_libraries(counter_node volume_counter ${catkin_LIBRARIES})

