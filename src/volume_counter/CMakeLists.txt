project (volume_counter)

add_library(volume_counter counter.cpp counterUtils.cpp)
target_link_libraries(volume_counter ${OpenCV_LIBRARIES})




