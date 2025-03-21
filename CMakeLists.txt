cmake_minimum_required(VERSION 3.20)
project(DOM_DRONE_ORTHOPHOTO_MAP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(COPY ${CMAKE_SOURCE_DIR}/sensor_width_camera_database.txt DESTINATION ${CMAKE_BINARY_DIR}/data)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(thirdparty)
