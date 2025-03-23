cmake_minimum_required(VERSION 3.20)
project(DOM_DRONE_ORTHOPHOTO_MAP VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

file(COPY ${CMAKE_SOURCE_DIR}/sensor_width_camera_database.txt DESTINATION ${CMAKE_BINARY_DIR}/data)

include_directories(~/thirdparty/exiv2-0.28.5-Linux-x86_64/include)
link_directories(~/thirdparty/exiv2-0.28.5-Linux-x86_64/lib)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(thirdparty)
