file(GLOB_RECURSE SOURCES *.cpp)

find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(exiv2 REQUIRED)

set(exiv2_INCLUDE_DIR ${exiv2_DIR}/../../../include)
set(exiv2_LIB_DIR ${exiv2_DIR}/../..)

set(onnxruntime_INCLUDE_DIR ${onnxruntime_DIR}/../../../include)
set(onnxruntime_LIB_DIR ${onnxruntime_DIR}/../..)


add_executable(orthophoto ${SOURCES})

target_include_directories(orthophoto PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${onnxruntime_INCLUDE_DIR}
  ${exiv2_INCLUDE_DIR}
)

target_link_directories(orthophoto PUBLIC
  ${OpenCV_LIB_DIR}
  ${onnxruntime_LIB_DIR}
  ${exiv2_LIB_DIR}
)
set(STATIC_FILE_PREFIX ${CMAKE_BINARY_DIR}/data/static)
set(MODEL_FILE_PREFIX ${STATIC_FILE_PREFIX}/models)
target_compile_definitions(orthophoto PRIVATE SENSOR_WIDTH_DATABASE="${STATIC_FILE_PREFIX}/sensor_width_camera_database.txt")
target_compile_definitions(orthophoto PRIVATE SUPERPOINT_WEIGHT="${MODEL_FILE_PREFIX}/superpoint.onnx")
target_compile_definitions(orthophoto PRIVATE SUPERPOINT_LIGHTGLUE_WEIGHT="${MODEL_FILE_PREFIX}/superpoint_lightglue_fused.onnx")
target_compile_definitions(orthophoto PRIVATE DISK_WEIGHT="${MODEL_FILE_PREFIX}/disk.onnx")
target_compile_definitions(orthophoto PRIVATE DISK_LIGHTGLUE_WEIGHT="${MODEL_FILE_PREFIX}/disk_lightglue_fused.onnx")
target_compile_definitions(orthophoto PRIVATE ${LOG_LEVEL})

target_link_libraries(orthophoto PUBLIC
  exiv2
  ${OpenCV_LIBS}
  onnxruntime
)

install(TARGETS orthophoto
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY ../include DESTINATION include)
