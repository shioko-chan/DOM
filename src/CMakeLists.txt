find_package(OpenCV REQUIRED)
find_package(onnxruntime REQUIRED)
find_package(exiv2 REQUIRED)
find_package(Ceres REQUIRED)
list(APPEND CMAKE_MODULE_PATH "/usr/share/cmake/geographiclib")
find_package(GeographicLib REQUIRED)

set(exiv2_INCLUDE_DIR ${exiv2_DIR}/../../../include)
set(exiv2_LIB_DIR ${exiv2_DIR}/../..)
set(onnxruntime_INCLUDE_DIR ${onnxruntime_DIR}/../../../include)
set(onnxruntime_LIB_DIR ${onnxruntime_DIR}/../..)

file(GLOB_RECURSE SOURCES *.cpp)

add_executable(orthophoto ${SOURCES})

target_include_directories(orthophoto PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:include>
  ${OpenCV_INCLUDE_DIRS}
  ${onnxruntime_INCLUDE_DIR}
  ${exiv2_INCLUDE_DIR}
  ${GeographicLib_INCLUDE_DIRS}
)

target_link_directories(orthophoto PUBLIC
  ${OpenCV_LIB_DIR}
  ${onnxruntime_LIB_DIR}
  ${exiv2_LIB_DIR}
)

set(MODEL_FILE_PREFIX ${CMAKE_BINARY_DIR}/data/models)
target_compile_definitions(orthophoto PRIVATE SUPERPOINT_WEIGHT="${MODEL_FILE_PREFIX}/superpoint.onnx")
target_compile_definitions(orthophoto PRIVATE SUPERPOINT_LIGHTGLUE_WEIGHT="${MODEL_FILE_PREFIX}/superpoint_lightglue_fused.onnx")
target_compile_definitions(orthophoto PRIVATE DISK_WEIGHT="${MODEL_FILE_PREFIX}/disk.onnx")
target_compile_definitions(orthophoto PRIVATE DISK_LIGHTGLUE_WEIGHT="${MODEL_FILE_PREFIX}/disk_lightglue_fused.onnx")
target_compile_definitions(orthophoto PRIVATE LOGLEVEL_${LOG_LEVEL})

if(ENABLE_VISUALIZE_OUTPUT)
  target_compile_definitions(orthophoto PRIVATE ENABLE_VISUALIZE_OUTPUT)
endif()

if(ENABLE_ASSERTION)
  target_compile_definitions(orthophoto PRIVATE ENABLE_ASSERTION)
endif()

target_link_libraries(orthophoto PUBLIC
  exiv2
  ${OpenCV_LIBS}
  onnxruntime
  ceres
  ${GeographicLib_LIBRARIES}
)

if(ENABLE_PARALLEL)
  find_package(TBB REQUIRED)
  target_compile_definitions(orthophoto PRIVATE ENABLE_PARALLEL)
  target_link_libraries(orthophoto PUBLIC
    TBB::tbb
  )
endif()

install(TARGETS orthophoto
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY ../include DESTINATION include)
