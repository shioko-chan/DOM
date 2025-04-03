#ifndef ORTHO_STATIC_RESOURCE_H
#define ORTHO_STATIC_RESOURCE_H

#include "log.hpp"

#ifndef SUPERPOINT_WEIGHT
  #define SUPERPOINT_WEIGHT "superpoint.onnx"
#endif

#ifndef SUPERPOINT_LIGHTGLUE_WEIGHT
  #define SUPERPOINT_LIGHTGLUE_WEIGHT "superpoint_lightglue_fused.onnx"
#endif

#ifndef DISK_WEIGHT
  #define DISK_WEIGHT "disk.onnx"
#endif

#ifndef DISK_LIGHTGLUE_WEIGHT
  #define DISK_LIGHTGLUE_WEIGHT "disk_lightglue_fused.onnx"
#endif

#ifndef SENSOR_WIDTH_DATABASE
  #define SENSOR_WIDTH_DATABASE "sensor_width_database.txt"
#endif

#endif