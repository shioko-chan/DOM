#ifndef ORTHO_CONFIG_HPP
#define ORTHO_CONFIG_HPP

#include <cstdint>

namespace Ortho {

constexpr double LIGHTGLUE_THRESHOLD{0.4};
constexpr int    MATCH_CNT_THRESHOLD{150};

constexpr int ORIGIN_RESOLUTION_LIM{2048};

constexpr int FEATURE_EXTRACTOR_RESOLUTION_LIM{1024};

constexpr double SUPERPOINT_THRESHOLD{0.1};
constexpr int    SUPERPOINT_KEYPOINT_MAXCNT{1024};

constexpr double DISK_THRESHOLD{0.05};
constexpr int    DISK_KEYPOINT_MAXCNT{1024};

constexpr int NEIGHBOR_PROPOSAL{16};

constexpr int64_t MEM_LIMIT{16UL * (1UL << 30U) /* 16GB */};

constexpr double SPATIAL_RESOLUTION{0.01}; // meters per pixel
constexpr double HEIGHT{125.0};            // meters

enum method_t : uint8_t { SUPERPOINT, DISK };

constexpr method_t FEATURE_EXTRACTION_METHOD{DISK};

} // namespace Ortho

#endif