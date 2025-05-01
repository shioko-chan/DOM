#ifndef ORTHO_CONFIG_HPP
#define ORTHO_CONFIG_HPP

#include <cstdint>

namespace Ortho {

constexpr double LIGHTGLUE_THRESHOLD{0.2};
constexpr int    MATCH_CNT_THRESHOLD{25};

constexpr int ORIGIN_RESOLUTION_LIM{2048};

constexpr int FEATURE_EXTRACTOR_RESOLUTION_LIM{1024};

constexpr double SUPERPOINT_THRESHOLD{0.05};
constexpr int    SUPERPOINT_KEYPOINT_MAXCNT{1024};

constexpr double DISK_THRESHOLD{0.05};
constexpr int    DISK_KEYPOINT_MAXCNT{1024};

constexpr int NEIGHBOR_PROPOSAL{8};

constexpr int64_t MEM_LIMIT{12UL * (1UL << 30U) /* 12GB */};

constexpr double SPATIAL_RESOLUTION{0.01}; // meters per pixel
constexpr double HEIGHT{125.};             // meters

enum method_t : uint8_t { SUPERPOINT, DISK };

constexpr method_t FEATURE_EXTRACTION_METHOD{DISK};

} // namespace Ortho

#endif