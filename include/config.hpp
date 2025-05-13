#ifndef SKYMERGE_CONFIG_HPP
#define SKYMERGE_CONFIG_HPP

#include <cstdint>

namespace SkyMerge {

constexpr double LIGHTGLUE_THRESHOLD{0.8};
constexpr int    MATCH_CNT_THRESHOLD{150};

constexpr int ORIGIN_RESOLUTION_LIM{10240};

constexpr int FEATURE_EXTRACTOR_RESOLUTION_LIM{1024};

constexpr double SUPERPOINT_THRESHOLD{0.25};
constexpr int    SUPERPOINT_KEYPOINT_MAXCNT{1024};

constexpr double DISK_THRESHOLD{0.25};
constexpr int    DISK_KEYPOINT_MAXCNT{1024};

constexpr int NEIGHBOR_PROPOSAL{16};

constexpr int64_t MEM_LIMIT{16UL * (1UL << 30U) /* 16GB */};

enum method_t : uint8_t { SUPERPOINT, DISK };

constexpr method_t FEATURE_EXTRACTION_METHOD{DISK};

constexpr double GRID_LENGTH{1.0}; // meters

constexpr double TARGET_RESOLUTION{20.0}; // pixels per meter

} // namespace SkyMerge

#endif