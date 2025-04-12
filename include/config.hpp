#ifndef ORTHO_CONFIG_HPP
#define ORTHO_CONFIG_HPP

namespace Ortho {

constexpr float LIGHTGLUE_THRESHOLD{0.2f};
constexpr int   INLIER_CNT_THRESHOLD{4};

constexpr float SUPERPOINT_THRESHOLD{0.05f};
constexpr int   SUPERPOINT_KEYPOINT_MAXCNT{1024};

constexpr float DISK_THRESHOLD{0.05f};
constexpr int   DISK_KEYPOINT_MAXCNT{1024};

constexpr int   NEIGHBOR_PROPOSAL{8};
constexpr float IOU_THRESHOLD{0.2f};

constexpr unsigned long MEM_LIMIT{16ul * (1ul << 30) /* 16GB */};

constexpr float SPATIAL_RESOLUTION{0.01f}; // meters per pixel
constexpr float HEIGHT{125.0f};            // meters

enum method_t { SUPERPOINT, DISK };

constexpr method_t FEATURE_EXTRACTION_METHOD{SUPERPOINT};

} // namespace Ortho

#endif