#ifndef ORTHO_CONFIG_HPP
#define ORTHO_CONFIG_HPP

namespace Ortho {

constexpr float LIGHTGLUE_THRESHOLD{0.2f};
constexpr int   INLIER_CNT_THRESHOLD{25};

constexpr float SUPERPOINT_THRESHOLD{0.05f};
constexpr int   SUPERPOINT_KEYPOINT_MAXCNT{1024};

constexpr float DISK_THRESHOLD{0.05f};
constexpr int   DISK_KEYPOINT_MAXCNT{1024};

constexpr int   NEIGHBOR_PROPOSAL{8};
constexpr float IOU_THRESHOLD{0.5f};

constexpr unsigned long MEM_LIMIT{4ul * (1ul << 30) /* 4GB */};

constexpr float SPATIAL_RESOLUTION = 0.1f; // meters per pixel

} // namespace Ortho

#endif