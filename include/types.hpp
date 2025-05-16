#ifndef SKYMERGE_TYPES_HPP
#define SKYMERGE_TYPES_HPP

#include <array>
#include <cstdint>
#include <functional>
#include <mutex>
#include <ostream>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

#include <pcl/common/common.h>

namespace SkyMerge {

template <typename T>
void hash_append(uint64_t& seed, const T& val) noexcept {
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
}

template <typename... Args>
auto hash(const Args&... args) noexcept -> uint64_t {
  uint64_t seed = 0;
  (hash_append(seed, args), ...);
  return seed;
}

using RotateAxisAngle = std::array<double, 3>;
using CameraArray     = std::array<double, 4>;
using DistortArray    = std::array<double, 5>;
using TranslateArray  = std::array<double, 3>;
template <typename T>
using USets = std::vector<std::unordered_set<T>>;

template <typename T>
using Sets = std::vector<std::set<T>>;

struct alignas(32) Match {
  size_t lhs, rhs;
  double score;
};

using Matches = std::vector<Match>;

struct alignas(16) PointIdx {
  int    img_idx;
  size_t pnt_idx;

  auto operator<=>(const PointIdx&) const        = default;
  auto operator==(const PointIdx&) const -> bool = default;

  friend auto operator<<(std::ostream& ostream, const PointIdx& idx) -> std::ostream& {
    return ostream << "(" << idx.img_idx << ", " << idx.pnt_idx << ")";
  }
};

struct PointIdxHasher {
  auto operator()(const PointIdx& point_idx) const noexcept -> uint64_t {
    return hash(point_idx.img_idx, point_idx.pnt_idx);
  }
};

template <typename T>
using PointIdxUMap = std::unordered_map<PointIdx, T, PointIdxHasher>;

template <typename T>
using PointIdxUMapRev = std::unordered_map<T, PointIdx>;

using PointIdxUSet = std::unordered_set<PointIdx, PointIdxHasher>;

using PointIdxs = std::vector<PointIdx>;

using Lock = std::unique_lock<std::mutex>;

using Tracks = std::vector<PointIdxs>;

struct alignas(64) TrackPoint {
  std::array<double, 3> pnt3d;
  PointIdxs             pnt2d_idx_vec;
};

using TrackPointVec = std::vector<TrackPoint>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point = cv::Point_<T>;

template <typename T>
  requires std::is_arithmetic_v<T>
struct PointHasher {
  auto operator()(const Point<T>& point) const -> uint64_t { return hash(point.x, point.y); }
};

template <typename T>
  requires std::is_arithmetic_v<T>
using Points = std::vector<Point<T>>;

template <typename T>
  requires std::is_arithmetic_v<T>
using PointSet = std::set<Point<T>>;

template <typename T>
  requires std::is_arithmetic_v<T>
using PointUSet = std::unordered_set<Point<T>, PointHasher<T>>;

template <typename T, typename U>
  requires std::is_arithmetic_v<T>
using PointUMap = std::unordered_map<Point<T>, U, PointHasher<T>>;

template <typename U, typename T>
  requires std::is_arithmetic_v<T>
using PointUMapRev = std::unordered_map<U, Point<T>>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point3 = cv::Point3_<T>;

template <typename T>
  requires std::is_arithmetic_v<T>
struct Point3Hasher {
  auto operator()(const Point3<T>& point) const -> uint64_t { return hash(point.x, point.y, point.z); }
};

template <typename T>
  requires std::is_arithmetic_v<T>
using Point3s = std::vector<Point3<T>>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point3Set = std::set<Point3<T>>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point3USet = std::unordered_set<Point3<T>, Point3Hasher<T>>;

using PointCloudPtr = typename pcl::PointCloud<pcl::PointXYZ>::Ptr;
using PointCloud    = pcl::PointCloud<pcl::PointXYZ>;
} // namespace SkyMerge
#endif
