#ifndef ORTHO_ALIAS_HPP
#define ORTHO_ALIAS_HPP

#include <array>
#include <filesystem>
#include <functional>
#include <mutex>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

namespace Ortho {

namespace fs = std::filesystem;

using RotateQArray = std::array<double, 4>;
using IntrinsicArray = std::array<double, 4>;
using TransposeArray = std::array<double, 3>;
template<typename T>
using USets = std::vector<std::unordered_set<T>>;

template<typename T>
using Sets = std::vector<std::set<T>>;

using KeyPoints = std::vector<cv::KeyPoint>;

using OrtValues = std::vector<Ort::Value>;

struct Match {
  size_t lhs, rhs;
  float  score;
};

using Matches = std::vector<Match>;

struct PointIdx {
  int img_idx;
  size_t pnt_idx;
  auto operator<=>(const PointIdx&) const = default;
};

template <typename T>
void hash_append(size_t& seed, const T& val) {
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename... Args>
size_t hash(const Args&... args) {
  size_t seed = 0;
  (hash_append(seed, args), ...);
  return seed;
}

struct PointIdxHasher {
  size_t operator()(const PointIdx& p) const { return hash(p.img_idx, p.pnt_idx); }
};

template <typename T>
using PointIdxUMap = std::unordered_map<PointIdx, T, PointIdxHasher>;

template <typename T>
using PointIdxUMapRev = std::unordered_map<T, PointIdx>;

using PointIdxUSet = std::unordered_set<PointIdx, PointIdxHasher>;

using PointIdxs = std::vector<PointIdx>;

using Lock = std::unique_lock<std::mutex>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point = cv::Point_<T>;

template <typename T>
  requires std::is_arithmetic_v<T>
struct PointHasher {
  size_t operator()(const Point<T>& p) const { return hash(p.x, p.y); }
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
  size_t operator()(const Point3<T>& p) const { return hash(p.x, p.y, p.z); }
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

} // namespace Ortho
#endif
