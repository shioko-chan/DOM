#ifndef ORTHO_CV_ALIAS_HPP
#define ORTHO_CV_ALIAS_HPP

#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <opencv2/opencv.hpp>

#include "types/hash.hpp"

namespace Ortho {

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

} // namespace Ortho
#endif