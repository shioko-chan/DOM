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
void hash_append(std::uint64_t& seed, const T& val) noexcept {
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
}

template <typename... Args>
auto hash(const Args&... args) noexcept -> std::uint64_t {
  std::uint64_t seed = 0;
  (hash_append(seed, args), ...);
  return seed;
}

constexpr size_t RotateAxisAngleSize = 3;
constexpr size_t TranslateArraySize  = 3;
constexpr size_t CameraArraySize     = 4;
constexpr size_t DistortArraySize    = 5;

constexpr size_t ResidualBlockSize = 2;
constexpr size_t Point3DSize       = 3;

using RotateAxisAngle = std::array<double, RotateAxisAngleSize>;
using TranslateArray  = std::array<double, TranslateArraySize>;
using CameraArray     = std::array<double, CameraArraySize>;
using DistortArray    = std::array<double, DistortArraySize>;

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
  auto operator()(const PointIdx& point_idx) const noexcept -> std::uint64_t {
    return hash(point_idx.img_idx, point_idx.pnt_idx);
  }
};

template <typename T>
using PointIdxUMap = std::unordered_map<PointIdx, T, PointIdxHasher>;

template <typename T>
using PointIdxUMapRev = std::unordered_map<T, PointIdx>;

using PointIdxUSet = std::unordered_set<PointIdx, PointIdxHasher>;

using PointIdxs = std::vector<PointIdx>;

using TempLock = std::lock_guard<std::mutex>;
using Lock     = std::unique_lock<std::mutex>;

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
  auto operator()(const Point<T>& point) const -> std::uint64_t { return hash(point.x, point.y); }
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
  auto operator()(const Point3<T>& point) const -> std::uint64_t { return hash(point.x, point.y, point.z); }
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

template <typename>
struct extract_arg_type;

template <template <typename> class Template, typename Arg>
struct extract_arg_type<Template<Arg>> {
  using type = Arg;
};

template <typename T>
using extract_arg_type_t = typename extract_arg_type<T>::type;

template <typename>
struct rebind_template;

template <template <typename> class Template, typename Arg>
struct rebind_template<Template<Arg>> {
  template <typename NewArg>
  using type = Template<NewArg>;
};

template <typename T, typename Arg>
using rebind_template_t = typename rebind_template<T>::template type<Arg>;

template <template <typename...> class, typename>
struct is_specialization_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_specialization_of<Template, Template<Args...>> : std::true_type {};

template <template <typename...> class Template, typename T>
inline constexpr bool is_specialization_of_v = is_specialization_of<Template, T>::value;

template <template <typename...> class Template, typename T>
concept specialization_of = is_specialization_of<Template, T>::value;

template <typename T>
concept arithmetic = std::is_arithmetic_v<std::remove_cvref_t<T>>;

template <typename T>
concept HasXY = requires(T point) {
  { point.x } -> arithmetic;
  { point.y } -> arithmetic;
};

template <typename T>
concept HasXYZ = HasXY<T> && requires(T point) {
  { point.z } -> arithmetic;
};

template <typename Func, typename... Args>
concept NoexceptCallable = std::is_nothrow_invocable_v<Func, Args...>;

template <typename Func, typename Ret, typename... Args>
concept NoexceptCallableWithRet = std::is_nothrow_invocable_r_v<Ret, Func, Args...>;

} // namespace SkyMerge
#endif
