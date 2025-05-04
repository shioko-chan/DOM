#ifndef ORTHO_UTILITY_HPP
#define ORTHO_UTILITY_HPP

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <Eigen/Dense>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/report_error.hpp"
#include "types/common_types.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

namespace fs = std::filesystem;

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

#ifdef ENABLE_VISUALIZE_OUTPUT
inline void export_pcd(const fs::path& path, const Point3s<double>& points) {
  std::ofstream file(path);
  file << "# .PCD v7 - Point Cloud Data\n";
  file << "VERSION .7\n";
  file << "FIELDS x y z\n";
  file << "SIZE 4 4 4\n";
  file << "TYPE F F F\n";
  file << "COUNT 1 1 1\n";
  file << "WIDTH " << points.size() << "\n";
  file << "HEIGHT 1\n";
  file << "VIEWPOINT 0 0 0 1 0 0 0\n";
  file << "POINTS " << points.size() << "\n";
  file << "DATA ascii\n";
  for(const auto& point : points) {
    file << std::fixed << std::setprecision(6) << point.y << " " << point.x << " " << -point.z << "\n";
  }
  file.close();
}

inline void export_pcd(const fs::path& path, const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) {
  pcl::io::savePCDFileASCII(path.string(), *cloud);
}
#endif

void print_run_time(const auto& start) noexcept {
  using namespace std::chrono_literals;
  auto end      = std::chrono::high_resolution_clock::now();
  auto duration = end - start;
  THIS_MESSAGE(
      "Function run time: {}s {}ms {}us {}ns",
      duration / 1s,
      duration % 1s / 1ms,
      duration % 1ms / 1us,
      duration % 1us / 1ns);
}

template <typename F>
auto make_timed(F&& func) noexcept {
  return [func = std::forward<F>(func)](auto&&... args) noexcept {
    auto start = std::chrono::high_resolution_clock::now();
    if constexpr(std::is_same_v<void, std::invoke_result_t<F, decltype(args)...>>) {
      std::invoke(func, std::forward<decltype(args)>(args)...);
      print_run_time(start);
    } else {
      auto result = std::invoke(func, std::forward<decltype(args)>(args)...);
      print_run_time(start);
      return result;
    }
  };
}

template <typename Func, typename... Args>
auto time_function(Func&& func, Args&&... args) noexcept {
  auto start = std::chrono::high_resolution_clock::now();
  if constexpr(std::is_same_v<void, std::invoke_result_t<Func, Args...>>) {
    std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    print_run_time(start);
  } else {
    auto result = std::invoke(std::forward<Func>(func), std::forward<Args>(args)...);
    print_run_time(start);
    return result;
  }
}

template <typename Func>
  requires std::is_nothrow_invocable_v<Func, size_t>
void run(size_t tasks, Func&& process, Progress& progress) noexcept {
  progress.reset(static_cast<int>(tasks));
#ifdef ENABLE_PARALLEL
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, tasks),
      [process = std::forward<Func>(process), &progress](const tbb::blocked_range<size_t>& range) noexcept {
        for(size_t i = range.begin(); i < range.end(); ++i, progress.update()) {
          process(i);
        }
      });
#else
  for(size_t i = 0; i < tasks; ++i, progress.update()) {
    std::forward<Func>(process)(i);
  }
#endif
}

template <typename Func>
  requires std::is_nothrow_invocable_v<Func, size_t>
void run(size_t tasks, Func&& process) noexcept {
#ifdef ENABLE_PARALLEL
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, tasks),
      [process = std::forward<Func>(process)](const tbb::blocked_range<size_t>& range) noexcept {
        for(size_t i = range.begin(); i < range.end(); ++i) {
          process(i);
        }
      });
#else
  for(size_t i = 0; i < tasks; ++i) {
    std::forward<Func>(process)(i);
  }
#endif
}

template <typename T>
  requires HasXY<T>
auto normalized2pixel(const cv::Size& size) {
  const auto& [width, height] = size;
  const double wf2            = width / 2.;
  const double hf2            = height / 2.;
  const double max2           = std::max(wf2, hf2);
  return std::views::transform([wf2, hf2, max2](const T& normalized) noexcept {
    return Point<double>{(normalized.x * max2) + wf2, (normalized.y * max2) + hf2};
  });
}

template <typename T>
  requires HasXY<T>
auto normalized2pixel(const T& normalized, const cv::Size& size) {
  const auto& [width, height] = size;
  const double wf2            = width / 2.;
  const double hf2            = height / 2.;
  const double max2           = std::max(wf2, hf2);
  return Point<double>{(normalized.x * max2) + wf2, (normalized.y * max2) + hf2};
}

inline auto rotate2qarray(cv::InputArray R_mat_input) noexcept -> RotateQArray {
  cv::Mat R_mat = R_mat_input.getMat();
  if(R_mat.type() != CV_64F) {
    R_mat.convertTo(R_mat, CV_64F);
  }
  Eigen::Matrix3d R_Eigen;
  cv::cv2eigen(R_mat, R_Eigen);
  Eigen::Quaterniond quaternion(R_Eigen);
  quaternion.normalize();
  return {quaternion.w(), quaternion.x(), quaternion.y(), quaternion.z()};
}

inline auto qarray2rotate(const RotateQArray& q_array) noexcept -> cv::Mat {
  Eigen::Quaterniond quaternion{q_array[0], q_array[1], q_array[2], q_array[3]};
  quaternion.normalize();
  Eigen::Matrix3d R_Eigen = quaternion.toRotationMatrix();
  cv::Mat         R_mat;
  cv::eigen2cv(R_Eigen, R_mat);
  return R_mat;
}

inline auto translate2array(cv::InputArray t_mat_input) noexcept -> TranslateArray {
  cv::Mat t_mat = t_mat_input.getMat();
  if(t_mat.type() != CV_64F) {
    t_mat.convertTo(t_mat, CV_64F);
  }
  return {t_mat.at<double>(0), t_mat.at<double>(1), t_mat.at<double>(2)};
}

inline auto array2translate(const TranslateArray& t_array) -> cv::Mat {
  cv::Mat t_mat = (cv::Mat_<double>(3, 1) << t_array[0], t_array[1], t_array[2]);
  return t_mat;
}

inline auto camera2array(cv::InputArray K_mat_input) noexcept -> CameraArray {
  cv::Mat K_mat = K_mat_input.getMat();
  if(K_mat.type() != CV_64F) {
    K_mat.convertTo(K_mat, CV_64F);
  }
  return {K_mat.at<double>(0, 0), K_mat.at<double>(1, 1), K_mat.at<double>(0, 2), K_mat.at<double>(1, 2)};
}

inline auto array2camera(const CameraArray& k_array) noexcept -> cv::Mat {
  // clang-format off
  cv::Mat K_mat =  (cv::Mat_<double>(3, 3) << 
    k_array[0], 0,    k_array[2],
    0,    k_array[1], k_array[3], 
    0,    0,    1);
  // clang-format on
  return K_mat;
}

inline auto distort2array(cv::InputArray d_mat_input) noexcept -> DistortArray {
  cv::Mat d_mat = d_mat_input.getMat();
  return {
      d_mat.at<double>(0),
      d_mat.at<double>(1),
      d_mat.at<double>(2),
      d_mat.at<double>(3),
      d_mat.at<double>(4),
      d_mat.at<double>(5)};
}

inline auto array2distort(const DistortArray& d_array) noexcept -> cv::Mat {
  cv::Mat d_mat = (cv::Mat_<double>(6, 1) << d_array[0], d_array[1], d_array[2], d_array[3], d_array[4], d_array[5]);
  return d_mat;
}

inline auto x_rotate_matrix(double radians) noexcept -> cv::Mat {
  // clang-format off
  cv::Mat R_mat = 
  (cv::Mat_<double>(3, 3) <<
    1, 0, 0,
    0, std::cos(radians), -std::sin(radians),
    0, std::sin(radians), std::cos(radians));
  // clang-format on
  return R_mat;
}

inline auto y_rotate_matrix(double radians) noexcept -> cv::Mat {
  // clang-format off
  cv::Mat R_mat = 
  (cv::Mat_<double>(3, 3) <<
    std::cos(radians), 0, std::sin(radians),
    0, 1, 0,
    -std::sin(radians), 0, std::cos(radians));
  // clang-format on
  return R_mat;
}

inline auto z_rotate_matrix(double radians) noexcept -> cv::Mat {
  // clang-format off
  cv::Mat R_mat = 
  (cv::Mat_<double>(3, 3) <<
    std::cos(radians), -std::sin(radians), 0,
    std::sin(radians), std::cos(radians), 0,
    0, 0, 1);
  // clang-format on
  return R_mat;
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>> || HasXYZ<std::ranges::range_value_t<Range>>
auto min_x(const Range& points) noexcept {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::x).x;
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>> || HasXYZ<std::ranges::range_value_t<Range>>
auto min_y(const Range& points) noexcept {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::y).y;
}

template <std::ranges::range Range>
  requires HasXYZ<std::ranges::range_value_t<Range>>
auto min_z(const Range& points) noexcept {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::z).z;
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>>
auto min(const Range& points) noexcept {
  return std::ranges::range_value_t<Range>{min_x(points), min_y(points)};
}

template <std::ranges::range Range>
  requires HasXYZ<std::ranges::range_value_t<Range>>
auto min(const Range& points) noexcept {
  return std::ranges::range_value_t<Range>{min_x(points), min_y(points), min_z(points)};
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>> || HasXYZ<std::ranges::range_value_t<Range>>
auto max_x(const Range& points) noexcept {
  return std::ranges::max(points, {}, &std::ranges::range_value_t<Range>::x).x;
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>> || HasXYZ<std::ranges::range_value_t<Range>>
auto max_y(const Range& points) noexcept {
  return std::ranges::max(points, {}, &std::ranges::range_value_t<Range>::y).y;
}

template <std::ranges::range Range>
  requires HasXYZ<std::ranges::range_value_t<Range>>
auto max_z(const Range& points) noexcept {
  return std::ranges::max(points, {}, &std::ranges::range_value_t<Range>::z).z;
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>>
auto max(const Range& points) noexcept {
  return std::ranges::range_value_t<Range>{max_x(points), max_y(points)};
}

template <std::ranges::range Range>
  requires HasXYZ<std::ranges::range_value_t<Range>>
auto max(const Range& points) noexcept {
  return std::ranges::range_value_t<Range>{max_x(points), max_y(points), max_z(points)};
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>> || HasXYZ<std::ranges::range_value_t<Range>>
auto avg_x(const Range& points) noexcept -> double {
  auto view = points | std::views::transform(&std::ranges::range_value_t<Range>::x);
  return 1. * std::accumulate(view.begin(), view.end(), 0.) / std::ranges::distance(points);
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>> || HasXYZ<std::ranges::range_value_t<Range>>
auto avg_y(const Range& points) noexcept -> double {
  auto view = points | std::views::transform(&std::ranges::range_value_t<Range>::y);
  return 1. * std::accumulate(view.begin(), view.end(), 0.) / std::ranges::distance(points);
}

template <std::ranges::range Range>
  requires HasXYZ<std::ranges::range_value_t<Range>>
auto avg_z(const Range& points) noexcept -> double {
  auto view = points | std::views::transform(&std::ranges::range_value_t<Range>::z);
  return 1. * std::accumulate(view.begin(), view.end(), 0.) / std::ranges::distance(points);
}

template <std::ranges::range Range>
  requires HasXY<std::ranges::range_value_t<Range>>
auto avg(const Range& points) noexcept -> Point<double> {
  return {avg_x(points), avg_y(points)};
}

template <std::ranges::range Range>
  requires HasXYZ<std::ranges::range_value_t<Range>>
auto avg(const Range& points) noexcept -> Point3<double> {
  return {avg_x(points), avg_y(points), avg_z(points)};
}

inline auto iou(const Points<double>& points0, const Points<double>& points1) noexcept -> double {
  THIS_ASSERTION_SHOULD_TRUE(cv::isContourConvex(points0) && cv::isContourConvex(points1), "non convex contour detected");
  const double area0          = cv::contourArea(points0);
  const double area1          = cv::contourArea(points1);
  const double area_intersect = cv::intersectConvexConvex(points0, points1, cv::noArray(), true);
  return area_intersect / (area0 + area1 - area_intersect);
}

inline auto intersection(const Points<double>& points0, const Points<double>& points1) noexcept -> Points<double> {
  THIS_ASSERTION_SHOULD_TRUE(cv::isContourConvex(points0) || cv::isContourConvex(points1), "non convex contour detected");
  Points<double> intersection;
  cv::intersectConvexConvex(points0, points1, intersection, true);
  return intersection;
}

inline auto abs_ceil(double input) noexcept -> double {
  if(input >= 0) {
    return std::ceil(input);
  }
  return std::floor(input);
}

inline void decimate_keep_aspect_ratio(cv::Mat* img_, int resolution) noexcept {
  double scale = std::min(1. * resolution / img_->cols, 1. * resolution / img_->rows);
  if(scale < 1.) {
    const int width  = std::min(static_cast<int>(std::round(img_->cols * scale)), resolution);
    const int height = std::min(static_cast<int>(std::round(img_->rows * scale)), resolution);
    try {
      cv::resize(*img_, *img_, cv::Size(width, height), 0.0, 0.0, cv::INTER_NEAREST);
    } catch(cv::Exception& exception) {
      report_error(exception, "An error occurred when resizing a image.");
    }
  }
}

inline void check_or_create_path(const fs::path& path) noexcept {
  std::error_code error_code;
  fs::create_directories(path, error_code);
  if(error_code) {
    report_error("{}", error_code.message());
  }
}

template <std::ranges::range Range>
auto bounding_rect(const Range& points) noexcept {
  using PointType = std::ranges::range_value_t<Range>;
  static_assert(HasXY<PointType>);
  using ArithmeticType = extract_arg_type_t<PointType>;
  auto min_point       = min(points);
  auto max_point       = max(points);
  return cv::Rect_<ArithmeticType>{min_point.x, min_point.y, max_point.x - min_point.x, max_point.y - min_point.y};
}

template <typename T, std::ranges::range Range>
  requires std::is_arithmetic_v<T>
auto convert_arithmetic_type(const Range& points) noexcept {
  using OldType = std::ranges::range_value_t<Range>;
  static_assert(std::is_arithmetic_v<extract_arg_type_t<OldType>>);
  using NewType = rebind_template_t<OldType, T>;
  return points | std::views::transform([](const auto& point) noexcept { return NewType{point}; });
}

inline auto mat2point(cv::InputArray mat_input) noexcept -> Point<double> {
  cv::Mat mat = mat_input.getMat();
  THIS_ASSERTION_SHOULD_EQ(mat.cols, 1);
  THIS_ASSERTION_SHOULD_EQ(mat.channels(), 1);
  THIS_ASSERTION_SHOULD_LEQ(2, mat.rows);
  THIS_ASSERTION_SHOULD_LEQ(mat.rows, 3);
  if(mat.depth() != CV_64F) {
    mat.convertTo(mat, CV_64F);
  }
  switch(mat.rows) {
    case 2:
      return {mat.at<double>(0, 0), mat.at<double>(1, 0)};
    case 3:
      return {mat.at<double>(0, 0) / mat.at<double>(2, 0), mat.at<double>(1, 0) / mat.at<double>(2, 0)};
    default:
      return {};
  }
}

inline auto mat2point3(cv::InputArray mat_input) noexcept -> Point3<double> {
  cv::Mat mat = mat_input.getMat();
  THIS_ASSERTION_SHOULD_EQ(mat.cols, 1);
  THIS_ASSERTION_SHOULD_EQ(mat.channels(), 1);
  THIS_ASSERTION_SHOULD_LEQ(3, mat.rows);
  THIS_ASSERTION_SHOULD_LEQ(mat.rows, 4);
  if(mat.depth() != CV_64F) {
    mat.convertTo(mat, CV_64F);
  }
  switch(mat.rows) {
    case 3:
      return {mat.at<double>(0, 0), mat.at<double>(1, 0), mat.at<double>(2, 0)};
    case 4:
      return {
          mat.at<double>(0, 0) / mat.at<double>(3, 0),
          mat.at<double>(1, 0) / mat.at<double>(3, 0),
          mat.at<double>(2, 0) / mat.at<double>(3, 0)};
    default:
      return {};
  }
}

template <typename T, typename U>
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
auto distance(const Point<T>& point0, const Point<U>& point1) noexcept -> double {
  return std::hypot(static_cast<double>(point0.x - point1.x), static_cast<double>(point0.y - point1.y));
}

template <typename T, typename U>
  requires std::is_arithmetic_v<T> && std::is_arithmetic_v<U>
auto distance(const Point3<T>& point0, const Point3<U>& point1) noexcept -> double {
  return std::hypot(
      static_cast<double>(point0.x - point1.x),
      static_cast<double>(point0.y - point1.y),
      static_cast<double>(point0.z - point1.z));
}

} // namespace Ortho

namespace cv {

template <typename T>
constexpr auto cv_type_of() noexcept -> int {
  if constexpr(std::is_same_v<T, float>) {
    return CV_32F;
  } else if constexpr(std::is_same_v<T, double>) {
    return CV_64F;
  } else if constexpr(std::is_same_v<T, int>) {
    return CV_32S;
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator*(InputArray lhs_, const Point_<T>& rhs) noexcept -> Mat {
  Mat lhs = lhs_.getMat();
  THIS_ASSERTION_SHOULD_EQ(lhs.channels(), 1);
  THIS_ASSERTION_SHOULD_EQ(lhs.type(), cv_type_of<T>());
  THIS_ASSERTION_SHOULD_LEQ(2, lhs.cols);
  THIS_ASSERTION_SHOULD_LEQ(lhs.cols, 3);
  if(lhs.cols == 2) {
    return lhs * (Mat_<T>(2, 1) << rhs.x, rhs.y);
  }
  return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, 1);
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator*(InputArray lhs_, const Point3_<T>& rhs) noexcept -> Mat {
  Mat lhs = lhs_.getMat();
  THIS_ASSERTION_SHOULD_EQ(lhs.channels(), 1);
  THIS_ASSERTION_SHOULD_EQ(lhs.type(), cv_type_of<T>());
  THIS_ASSERTION_SHOULD_LEQ(3, lhs.cols);
  THIS_ASSERTION_SHOULD_LEQ(lhs.cols, 4);
  if(lhs.cols == 3) {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
  }
  return lhs * (Mat_<T>(4, 1) << rhs.x, rhs.y, rhs.z, 1);
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator+(InputArray lhs_, const Point_<T>& rhs) noexcept -> Mat {
  Mat lhs = lhs_.getMat();
  THIS_ASSERTION_SHOULD_EQ(lhs.channels(), 1);
  THIS_ASSERTION_SHOULD_EQ(lhs.type(), cv_type_of<T>());
  assert(lhs.cols == 2 && lhs.rows == 1 || lhs.cols == 1 && lhs.rows == 2);
  if(lhs.cols == 2) {
    return lhs + (Mat_<T>(1, 2) << rhs.x, rhs.y);
  }
  return lhs + (Mat_<T>(2, 1) << rhs.x, rhs.y);
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator+(InputArray lhs_, const Point3_<T>& rhs) noexcept -> Mat {
  Mat lhs = lhs_.getMat();
  THIS_ASSERTION_SHOULD_EQ(lhs.channels(), 1);
  THIS_ASSERTION_SHOULD_EQ(lhs.type(), cv_type_of<T>());
  assert(lhs.cols == 3 && lhs.rows == 1 || lhs.cols == 1 && lhs.rows == 3);
  if(lhs.cols == 3) {
    return lhs + (Mat_<T>(1, 3) << rhs.x, rhs.y, rhs.z);
  }
  return lhs + (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator+(const Point_<T>& lhs, InputArray rhs_) noexcept -> Mat {
  return rhs_ + lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator+(const Point3_<T>& lhs, InputArray rhs_) noexcept -> Mat {
  return rhs_ + lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator-(InputArray lhs_, const Point_<T>& rhs) noexcept -> Mat {
  Point_<T> point(-rhs.x, -rhs.y);
  return lhs_ + point;
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator-(InputArray lhs_, const Point3_<T>& rhs) noexcept -> Mat {
  Point3_<T> point(-rhs.x, -rhs.y, -rhs.z);
  return lhs_ + point;
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator-(const Point_<T>& lhs, InputArray rhs) noexcept -> Mat {
  return -(rhs - lhs);
}

template <typename T>
  requires std::is_arithmetic_v<T>
auto operator-(const Point3_<T>& lhs, InputArray rhs) noexcept -> Mat {
  return -(rhs - lhs);
}

} // namespace cv
#endif
