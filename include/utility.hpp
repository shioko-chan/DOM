#ifndef ORTHO_UTILITY_HPP
#define ORTHO_UTILITY_HPP

#include <cassert>
#include <cmath>
#include <filesystem>
#include <ranges>
#include <string_view>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace Ortho {
template <typename T>
  requires std::is_arithmetic_v<T>
using Point = cv::Point_<T>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Points = std::vector<Point<T>>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point3 = cv::Point3_<T>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Point3s = std::vector<Point3<T>>;

template <std::ranges::range Range>
auto min_x(const Range& points) {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::x).x;
}

template <std::ranges::range Range>
auto min_y(const Range& points) {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::y).y;
}

template <std::ranges::range Range>
Point<float> min(const Range& points) {
  return Point<float>(min_x(points), min_y(points));
}

template <std::ranges::range Range>
auto max_x(const Range& points) {
  return std::ranges::max(points, {}, &std::ranges::range_value_t<Range>::x).x;
}

template <std::ranges::range Range>
auto max_y(const Range& points) {
  return std::ranges::max(points, {}, &std::ranges::range_value_t<Range>::y).y;
}

template <std::ranges::range Range>
Point<float> max(const Range& points) {
  return Point<float>(max_x(points), max_y(points));
}

template <std::ranges::range Range>
float avg_x(const Range& points) {
  auto v = points | std::views::transform(&std::ranges::range_value_t<Range>::x);
  return 1.0f * std::accumulate(v.begin(), v.end(), 0.0f) / std::ranges::distance(points);
}

template <std::ranges::range Range>
float avg_y(const Range& points) {
  auto v = points | std::views::transform(&std::ranges::range_value_t<Range>::y);
  return 1.0f * std::accumulate(v.begin(), v.end(), 0.0f) / std::ranges::distance(points);
}

template <std::ranges::range Range>
Point<float> avg(const Range& points) {
  return Point<float>(avg_x(points), avg_y(points));
}

float iou(const Points<float>& points0, const Points<float>& points1) {
  assert(cv::isContourConvex(points0) && cv::isContourConvex(points1));
  const float area0 = cv::contourArea(points0), area1 = cv::contourArea(points1);
  const float area_intersect = cv::intersectConvexConvex(points0, points1, cv::noArray(), true);
  return area_intersect / (area0 + area1 - area_intersect);
}

Points<float> intersection(const Points<float>& points0, const Points<float>& points1) {
  if(!cv::isContourConvex(points0) || !cv::isContourConvex(points1)) {
    std::cerr << "points0: " << points0 << std::endl;
    std::cerr << "points1: " << points1 << std::endl;
    throw std::runtime_error("Image has non-convex span");
  }
  Points<float> intersection;
  cv::intersectConvexConvex(points0, points1, intersection, true);
  return intersection;
}

float abs_ceil(float x) {
  if(x >= 0) {
    return std::ceil(x);
  } else {
    return std::floor(x);
  }
}

void decimate_keep_aspect_ratio(cv::Mat* img_, cv::Size resolution = {1024, 1024}) {
  const float scale =
      std::min(resolution.width / static_cast<float>(img_->cols), resolution.height / static_cast<float>(img_->rows));
  if(scale < 1.0f) {
    const int w = std::min(static_cast<int>(std::round(img_->cols * scale)), resolution.width);
    const int h = std::min(static_cast<int>(std::round(img_->rows * scale)), resolution.height);
    cv::resize(*img_, *img_, cv::Size(w, h), 0.0, 0.0, cv::INTER_NEAREST);
  }
}

void check_or_create_path(const fs::path& path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  if(ec) {
    throw std::runtime_error(ec.message());
  }
}

template <std::ranges::range Range>
cv::Rect2f boundingRect(const Range& points) {
  Point<float> min_point = min(points);
  Point<float> max_point = max(points);
  return cv::Rect2f(min_point.x, min_point.y, max_point.x - min_point.x, max_point.y - min_point.y);
}

cv::Mat get_projection_matrix(const cv::Mat& R, const cv::Mat& t, const cv::Mat& K) {
  cv::Mat M;
  cv::hconcat(R, t, M);
  return K * M;
}

Point<float> mat2point(const cv::Mat& mat) {
  assert(mat.cols == 1 && (mat.rows == 2 || mat.rows == 3) && mat.type() == CV_32F);
  if(mat.rows == 2) {
    return Point<float>(mat.at<float>(0), mat.at<float>(1));
  } else {
    return Point<float>(mat.at<float>(0) / mat.at<float>(2), mat.at<float>(1) / mat.at<float>(2));
  }
}

Point3<float> mat2point3(const cv::Mat& mat) {
  assert(mat.cols == 1 && (mat.rows == 3 || mat.rows == 4) && mat.type() == CV_32F);
  if(mat.rows == 3) {
    return Point3<float>(mat.at<float>(0), mat.at<float>(1), mat.at<float>(2));
  } else {
    return Point3<float>(
        mat.at<float>(0) / mat.at<float>(3), mat.at<float>(1) / mat.at<float>(3), mat.at<float>(2) / mat.at<float>(3));
  }
}

} // namespace Ortho

namespace cv {

template <typename T>
constexpr int cv_type_of() {
  if constexpr(std::is_same_v<T, float>) {
    return CV_32F;
  } else if constexpr(std::is_same_v<T, double>) {
    return CV_64F;
  } else if constexpr(std::is_same_v<T, int>) {
    return CV_32S;
  } else {
    static_assert(!std::is_same_v<T, T>, "Unsupported type");
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator*(const MatExpr& lhs_, const Point_<T>& rhs) {
  Mat lhs = lhs_;
  assert((lhs.cols == 2 || lhs.cols == 3) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 2) {
    return lhs * (Mat_<T>(2, 1) << rhs.x, rhs.y);
  } else {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, 1);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator*(const MatExpr& lhs_, const Point3_<T>& rhs) {
  Mat lhs = lhs_;
  assert((lhs.cols == 3 || lhs.cols == 4) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 3) {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
  } else {
    return lhs * (Mat_<T>(4, 1) << rhs.x, rhs.y, rhs.z, 1);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const MatExpr& lhs_, const Point_<T>& rhs) {
  Mat lhs = lhs_;
  assert((lhs.cols == 2 && lhs.rows == 1 || lhs.cols == 1 && lhs.rows == 2) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 2) {
    return lhs + (Mat_<T>(1, 2) << rhs.x, rhs.y);
  } else {
    return lhs + (Mat_<T>(2, 1) << rhs.x, rhs.y);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const MatExpr& lhs_, const Point3_<T>& rhs) {
  Mat lhs = lhs_;
  assert((lhs.cols == 3 && lhs.rows == 1 || lhs.cols == 1 && lhs.rows == 3) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 3) {
    return lhs + (Mat_<T>(1, 3) << rhs.x, rhs.y, rhs.z);
  } else {
    return lhs + (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const Point_<T>& lhs, const MatExpr& rhs_) {
  return rhs_ + lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const Point3_<T>& lhs, const MatExpr& rhs_) {
  return rhs_ + lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator-(const MatExpr& lhs_, const Point_<T>& rhs) {
  Point_<T> p(-rhs.x, -rhs.y);
  return lhs_ + p;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator-(const MatExpr& lhs_, const Point3_<T>& rhs) {
  Point3_<T> p(-rhs.x, -rhs.y, -rhs.z);
  return lhs_ + p;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator*(const Mat& lhs, const Point_<T>& rhs) {
  assert((lhs.cols == 2 || lhs.cols == 3) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 2) {
    return lhs * (Mat_<T>(2, 1) << rhs.x, rhs.y);
  } else {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, 1);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator*(const Mat& lhs, const Point3_<T>& rhs) {
  assert((lhs.cols == 3 || lhs.cols == 4) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 3) {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
  } else {
    return lhs * (Mat_<T>(4, 1) << rhs.x, rhs.y, rhs.z, 1);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const Mat& lhs, const Point_<T>& rhs) {
  assert((lhs.cols == 2 && lhs.rows == 1 || lhs.cols == 1 && lhs.rows == 2) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 2) {
    return lhs + (Mat_<T>(1, 2) << rhs.x, rhs.y);
  } else {
    return lhs + (Mat_<T>(2, 1) << rhs.x, rhs.y);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const Mat& lhs, const Point3_<T>& rhs) {
  assert((lhs.cols == 3 && lhs.rows == 1 || lhs.cols == 1 && lhs.rows == 3) && lhs.type() == cv_type_of<T>());
  if(lhs.cols == 3) {
    return lhs + (Mat_<T>(1, 3) << rhs.x, rhs.y, rhs.z);
  } else {
    return lhs + (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const Point_<T>& lhs, const Mat& rhs_) {
  return rhs_ + lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator+(const Point3_<T>& lhs, const Mat& rhs_) {
  return rhs_ + lhs;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator-(const Mat& lhs, const Point_<T>& rhs) {
  Point_<T> p(-rhs.x, -rhs.y);
  return lhs + p;
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator-(const Mat& lhs, const Point3_<T>& rhs) {
  Point3_<T> p(-rhs.x, -rhs.y, -rhs.z);
  return lhs + p;
}

} // namespace cv
#endif