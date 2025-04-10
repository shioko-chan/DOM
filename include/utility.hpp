#ifndef ORTHO_UTILITY_HPP
#define ORTHO_UTILITY_HPP

#include <cmath>
#include <filesystem>
#include <ranges>

#include <opencv2/opencv.hpp>

#include "log.hpp"

namespace fs = std::filesystem;

namespace Ortho {
template <typename T>
  requires std::is_arithmetic_v<T>
using Point = cv::Point_<T>;

template <typename T>
  requires std::is_arithmetic_v<T>
using Points = std::vector<Point<T>>;

template <std::ranges::range Range>
auto min_x(const Range& points) {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::x).x;
}

template <std::ranges::range Range>
auto min_y(const Range& points) {
  return std::ranges::min(points, {}, &std::ranges::range_value_t<Range>::y).y;
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
  if(!cv::isContourConvex(points0) || !cv::isContourConvex(points1)) {
    throw std::runtime_error("Image has non-convex span");
  }
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

void check_and_create_path(const fs::path& path) {
  std::error_code ec;
  fs::create_directories(path, ec);
  if(ec) {
    ERROR("{} could not be created: {}", path.string(), ec.message());
    throw std::runtime_error(ec.message());
  }
}
} // namespace Ortho

namespace cv {
template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator*(const Mat& lhs, const Point_<T>& rhs) {
  if(lhs.cols != 2 && lhs.cols != 3) {
    throw std::runtime_error("Matrix must have 2 or 3 columns");
  }
  if(lhs.cols == 2) {
    return lhs * (Mat_<T>(2, 1) << rhs.x, rhs.y);
  } else {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, 1);
  }
}

template <typename T>
  requires std::is_arithmetic_v<T>
Mat operator*(const Mat& lhs, const Point3_<T>& rhs) {
  if(lhs.cols != 3 && lhs.cols != 4) {
    throw std::runtime_error("Matrix must have 3 or 4 columns");
  }
  if(lhs.cols == 3) {
    return lhs * (Mat_<T>(3, 1) << rhs.x, rhs.y, rhs.z);
  } else {
    return lhs * (Mat_<T>(4, 1) << rhs.x, rhs.y, rhs.z, 1);
  }
}
} // namespace cv
#endif