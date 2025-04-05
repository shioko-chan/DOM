#ifndef ORTHO_UTILITY_HPP
#define ORTHO_UTILITY_HPP

#include <cmath>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace Ortho {

template <typename T>
using Point = cv::Point_<T>;

template <typename T>
using Points = std::vector<Point<T>>;

template <typename T>
T min_x(const Points<T>& points) {
  return std::min_element(points.begin(), points.end(), [](const Point<T>& a, const Point<T>& b) { return a.x < b.x; })->x;
}

template <typename T>
T min_y(const Points<T>& points) {
  return std::min_element(points.begin(), points.end(), [](const Point<T>& a, const Point<T>& b) { return a.y < b.y; })->y;
}

template <typename T>
T max_x(const Points<T>& points) {
  return std::max_element(points.begin(), points.end(), [](const Point<T>& a, const Point<T>& b) { return a.x < b.x; })->x;
}

template <typename T>
T max_y(const Points<T>& points) {
  return std::max_element(points.begin(), points.end(), [](const Point<T>& a, const Point<T>& b) { return a.y < b.y; })->y;
}

template <typename T>
float avg_x(const Points<T>& points) {
  return 1.0f
         * std::accumulate(
             points.begin(), points.end(), 0, [](const T& sum, const Point<T>& point) { return sum + point.x; })
         / points.size();
}

template <typename T>
float avg_y(const Points<T>& points) {
  return 1.0f
         * std::accumulate(
             points.begin(), points.end(), 0, [](const T& sum, const Point<T>& point) { return sum + point.y; })
         / points.size();
}

template <typename T>
Point<float> avg(const Points<T>& points) {
  return Point<float>(avg_x(points), avg_y(points));
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

float iou(const Points<float>& points0, const Points<float>& points1) {
  if(!cv::isContourConvex(points0) || !cv::isContourConvex(points1)) {
    throw std::runtime_error("Image has non-convex span");
  }
  const float area0 = cv::contourArea(points0), area1 = cv::contourArea(points1);
  const float area_intersect = cv::intersectConvexConvex(points0, points1, cv::noArray());
  return area_intersect / (area0 + area1 - area_intersect);
}

Points<float> intersection(const Points<float>& points0, const Points<float>& points1) {
  if(!cv::isContourConvex(points0) || !cv::isContourConvex(points1)) {
    throw std::runtime_error("Image has non-convex span");
  }
  Points<float> intersection;
  cv::intersectConvexConvex(points0, points1, intersection);
  return intersection;
}
} // namespace Ortho
#endif