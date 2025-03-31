#ifndef ORTHO_UTILITY_HPP
#define ORTHO_UTILITY_HPP

#include <cmath>

#include <opencv2/opencv.hpp>

namespace Ortho {

template <typename T>
using Points = std::vector<cv::Point_<T>>;

template <typename T>
T min_x(const Points<T>& points) {
  return std::min_element(
             points.begin(), points.end(), [](const cv::Point_<T>& a, const cv::Point_<T>& b) { return a.x < b.x; })
      ->x;
}

template <typename T>
T min_y(const Points<T>& points) {
  return std::min_element(
             points.begin(), points.end(), [](const cv::Point_<T>& a, const cv::Point_<T>& b) { return a.y < b.y; })
      ->y;
}

template <typename T>
T max_x(const Points<T>& points) {
  return std::max_element(
             points.begin(), points.end(), [](const cv::Point_<T>& a, const cv::Point_<T>& b) { return a.x < b.x; })
      ->x;
}

template <typename T>
T max_y(const Points<T>& points) {
  return std::max_element(
             points.begin(), points.end(), [](const cv::Point_<T>& a, const cv::Point_<T>& b) { return a.y < b.y; })
      ->y;
}

template <typename T>
float avg_x(const Points<T>& points) {
  return 1.0f
         * std::accumulate(
             points.begin(), points.end(), 0, [](const T& sum, const cv::Point_<T>& point) { return sum + point.x; })
         / points.size();
}

template <typename T>
float avg_y(const Points<T>& points) {
  return 1.0f
         * std::accumulate(
             points.begin(), points.end(), 0, [](const T& sum, const cv::Point_<T>& point) { return sum + point.y; })
         / points.size();
}

template <typename T>
cv::Point2f avg(const Points<T>& points) {
  return cv::Point2f(avg_x(points), avg_y(points));
}

float abs_ceil(float x) {
  if(x >= 0) {
    return std::ceil(x);
  } else {
    return std::floor(x);
  }
}
} // namespace Ortho
#endif