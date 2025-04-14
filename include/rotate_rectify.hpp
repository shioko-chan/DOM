#ifndef ROTATE_RECTIFY_HPP
#define ROTATE_RECTIFY_HPP

#include <functional>
#include <opencv2/opencv.hpp>
#include <ranges>

#include "config.hpp"
#include "pose_intrinsic.hpp"
#include "utility.hpp"

namespace Ortho {

using PointsPipeline = std::function<Points<float>(const Points<float>&)>;

constexpr float square_size{4096};

struct RectifyResult {
  cv::Mat img, mask;
};

RectifyResult rotate_rectify(const cv::Size img_size, const Pose& pose, const cv::Mat& img_) {
  cv::Mat img;
  cv::resize(img_, img, img_size);
  auto [w, h] = img_size;
  if(w < 5 || h < 5) {
    throw std::runtime_error("Image size is too small");
  }
  cv::Mat mask = cv::Mat::ones(h - 4, w - 4, CV_8UC1) * 255;
  cv::copyMakeBorder(mask, mask, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));
  const Points src{Point<float>(0, 0), Point<float>(w - 1, 0), Point<float>(w - 1, h - 1), Point<float>(0, h - 1)};
  auto         v0   = src | std::views::transform([&pose, w, h](const auto& point) {
              cv::Mat point_ = (cv::Mat_<float>(2, 1) << point.x - w / 2.0f, point.y - h / 2.0f);
              cv::normalize(point_, point_);
              point_.push_back(1.0f);
              cv::Mat ray = pose.R() * point_;
              return Point<float>(ray.at<float>(0, 0) / ray.at<float>(2, 0), ray.at<float>(1, 0) / ray.at<float>(2, 0));
            });
  auto         rect = boundingRect(v0);
  auto         v1 =
      v0 | std::views::transform([rect](const auto& point) { return Point<float>(point.x - rect.x, point.y - rect.y); });
  float max_side = std::max(rect.width, rect.height);
  if(max_side < 1e-6f) {
    max_side = 1.0f;
  }
  float         factor = square_size / max_side;
  auto          v2     = v1 | std::views::transform([factor](const auto& point) {
              return Point<float>(point.x * factor, point.y * factor);
            });
  auto          rect1  = boundingRect(v2);
  Points<float> dst(v2.begin(), v2.end());
  const cv::Mat M = cv::getPerspectiveTransform(src, dst);
  cv::Mat       img_res, mask_res;
  cv::Size      size = cv::Size(std::ceil(rect1.width), std::ceil(rect1.height));
  cv::warpPerspective(img, img_res, M, size, cv::INTER_CUBIC);
  cv::warpPerspective(mask, mask_res, M, size, cv::INTER_NEAREST);
  return {
      .img  = std::move(img_res),
      .mask = std::move(mask_res),
  };
}
} // namespace Ortho
#endif