#ifndef ROTATE_RECTIFY_HPP
#define ROTATE_RECTIFY_HPP

#include <iterator>
#include <ranges>

#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "tools/utility.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

struct alignas(128) RectifyResult {
  cv::Mat img, mask;
  cv::Mat pers_mat;
};

inline auto rotate_rectify(const cv::Mat& R_cam2world, const cv::Mat& img) -> RectifyResult {
  auto [w, h] = img.size();
  if(w < 5 || h < 5) {
    throw std::runtime_error("Image size is too small");
  }
  cv::Mat mask = cv::Mat::ones(h - 4, w - 4, CV_8UC1) * 255;
  cv::copyMakeBorder(mask, mask, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));
  Points<double> src{{0., 0.}, {1. * (w - 1), 0.}, {1. * (w - 1), 1. * (h - 1)}, {0., 1. * (h - 1)}};
  auto view0 = src | std::views::transform([&R_cam2world, w, h](const Point<double>& point) noexcept -> Point<double> {
                 cv::Mat point_ = (cv::Mat_<double>(2, 1) << point.x - (w / 2.), point.y - (h / 2.));
                 cv::normalize(point_, point_);
                 point_.push_back(1.);
                 cv::Mat ray  = R_cam2world * point_;
                 double  homo = ray.at<double>(2, 0);
                 return {ray.at<double>(0, 0) / homo, ray.at<double>(1, 0) / homo};
               });
  auto rect  = bounding_rect(view0);
  auto view1 = view0 | std::views::transform([rect](const Point<double>& point) noexcept -> Point<double> {
                 return {point.x - rect.x, point.y - rect.y};
               });
  double max_side = std::max(rect.width, rect.height);
  if(max_side < 1e-6F) {
    max_side = 1.F;
  }
  double         factor = IMG_SIZE / max_side;
  auto           view2  = view1 | std::views::transform([factor](const Point<double>& point) noexcept -> Point<double> {
                 return {point.x * factor, point.y * factor};
               });
  auto           rect1  = bounding_rect(view2);
  Points<double> dst{view2.begin(), view2.end()};
  Points<float>  src_float;
  Points<float>  dst_float;
  std::ranges::move(convert_arithmetic_type<float>(src), std::back_inserter(src_float));
  std::ranges::move(convert_arithmetic_type<float>(dst), std::back_inserter(dst_float));
  cv::Mat  perspective_mat = cv::getPerspectiveTransform(src_float, dst_float);
  cv::Mat  img_res;
  cv::Mat  mask_res;
  cv::Size size = cv::Size(std::ceil(rect1.width), std::ceil(rect1.height));
  cv::warpPerspective(img, img_res, perspective_mat, size, cv::INTER_CUBIC);
  cv::warpPerspective(mask, mask_res, perspective_mat, size, cv::INTER_NEAREST);
  perspective_mat.convertTo(perspective_mat, CV_64F);
  return {
      .img      = std::move(img_res),
      .mask     = std::move(mask_res),
      .pers_mat = std::move(perspective_mat),
  };
}
} // namespace Ortho
#endif
