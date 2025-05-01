#ifndef ROTATE_RECTIFY_HPP
#define ROTATE_RECTIFY_HPP

#include <iterator>
#include <opencv2/core.hpp>
#include <ranges>

#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "tools/debug.hpp"
#include "tools/utility.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

struct alignas(128) RectifyResult {
  cv::Mat        img;
  Points<double> pixel_span;
  cv::Mat        pers_mat;
};

inline auto rotate_rectify(const cv::Mat& R_cam2world, const cv::Mat& img) noexcept -> RectifyResult {
  auto [width, height] = img.size();
  THIS_ASSERTION_SHOULD_LEQ(4, width, "Image size is too small");
  THIS_ASSERTION_SHOULD_LEQ(4, height, "Image size is too small");
  Points<double> src{{0., 0.}, {1. * (width - 1), 0.}, {1. * (width - 1), 1. * (height - 1)}, {0., 1. * (height - 1)}};
  Points<double>
       pixel_span{{2., 2.}, {1. * (width - 3), 2.}, {1. * (width - 3), 1. * (height - 3)}, {2., 1. * (height - 3)}};
  auto view0 =
      src | std::views::transform([&R_cam2world, width, height](const Point<double>& point) noexcept -> Point<double> {
        cv::Mat point_ = (cv::Mat_<double>(2, 1) << point.x - (width / 2.), point.y - (height / 2.));
        cv::normalize(point_, point_);
        point_.push_back(1.);
        cv::Mat ray  = R_cam2world * point_;
        double  homo = ray.at<double>(2, 0);
        return {ray.at<double>(0, 0) / homo, ray.at<double>(1, 0) / homo};
      });
  auto           rect     = bounding_rect(view0);
  auto           view1    = view0 | std::views::transform([rect](const Point<double>& point) noexcept -> Point<double> {
                 return {point.x - rect.x, point.y - rect.y};
               });
  double         max_side = std::max(rect.width, rect.height);
  double         factor   = 1.0 * static_cast<double>(FEATURE_EXTRACTOR_RESOLUTION_LIM) / max_side;
  auto           view2 = view1 | std::views::transform([factor](const Point<double>& point) noexcept -> Point<double> {
                 return {point.x * factor, point.y * factor};
               });
  Points<double> dst{view2.begin(), view2.end()};
  auto           rect1 = bounding_rect(dst);
  Points<float>  src_float;
  Points<float>  dst_float;
  std::ranges::move(convert_arithmetic_type<float>(src), std::back_inserter(src_float));
  std::ranges::move(convert_arithmetic_type<float>(dst), std::back_inserter(dst_float));
  cv::Mat  perspective_mat = cv::getPerspectiveTransform(src_float, dst_float);
  cv::Mat  img_res;
  cv::Size size = cv::Size(std::ceil(rect1.width), std::ceil(rect1.height));
  cv::warpPerspective(img, img_res, perspective_mat, size, cv::INTER_CUBIC);
  Points<double> pixel_span_after;
  cv::perspectiveTransform(pixel_span, pixel_span_after, perspective_mat);
  perspective_mat.convertTo(perspective_mat, CV_64F);
  return {
      .img        = std::move(img_res),
      .pixel_span = std::move(pixel_span_after),
      .pers_mat   = std::move(perspective_mat),
  };
}
} // namespace Ortho
#endif
