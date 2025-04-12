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

struct RectifyResult {
  cv::Mat        img, mask;
  Points<float>  ground_span;
  PointsPipeline world2img;
};

RectifyResult rotate_rectify(const cv::Size img_size, const Pose& pose, const Intrinsic& intrinsic, const cv::Mat& img_) {
  cv::Mat img;
  cv::resize(img_, img, img_size);
  auto [w, h] = img_size;
  if(w < 5 || h < 5) {
    throw std::runtime_error("Image size is too small");
  }
  cv::Mat mask = cv::Mat::ones(h - 4, w - 4, CV_8UC1) * 255;
  cv::copyMakeBorder(mask, mask, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));
  const Points  src{Point<float>(0, 0), Point<float>(w - 1, 0), Point<float>(w - 1, h - 1), Point<float>(0, h - 1)};
  auto          world_points = src | std::views::transform([&pose, &intrinsic](auto&& point) {
                        cv::Mat ray = pose.R() * intrinsic.K().inv() * point;
                        cv::normalize(ray, ray);
                        float       denominator = ray.at<float>(2, 0);
                        const float eps         = 0.1f;
                        if(std::abs(denominator) < eps) {
                          denominator = (denominator < 0 ? -eps : eps);
                        }
                        // const float   gamma = pose.altitude / denominator;
                        const float   gamma = HEIGHT / denominator;
                        const cv::Mat xyz_w = gamma * ray;
                        return Point<float>(xyz_w.at<float>(0, 0), -xyz_w.at<float>(1, 0));
                      });
  auto          pixel_points = world_points | std::views::transform([](auto&& point) {
                        return Point<float>(point.x / SPATIAL_RESOLUTION, point.y / SPATIAL_RESOLUTION);
                      });
  const float   min_x = Ortho::min_x(pixel_points), max_y = Ortho::max_y(pixel_points);
  auto          img_points = pixel_points | std::views::transform([min_x, max_y](auto&& point) {
                      return Point<float>(point.x - min_x, max_y - point.y);
                    });
  const int     width      = static_cast<int>(abs_ceil(Ortho::max_x(img_points)));
  const int     height     = static_cast<int>(abs_ceil(Ortho::max_y(img_points)));
  Points<float> dst(img_points.begin(), img_points.end());
  Points<float> src3(src.begin(), src.end() - 1), dst3(dst.begin(), dst.end() - 1);
  const cv::Mat M = cv::getAffineTransform(src3, dst3);
  cv::Mat       img_res, mask_res;
  cv::warpAffine(img, img_res, M, cv::Size(width, height), cv::INTER_CUBIC);
  cv::warpAffine(mask, mask_res, M, cv::Size(width, height), cv::INTER_NEAREST);
  auto abs_world_points = world_points | std::views::transform([&pose](auto&& point) {
                            return Point<float>(point.x + pose.coord.x, point.y + pose.coord.y);
                          });
  return {
      .img         = std::move(img_res),
      .mask        = std::move(mask_res),
      .ground_span = Points<float>(abs_world_points.begin(), abs_world_points.end()),
      .world2img =
          PointsPipeline([min_x = min_x, max_y = max_y, &pose, &intrinsic](const Points<float>& abs_world_points) {
            auto img_points = abs_world_points | std::views::transform([&pose, &min_x, &max_y](auto&& point) {
                                return Point<float>(
                                    (point.x - pose.coord.x) / SPATIAL_RESOLUTION - min_x,
                                    max_y - (point.y - pose.coord.y) / SPATIAL_RESOLUTION);
                              });
            return Points<float>(img_points.begin(), img_points.end());
          })};
}

} // namespace Ortho

#endif