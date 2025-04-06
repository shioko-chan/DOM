#ifndef ROTATE_RECTIFY_HPP
#define ROTATE_RECTIFY_HPP

#include <functional>
#include <opencv2/opencv.hpp>
#include <ranges>

#include "pose_intrinsic.hpp"
#include "utility.hpp"

namespace Ortho {

using PointsPipeline = std::function<Points<float>(const Points<float>&)>;

struct RectifyResult {
  cv::Mat        img, mask;
  Points<float>  ground_span;
  PointsPipeline world2img;
};

template <std::ranges::range Range>
inline auto backproject(const Range& img_points, const Pose& pose, const Intrinsic& intrinsic) {
  return img_points | std::views::transform([&pose, &intrinsic](auto&& point) {
           cv::Mat point_      = (cv::Mat_<float>(3, 1) << point.x, point.y, 1);
           cv::Mat K_inv       = intrinsic.K().inv();
           float   denominator = cv::Mat(pose.R().row(2) * K_inv * point_).at<float>(0, 0);
           if(std::abs(denominator) < 1e-6) {
             return Point<float>(0, 0);
           }
           float   gamma = pose.altitude / denominator;
           cv::Mat xyz_w = gamma * pose.R() * K_inv * point_ + (cv::Mat_<float>(3, 1) << 0, 0, -pose.altitude);
           return Point<float>(xyz_w.at<float>(0, 0), xyz_w.at<float>(1, 0));
         });
}

template <std::ranges::range Range>
inline auto project(const Range& world_points, const Pose& pose, const Intrinsic& intrinsic) {
  // const float ww = max_x(world_points) - min_x(world_points), wh = max_y(world_points) - min_y(world_points);
  // const float fx = pose.altitude_ref * intrinsic.K().at<float>(0, 2) * 2 / ww,
  //             fy = pose.altitude_ref * intrinsic.K().at<float>(1, 2) * 2 / wh;
  return world_points | std::views::transform([&pose, &intrinsic](auto&& point) {
           cv::Mat p_cam = (cv::Mat_<float>(3, 1) << point.x / pose.altitude_ref, point.y / pose.altitude_ref, 1);
           cv::Mat p_img = intrinsic.K() * p_cam;
           return Point<float>(p_img.at<float>(0, 0), p_img.at<float>(1, 0));
         });
}

RectifyResult rotate_rectify(const cv::Size img_size, const Pose& pose, const Intrinsic& intrinsic, const cv::Mat& img) {
  auto [w, h] = img_size;
  if(w < 5 || h < 5) {
    throw std::runtime_error("Image size is too small");
  }
  cv::Mat mask = cv::Mat::ones(h - 4, w - 4, CV_8UC1) * 255;
  cv::copyMakeBorder(mask, mask, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));
  Points        src{Point<float>(w - 1, 0), Point<float>(0, 0), Point<float>(0, h - 1), Point<float>(w - 1, h - 1)};
  auto          world_points     = backproject(src, pose, intrinsic);
  auto          abs_world_points = world_points | std::views::transform([&pose](auto&& point) {
                            return Point<float>(point.x + pose.coord.x, point.y + pose.coord.y);
                          });
  Point<float>  avg_point        = avg(world_points);
  auto          camera_points    = world_points | std::views::transform([&avg_point](auto&& point) {
                         return Point<float>(point.x - avg_point.x, point.y - avg_point.y);
                       });
  auto          img_points       = project(camera_points, pose, intrinsic);
  Points<float> dst(img_points.begin(), img_points.end());
  cv::Rect      rect = cv::boundingRect(dst);
  std::for_each(dst.begin(), dst.end(), [&rect](auto&& point) {
    point.x -= rect.x;
    point.y -= rect.y;
  });
  cv::Mat M = cv::getPerspectiveTransform(src, dst);
  cv::Mat img_res, mask_res;
  cv::warpPerspective(img, img_res, M, cv::Size(rect.width, rect.height), cv::INTER_CUBIC);
  cv::warpPerspective(mask, mask_res, M, cv::Size(rect.width, rect.height), cv::INTER_CUBIC);
  return {
      .img         = std::move(img_res),
      .mask        = std::move(mask_res),
      .ground_span = Points<float>(abs_world_points.begin(), abs_world_points.end()),
      .world2img   = PointsPipeline([rect, avg_point, &pose, &intrinsic](const Points<float>& abs_world_points) {
        auto camera_points =
            abs_world_points | std::views::transform([&pose, &avg_point](auto&& point) {
              return Point<float>(point.x - pose.coord.x - avg_point.x, point.y - pose.coord.y - avg_point.y);
            });
        auto img_points = project(camera_points, pose, intrinsic) | std::views::transform([&rect](auto&& point) {
                            return Point<float>(point.x - rect.x, point.y - rect.y);
                          });
        return Points<float>(img_points.begin(), img_points.end());
      })};
}

} // namespace Ortho

#endif