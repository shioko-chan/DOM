#ifndef ROTATE_RECTIFY_HPP
#define ROTATE_RECTIFY_HPP

#include <functional>
#include <opencv2/opencv.hpp>
#include <ranges>

#include "pose_intrinsic.hpp"
#include "utility.hpp"

namespace Ortho {

const float resolution = 0.1f; // meters per pixel
using PointsPipeline   = std::function<Points<float>(const Points<float>&)>;

struct RectifyResult {
  cv::Mat        img, mask;
  Points<float>  ground_span;
  PointsPipeline world2img;
};

RectifyResult rotate_rectify(const cv::Size img_size, const Pose& pose, const Intrinsic& intrinsic, const cv::Mat& img) {
  auto [w, h] = img_size;
  if(w < 5 || h < 5) {
    throw std::runtime_error("Image size is too small");
  }
  cv::Mat mask = cv::Mat::ones(h - 4, w - 4, CV_8UC1) * 255;
  cv::copyMakeBorder(mask, mask, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0));
  const Points   src{Point<float>(w - 1, 0), Point<float>(0, 0), Point<float>(0, h - 1), Point<float>(w - 1, h - 1)};
  auto           world_points = src | std::views::transform([&pose, &intrinsic](auto&& point) {
                        cv::Mat ray = pose.R() * intrinsic.K().inv() * point;
                        cv::normalize(ray, ray);
                        float       denominator = ray.at<float>(2, 0);
                        const float eps         = 0.1f;
                        if(std::abs(denominator) < eps) {
                          denominator = (denominator < 0 ? -eps : eps);
                        }
                        const float   gamma = pose.altitude / denominator;
                        const cv::Mat xyz_w = gamma * ray;
                        return Point<float>(xyz_w.at<float>(0, 0), xyz_w.at<float>(1, 0));
                      });
  auto           img_points   = world_points | std::views::transform([](auto&& point) {
                      return Point<float>(point.x / resolution, point.y / resolution);
                    });
  Points<float>  dst(img_points.begin(), img_points.end());
  const cv::Rect rect = cv::boundingRect(dst);
  std::for_each(dst.begin(), dst.end(), [&rect](Point<float>& point) {
    point.x -= rect.x;
    point.y -= rect.y;
  });
  const cv::Mat M = cv::getPerspectiveTransform(src, dst);
  cv::Mat       img_res, mask_res;
  cv::warpPerspective(img, img_res, M, cv::Size(rect.width, rect.height), cv::INTER_CUBIC);
  cv::warpPerspective(mask, mask_res, M, cv::Size(rect.width, rect.height), cv::INTER_CUBIC);
  auto abs_world_points = world_points | std::views::transform([&pose](auto&& point) {
                            return Point<float>(point.x + pose.coord.x, point.y + pose.coord.y);
                          });
  return {
      .img         = std::move(img_res),
      .mask        = std::move(mask_res),
      .ground_span = Points<float>(abs_world_points.begin(), abs_world_points.end()),
      .world2img   = PointsPipeline([rect, &pose, &intrinsic](const Points<float>& abs_world_points) {
        auto img_points =
            abs_world_points | std::views::transform([&pose](auto&& point) {
              return Point<float>(point.x - pose.coord.x, point.y - pose.coord.y);
            })
            | std::views::transform([](auto&& point) { return Point<float>(point.x / resolution, point.y / resolution); })
            | std::views::transform([rect](auto&& point) { return Point<float>(point.x - rect.x, point.y - rect.y); });

        return Points<float>(img_points.begin(), img_points.end());
      })};
}

} // namespace Ortho

#endif