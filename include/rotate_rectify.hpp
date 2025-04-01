#ifndef ROTATE_RECTIFY_HPP
#define ROTATE_RECTIFY_HPP

#include <ranges>

#include <opencv2/opencv.hpp>

#include "pose_intrinsic.hpp"
#include "utility.hpp"

namespace views = std::views;

namespace Ortho {

using MatAndMask = std::pair<cv::Mat, cv::Mat>;

MatAndMask rotate_rectify(const cv::Size img_size, const Pose& pose, const Intrinsic& intrinsic, const cv::Mat& img) {
  auto [w, h] = img_size;
  if(w < 7 || h < 7) {
    throw std::runtime_error("Image size is too small");
  }
  std::vector<cv::Point2f> src =
      {cv::Point2f(w - 1, 0), cv::Point2f(0, 0), cv::Point2f(0, h - 1), cv::Point2f(w - 1, h - 1)};
  cv::Mat mask = cv::Mat::ones(h - 6, w - 6, CV_8UC1) * 255;
  cv::copyMakeBorder(mask, mask, 3, 3, 3, 3, cv::BORDER_CONSTANT, cv::Scalar(0));
  auto pipeline = [&pose, &intrinsic](const std::vector<cv::Point2f>& src, const cv::Mat& img) {
    auto v0 = src | views::transform([&pose, &intrinsic](auto&& point) {
                cv::Mat point_ = (cv::Mat_<float>(3, 1) << point.x, point.y, 1);
                cv::Mat K_inv  = intrinsic.K().inv();
                float   gamma  = -pose.t().at<float>(2, 0) / cv::Mat(pose.R().row(2) * K_inv * point_).at<float>(0, 0);
                cv::Mat xyz_w  = gamma * pose.R() * K_inv * point_ + pose.t();
                return cv::Point2f(xyz_w.at<float>(0, 0), xyz_w.at<float>(1, 0));
              });

    std::vector<cv::Point2f> world_points(v0.begin(), v0.end());

    cv::Point2f avg_point = avg(world_points);
    std::for_each(world_points.begin(), world_points.end(), [&avg_point](auto&& point) {
      point.x -= avg_point.x;
      point.y -= avg_point.y;
    });

    const float ww = max_x(world_points) - min_x(world_points), wh = max_y(world_points) - min_y(world_points);
    const float fx = pose.altitude_ref * intrinsic.K().at<float>(0, 2) * 2 / ww,
                fy = pose.altitude_ref * intrinsic.K().at<float>(1, 2) * 2 / wh;

    auto v2 = world_points | views::transform([&pose, &intrinsic](auto&& point) {
                cv::Mat p_cam = (cv::Mat_<float>(3, 1) << point.x / pose.altitude_ref, point.y / pose.altitude_ref, 1);
                cv::Mat p_img = intrinsic.K() * p_cam;
                return cv::Point2f(p_img.at<float>(0, 0), p_img.at<float>(1, 0));
              });

    std::vector<cv::Point2f> dst(v2.begin(), v2.end());

    std::vector<cv::Point> dst_int(dst.size());
    std::transform(dst.begin(), dst.end(), dst_int.begin(), [](auto&& point) {
      return cv::Point(abs_ceil(point.x), abs_ceil(point.y));
    });
    cv::Rect rect = cv::boundingRect(dst);

    std::for_each(dst.begin(), dst.end(), [&rect](auto&& point) {
      point.x -= rect.x;
      point.y -= rect.y;
    });

    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat dst_img;
    cv::warpPerspective(img, dst_img, M, cv::Size(rect.width, rect.height), cv::INTER_CUBIC);
    return dst_img;
  };

  return {pipeline(src, img), pipeline(src, mask)};
}

} // namespace Ortho

#endif