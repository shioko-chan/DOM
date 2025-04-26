#ifndef ORTHO_STITCHER_HPP
#define ORTHO_STITCHER_HPP

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "progress.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {

class Stitcher {
public:

  Stitcher() = delete;

  Stitcher(ImgsData& imgs_data, fs::path temporary_save_path, float scale = 1.0f) :
      imgs_data(imgs_data), temporary_save_path(temporary_save_path), scale(scale) {
    check_or_create_path(temporary_save_path);
  }

  cv::Mat stitch() {
    if(imgs_data.empty()) {
      throw std::runtime_error("没有图片数据！");
    }
    computeWorldBounds();
    cv::Mat result(
        static_cast<int>(world_height * scale), static_cast<int>(world_width * scale), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat resultMask(result.size(), CV_8UC1, cv::Scalar(0));
    // cv::detail::MultiBandBlender blender(false, 5);
    // blender.prepare(cv::Rect(0, 0, result.cols, result.rows));
    for(size_t i = 0; i < imgs_data.size(); ++i) {
      ImgData& img_data        = imgs_data[i];
      cv::Mat  srcImg          = img_data.get_img().get();
      cv::Mat  srcMask         = img_data.get_mask().get();
      cv::Mat  transformMatrix = calculateTransformMatrix(img_data);
      cv::Mat  warped, warpedMask;
      cv::warpAffine(
          srcImg, warped, transformMatrix, result.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::warpAffine(
          srcMask, warpedMask, transformMatrix, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      // blender.feed(warped, warpedMask, cv::Point(0, 0));
      cv::Mat tempMask;
      cv::bitwise_and(warpedMask, ~resultMask, tempMask);
      warped.copyTo(result, tempMask);
      cv::bitwise_or(resultMask, warpedMask, resultMask);
    }
    // blender.blend(result, resultMask);
    return result;
  }

private:

  ImgsData& imgs_data;
  fs::path  temporary_save_path;
  float     scale;

  float world_min_x  = 0.0f;
  float world_min_y  = 0.0f;
  float world_max_x  = 0.0f;
  float world_max_y  = 0.0f;
  float world_width  = 0.0f;
  float world_height = 0.0f;

  Points<float> img_corners(const ImgData& img_data) const {
    cv::Size img_size = img_data.get_size();
    return {
        cv::Point2f(0, 0),
        cv::Point2f(img_size.width - 1, 0),
        cv::Point2f(img_size.width - 1, img_size.height - 1),
        cv::Point2f(0, img_size.height - 1)};
  }

  Points<float> ground_corners(const ImgData& img_data) const {
    auto corners = img_corners(img_data);
    auto v       = corners | std::views::transform([&img_data](const auto& pnt) {
               cv::Mat cam_ray       = img_data.K_bproj() * pnt;
               cv::Mat world_dir     = img_data.R_bproj() * cam_ray;
               cv::Mat camera_origin = img_data.t_bproj();
               float   lambda        = -camera_origin.at<float>(2) / world_dir.at<float>(2);
               cv::Mat intersect     = camera_origin + lambda * world_dir;
               return Point<float>(intersect.at<float>(0), intersect.at<float>(1));
             });
    return Points<float>{v.begin(), v.end()};
  }

  void computeWorldBounds() {
    if(imgs_data.empty()) {
      return;
    }
    world_min_x = std::numeric_limits<float>::max(), world_min_y = std::numeric_limits<float>::max();
    world_max_x = std::numeric_limits<float>::lowest(), world_max_y = std::numeric_limits<float>::lowest();
    for(size_t i = 0; i < imgs_data.size(); ++i) {
      const auto& img_data      = imgs_data[i];
      auto        world_corners = ground_corners(img_data);
      world_min_x               = std::min(world_min_x, min_x(world_corners));
      world_min_y               = std::min(world_min_y, min_y(world_corners));
      world_max_x               = std::max(world_max_x, max_x(world_corners));
      world_max_y               = std::max(world_max_y, max_y(world_corners));
    }
    world_width  = world_max_x - world_min_x;
    world_height = world_max_y - world_min_y;

    LOG_INFO("世界坐标系范围: ({}, {}) - ({}, {})", world_min_x, world_min_y, world_max_x, world_max_y);
    LOG_INFO("世界坐标系尺寸: {} x {}", world_width, world_height);
  }

  cv::Mat calculateTransformMatrix(const ImgData& img_data) const {
    cv::Size                 img_size      = img_data.get_size();
    auto                     src_corners   = img_corners(img_data);
    auto                     world_corners = ground_corners(img_data);
    std::vector<cv::Point2f> dst_corners;
    for(const auto& corner : world_corners) {
      dst_corners.push_back(cv::Point2f((corner.x - world_min_x) * scale, (corner.y - world_min_y) * scale));
    }
    return cv::estimateAffinePartial2D(src_corners, dst_corners);
  }
};

} // namespace Ortho

#endif // ORTHO_STITCHER_HPP
