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

    // 计算世界坐标系边界
    computeWorldBounds();

    // 创建结果图像
    cv::Mat result(
        static_cast<int>(world_height * scale), static_cast<int>(world_width * scale), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat resultMask(result.size(), CV_8UC1, cv::Scalar(0));

    // 使用多波段混合器
    cv::detail::MultiBandBlender blender(false, 5);
    blender.prepare(cv::Rect(0, 0, result.cols, result.rows));

    LOG_INFO("开始拼接 {} 张图片", imgs_data.size());

    for(size_t i = 0; i < imgs_data.size(); ++i) {
      ImgData& img_data = imgs_data[i];

      // 获取原始图像和掩码
      cv::Mat srcImg  = img_data.get_img().get();
      cv::Mat srcMask = img_data.get_mask().get();

      // 计算从源图像到目标图像的变换矩阵
      cv::Mat transformMatrix = calculateTransformMatrix(img_data);

      // 进行图像变换
      cv::Mat warped, warpedMask;
      cv::warpAffine(
          srcImg, warped, transformMatrix, result.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::warpAffine(
          srcMask, warpedMask, transformMatrix, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

      // 将图像添加到混合器
      blender.feed(warped, warpedMask, cv::Point(0, 0));
    }

    // 混合图像
    blender.blend(result, resultMask);

    // 保存结果
    std::string result_path = (temporary_save_path / "stitched_result.jpg").string();
    cv::imwrite(result_path, result);
    LOG_INFO("拼接结果已保存至: {}", result_path);

    return result;
  }

private:

  ImgsData& imgs_data;
  fs::path  temporary_save_path;
  float     scale;

  // 世界坐标系边界
  float world_min_x  = 0.0f;
  float world_min_y  = 0.0f;
  float world_max_x  = 0.0f;
  float world_max_y  = 0.0f;
  float world_width  = 0.0f;
  float world_height = 0.0f;

  void computeWorldBounds() {
    if(imgs_data.empty()) {
      return;
    }

    // 初始化边界值
    world_min_x = std::numeric_limits<float>::max();
    world_min_y = std::numeric_limits<float>::max();
    world_max_x = std::numeric_limits<float>::lowest();
    world_max_y = std::numeric_limits<float>::lowest();

    // 计算所有图像在世界坐标系中的边界
    for(size_t i = 0; i < imgs_data.size(); ++i) {
      ImgData& img_data = imgs_data[i];

      // 获取图像尺寸
      cv::Size img_size = img_data.get_size();

      // 获取图像四个角点
      std::vector<cv::Point2f> src_corners =
          {cv::Point2f(0, 0),
           cv::Point2f(img_size.width - 1, 0),
           cv::Point2f(img_size.width - 1, img_size.height - 1),
           cv::Point2f(0, img_size.height - 1)};
      auto                     v = src_corners | std::views::transform([&img_data](const auto& pnt) {
                 // 2. 通过逆内参矩阵K_bproj转换到相机坐标系
                 cv::Mat cam_ray = img_data.K_bproj() * pnt;
                 // 3. 通过逆旋转矩阵R_bproj转换到世界坐标系方向向量
                 cv::Mat world_dir = img_data.R_bproj() * cam_ray;
                 // 4. 获取相机原点在世界坐标系的坐标 (已含平移补偿)
                 cv::Mat camera_origin = img_data.t_bproj();
                 // 5. 计算与z=0平面的交点参数lambda
                 float lambda = -camera_origin.at<float>(2) / world_dir.at<float>(2);
                 // 6. 计算交点坐标
                 cv::Mat intersect = camera_origin + lambda * world_dir;
                 return cv::Point2f(intersect.at<float>(0), intersect.at<float>(1));
               });
      std::vector<cv::Point2f> world_corners(v.begin(), v.end());

      for(const auto& corner : world_corners) {
        world_min_x = std::min(world_min_x, corner.x);
        world_min_y = std::min(world_min_y, corner.y);
        world_max_x = std::max(world_max_x, corner.x);
        world_max_y = std::max(world_max_y, corner.y);
      }
    }
    world_width  = world_max_x - world_min_x;
    world_height = world_max_y - world_min_y;

    LOG_INFO("世界坐标系范围: ({}, {}) - ({}, {})", world_min_x, world_min_y, world_max_x, world_max_y);
    LOG_INFO("世界坐标系尺寸: {} x {}", world_width, world_height);
  }

  cv::Mat calculateTransformMatrix(ImgData& img_data) {
    cv::Size                 img_size = img_data.get_size();
    std::vector<cv::Point2f> src_corners =
        {cv::Point2f(0, 0),
         cv::Point2f(img_size.width - 1, 0),
         cv::Point2f(img_size.width - 1, img_size.height - 1),
         cv::Point2f(0, img_size.height - 1)};
    auto                     v = src_corners | std::views::transform([&img_data](const auto& pnt) {
               // 2. 通过逆内参矩阵K_bproj转换到相机坐标系
               cv::Mat cam_ray = img_data.K_bproj() * pnt;
               // 3. 通过逆旋转矩阵R_bproj转换到世界坐标系方向向量
               cv::Mat world_dir = img_data.R_bproj() * cam_ray;
               // 4. 获取相机原点在世界坐标系的坐标 (已含平移补偿)
               cv::Mat camera_origin = img_data.t_bproj();
               // 5. 计算与z=0平面的交点参数lambda
               float lambda = -camera_origin.at<float>(2) / world_dir.at<float>(2);
               // 6. 计算交点坐标
               cv::Mat intersect = camera_origin + lambda * world_dir;
               return cv::Point2f(intersect.at<float>(0), intersect.at<float>(1));
             });
    std::vector<cv::Point2f> world_corners(v.begin(), v.end());

    std::vector<cv::Point2f> dst_corners;
    for(const auto& corner : world_corners) {
      dst_corners.push_back(cv::Point2f((corner.x - world_min_x) * scale, (corner.y - world_min_y) * scale));
    }
    return cv::estimateAffinePartial2D(src_corners, dst_corners);
  }
};

} // namespace Ortho

#endif // ORTHO_STITCHER_HPP
