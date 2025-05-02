#ifndef ORTHO_STITCHER_HPP
#define ORTHO_STITCHER_HPP

#include <algorithm>
#include <filesystem>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace Ortho {

namespace fs = std::filesystem;

class Stitcher {
public:

  Stitcher() = delete;

  explicit Stitcher(const fs::path& temporary_save_path, double scale = 10.0) :
      temporary_save_path(temporary_save_path), scale(scale) {
    check_or_create_path(temporary_save_path);
  }

  auto stitch(ImgsData& imgs_data, Progress& progress) -> cv::Mat {
    progress.reset(static_cast<int>(imgs_data.size()));
    if(imgs_data.empty()) {
      return cv::Mat{};
    }
    computeWorldBounds(imgs_data);
    cv::Mat result(
        static_cast<int>(world_height * scale), static_cast<int>(world_width * scale), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat resultMask(result.size(), CV_8UC1, cv::Scalar(0));
    // cv::detail::MultiBandBlender blender(false, 5);
    // blender.prepare(cv::Rect(0, 0, result.cols, result.rows));
    for(auto& img_data : imgs_data) {
      cv::Mat srcImg          = img_data.origin_img().get();
      cv::Mat srcMask         = cv::Mat::ones(srcImg.size(), CV_8UC1) * 255;
      cv::Mat transformMatrix = calculateTransformMatrix(img_data);
      cv::Mat warped;
      cv::Mat warpedMask;
      cv::warpPerspective(
          srcImg, warped, transformMatrix, result.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::warpPerspective(
          srcMask, warpedMask, transformMatrix, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      // blender.feed(warped, warpedMask, cv::Point(0, 0));
      cv::Mat tempMask;
      cv::bitwise_and(warpedMask, ~resultMask, tempMask);
      warped.copyTo(result, tempMask);
      cv::bitwise_or(resultMask, warpedMask, resultMask);
      progress.update();
      // {
      //   cv::Mat show;
      //   cv::resize(result, show, cv::Size{}, 0.2, 0.2);
      //   cv::imshow("stitch", show);
      //   cv::waitKey(0);
      // }
    }

    // blender.blend(result, resultMask);
    return result;
  }

private:

  fs::path temporary_save_path;

  double scale;

  double world_min_x{0.}, world_min_y{0.};
  double world_max_x{0.}, world_max_y{0.};
  double world_width{0.}, world_height{0.};

  static auto img_corners(ImgData& img_data) -> Points<double> {
    cv::Size img_size = img_data.origin_img().get_size();
    double   width{static_cast<double>(img_size.width)};
    double   height{static_cast<double>(img_size.height)};
    return {
        Point<double>{0., 0.},
        Point<double>{width - 1, 0.},
        Point<double>{width - 1, height - 1},
        Point<double>{0., height - 1}};
  }

  static auto ground_corners(ImgData& img_data) -> Points<double> {
    auto corners = img_corners(img_data);
    auto view    = corners | std::views::transform([&img_data](const auto& pnt) noexcept -> Point<double> {
                  cv::Mat pnt_mat   = (cv::Mat_<double>(3, 1) << pnt.x, -pnt.y, 1.0);
                  cv::Mat world_dir = img_data.R_c2w() * mat2point(img_data.K().inv() * pnt_mat);
                  // cv::Mat world_dir = img_data.R_c2w() * mat2point(img_data.K().inv() * pnt);
                  cv::normalize(world_dir, world_dir);
                  double  lambda    = -img_data.t_c2w().at<double>(2, 0) / world_dir.at<double>(2, 0);
                  cv::Mat intersect = lambda * world_dir + img_data.t_c2w();
                  THIS_ASSERTION_SHOULD_LES(intersect.at<double>(2, 0), 1e-6);
                  return {intersect.at<double>(0, 0), intersect.at<double>(1, 0)};
                });
    return {view.begin(), view.end()};
  }

  void computeWorldBounds(ImgsData& imgs_data) {
    if(imgs_data.empty()) {
      return;
    }
    world_min_x = std::numeric_limits<double>::max(), world_min_y = std::numeric_limits<double>::max();
    world_max_x = std::numeric_limits<double>::lowest(), world_max_y = std::numeric_limits<double>::lowest();
    for(auto& img_data : imgs_data) {
      auto world_corners = ground_corners(img_data);
      world_min_x        = std::min(world_min_x, min_x(world_corners));
      world_min_y        = std::min(world_min_y, min_y(world_corners));
      world_max_x        = std::max(world_max_x, max_x(world_corners));
      world_max_y        = std::max(world_max_y, max_y(world_corners));
    }
    world_width  = world_max_x - world_min_x;
    world_height = world_max_y - world_min_y;
    THIS_LOG_INFO("世界坐标系范围: ({}, {}) - ({}, {})", world_min_x, world_min_y, world_max_x, world_max_y);
    THIS_LOG_INFO("世界坐标系尺寸: {} x {}", world_width, world_height);
  }

  [[nodiscard]] auto calculateTransformMatrix(ImgData& img_data) const -> cv::Mat {
    cv::Size       img_size      = img_data.origin_img().get_size();
    auto           src_corners   = img_corners(img_data);
    auto           world_corners = ground_corners(img_data);
    Points<double> dst_corners;
    for(const auto& corner : world_corners) {
      // dst_corners.emplace_back((corner.x - world_min_x) * scale, (corner.y - world_min_y) * scale);
      dst_corners.emplace_back((corner.x - world_min_x) * scale, (world_max_y - corner.y) * scale);
    }
    Points<float> src_float;
    Points<float> dst_float;
    {
      auto view = convert_arithmetic_type<float>(src_corners);
      src_float.assign(view.begin(), view.end());
    }
    {
      auto view = convert_arithmetic_type<float>(dst_corners);
      dst_float.assign(view.begin(), view.end());
    }
    return cv::getPerspectiveTransform(src_float, dst_float);
  }
};

} // namespace Ortho

#endif
