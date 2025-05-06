#ifndef ORTHO_ALGO_STITCH_HPP
#define ORTHO_ALGO_STITCH_HPP

#include <filesystem>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace Ortho {
namespace fs = std::filesystem;

class DSMStitcher {
public:

  DSMStitcher() = delete;

  explicit DSMStitcher(
      const fs::path& temporary_save_path,
      double          resolution  = 0.5,
      double          world_min_x = 0,
      double          world_min_y = 0) :
      temporary_save_path(temporary_save_path), resolution(resolution), world_min_x_(world_min_x),
      world_min_y_(world_min_y) {
    check_or_create_path(temporary_save_path);
  }

  auto stitch(ImgsData& imgs_data, const cv::Mat& dsm, Progress& progress) -> cv::Mat {
    if(imgs_data.empty() || dsm.empty()) {
      THIS_MESSAGE("empty imgs_data or dsm");
      return cv::Mat{};
    }
    cv::Mat result(dsm.rows, dsm.cols, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat weights(dsm.rows, dsm.cols, CV_32F, cv::Scalar(0));
    cv::Mat resultMask(result.size(), CV_8UC1, cv::Scalar(0));
    THIS_MESSAGE("start to back project each image...");
    progress.reset(static_cast<int>(imgs_data.size() + 1));
    for(auto& img_data : imgs_data) {
      processImage(img_data, dsm, result, weights, resultMask, imgs_data.M());
      progress.update();
    }
    normalizeResult(result, weights);
    progress.update();
    THIS_MESSAGE("stitching done");
    return result;
  }

private:

  fs::path temporary_save_path;
  double   resolution;
  double   world_min_x_;
  double   world_min_y_;

  [[nodiscard]] auto backProject(const cv::Mat& dsm, int row, int col, const ImgData& img_data, const cv::Mat& M) const
      -> Point<float> {
    double  x            = world_min_x_ + (col * resolution);
    double  y            = world_min_y_ + (row * resolution);
    double  z            = dsm.at<float>(row, col);
    cv::Mat world_point  = (cv::Mat_<double>(4, 1) << x, y, z, 1.0);
    cv::Mat camera_point = img_data.R_w2c() * (world_point.rowRange(0, 3) - img_data.t_c2w());
    if(camera_point.at<double>(2, 0) <= 0) {
      return {-1, -1};
    }
    cv::Mat img_point = M * camera_point;
    auto    u         = static_cast<float>(img_point.at<double>(0, 0) / img_point.at<double>(2, 0));
    auto    v         = static_cast<float>(img_point.at<double>(1, 0) / img_point.at<double>(2, 0));
    return {u, v};
  }

  static auto isPointInImage(const cv::Point2f& point, const cv::Size& img_size) -> bool {
    return point.x >= 0 && point.y >= 0 && point.x < img_size.width && point.y < img_size.height;
  }

  void processImage(
      ImgData&       img_data,
      const cv::Mat& dsm,
      cv::Mat&       result,
      cv::Mat&       weights,
      cv::Mat&       resultMask,
      const cv::Mat& M) const {
    cv::Mat  img      = img_data.origin_img().get();
    cv::Size img_size = img.size();
    for(int row = 0; row < dsm.rows; ++row) {
      for(int col = 0; col < dsm.cols; ++col) {
        if(std::isnan(dsm.at<float>(row, col))) {
          continue;
        }
        cv::Point2f img_point = backProject(dsm, row, col, img_data, M);
        if(isPointInImage(img_point, img_size)) {
          cv::Vec3b   color;
          cv::Point2i p0(static_cast<int>(img_point.x), static_cast<int>(img_point.y));
          cv::Point2i p1(p0.x + 1, p0.y);
          cv::Point2i p2(p0.x, p0.y + 1);
          cv::Point2i p3(p0.x + 1, p0.y + 1);
          float       dx = img_point.x - p0.x;
          float       dy = img_point.y - p0.y;
          if(p1.x < img_size.width && p2.y < img_size.height && p3.x < img_size.width && p3.y < img_size.height) {
            cv::Vec3b color00 = img.at<cv::Vec3b>(p0.y, p0.x);
            cv::Vec3b color01 = img.at<cv::Vec3b>(p0.y, p1.x);
            cv::Vec3b color10 = img.at<cv::Vec3b>(p2.y, p0.x);
            cv::Vec3b color11 = img.at<cv::Vec3b>(p3.y, p3.x);
            for(int c = 0; c < 3; ++c) {
              float top    = color00[c] * (1 - dx) + color01[c] * dx;
              float bottom = color10[c] * (1 - dx) + color11[c] * dx;
              color[c]     = static_cast<uchar>(top * (1 - dy) + bottom * dy);
            }
          } else {
            color = img.at<cv::Vec3b>(p0.y, p0.x);
          }
          float weight       = 1.0f;
          auto& result_pixel = result.at<cv::Vec3f>(row, col);
          auto& pixel_weight = weights.at<float>(row, col);
          for(int c = 0; c < 3; ++c) {
            result_pixel[c] = (result_pixel[c] * pixel_weight + color[c] * weight) / (pixel_weight + weight);
          }
          pixel_weight += weight;
          resultMask.at<uchar>(row, col) = 255;
        }
      }
    }
  }

  static void normalizeResult(cv::Mat& result, const cv::Mat& weights) {
    cv::Mat normalized_result(result.size(), CV_8UC3);
    for(int row = 0; row < result.rows; ++row) {
      for(int col = 0; col < result.cols; ++col) {
        float weight = weights.at<float>(row, col);
        if(weight > 0) {
          cv::Vec3f pixel = result.at<cv::Vec3f>(row, col);
          normalized_result.at<cv::Vec3b>(row, col) =
              cv::Vec3b(static_cast<uchar>(pixel[0]), static_cast<uchar>(pixel[1]), static_cast<uchar>(pixel[2]));
        } else {
          normalized_result.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 0);
        }
      }
    }
    result = normalized_result;
  }
};

} // namespace Ortho

#endif // ORTHO_ALGO_STITCH_HPP
