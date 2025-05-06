#ifndef ORTHO_ALGO_STITCH_HPP
#define ORTHO_ALGO_STITCH_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ds/dsm.hpp"
#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"

namespace Ortho {

class DSMStitcher {
public:

  static auto stitch(ImgsData& imgs_data, const DSM& dsm, Progress& progress) noexcept -> cv::Mat {
    if(imgs_data.empty() || dsm.empty()) {
      THIS_MESSAGE("empty imgs_data or dsm");
      return {};
    }

    cv::Mat result  = cv::Mat::zeros(dsm.rows(), dsm.cols(), CV_32FC3);
    cv::Mat weights = cv::Mat::zeros(dsm.rows(), dsm.cols(), CV_32F);

    THIS_MESSAGE("start to back project each image...");
    progress.reset(static_cast<int>(imgs_data.size() + 1));

    for(auto& img_data : imgs_data) {
      project_image(img_data, dsm, result, weights);
      progress.update();
    }

    cv::Mat output;
    normalize_result(result, weights, output);
    progress.update();
    THIS_MESSAGE("stitching done");

    return output;
  }

private:

  static void project_image(ImgData& img_data, const DSM& dsm, cv::Mat& result, cv::Mat& weights) noexcept {
    cv::Mat img = img_data.origin_img().get();

    cv::Mat map_x      = cv::Mat::zeros(dsm.size(), CV_32F);
    cv::Mat map_y      = cv::Mat::zeros(dsm.size(), CV_32F);
    cv::Mat valid_mask = cv::Mat::zeros(dsm.size(), CV_8U);
    for(int y = 0; y < dsm.rows(); y++) {
      for(int x = 0; x < dsm.cols(); x++) {
        float height = dsm.get_height(y, x);
        if(std::isnan(height)) {
          continue;
        }
        cv::Point3f world_pt = dsm.grid_to_world_3d(y, x);

        cv::Mat world_mat = (cv::Mat_<double>(3, 1) << world_pt.x, world_pt.y, world_pt.z);

        cv::Mat camera_pt = img_data.R_w2c() * (world_mat - img_data.t_c2w());

        if(camera_pt.at<double>(2, 0) <= 0) {
          continue;
        }

        cv::Mat img_pt = img_data.M() * camera_pt;
        float   u      = static_cast<float>(img_pt.at<double>(0, 0) / img_pt.at<double>(2, 0));
        float   v      = static_cast<float>(img_pt.at<double>(1, 0) / img_pt.at<double>(2, 0));

        if(u < 0 || v < 0 || u >= img.cols || v >= img.rows) {
          continue;
        }

        map_x.at<float>(y, x)      = u;
        map_y.at<float>(y, x)      = v;
        valid_mask.at<uchar>(y, x) = 255;
      }
    }

    cv::Mat remapped;
    cv::remap(img, remapped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

    for(int y = 0; y < dsm.rows(); y++) {
      for(int x = 0; x < dsm.cols(); x++) {
        if(valid_mask.at<uchar>(y, x) > 0) {
          cv::Vec3b  color        = remapped.at<cv::Vec3b>(y, x);
          cv::Vec3f& result_pixel = result.at<cv::Vec3f>(y, x);
          float&     weight       = weights.at<float>(y, x);

          float current_weight = 1.0f;
          float total_weight   = weight + current_weight;

          if(total_weight > 0) {
            for(int c = 0; c < 3; c++) {
              result_pixel[c] = (result_pixel[c] * weight + color[c] * current_weight) / total_weight;
            }
            weight = total_weight;
          }
        }
      }
    }
  }

  static void normalize_result(const cv::Mat& result, const cv::Mat& weights, cv::Mat& output) noexcept {
    output = cv::Mat::zeros(result.size(), CV_8UC3);

    for(int y = 0; y < result.rows; y++) {
      for(int x = 0; x < result.cols; x++) {
        if(weights.at<float>(y, x) > 0) {
          cv::Vec3f pixel            = result.at<cv::Vec3f>(y, x);
          output.at<cv::Vec3b>(y, x) = cv::Vec3b(
              static_cast<uchar>(std::min(255.0f, std::max(0.0f, pixel[0]))),
              static_cast<uchar>(std::min(255.0f, std::max(0.0f, pixel[1]))),
              static_cast<uchar>(std::min(255.0f, std::max(0.0f, pixel[2]))));
        }
      }
    }
  }
};

} // namespace Ortho

#endif // ORTHO_ALGO_STITCH_HPP
