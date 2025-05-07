#ifndef ORTHO_ALGO_STITCH_HPP
#define ORTHO_ALGO_STITCH_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ds/dsm.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace Ortho {

static auto stitch(ImgsData& imgs_data, DSM& dsm, Progress& progress) noexcept -> cv::Mat {
  if(imgs_data.empty() || dsm.empty()) {
    THIS_LOG_ERROR("empty imgs_data or dsm");
    return {};
  }
  THIS_MESSAGE("start stitching");
  cv::Mat texture(dsm.rows(), dsm.cols(), CV_8UC3, cv::Scalar(0, 0, 0));
  progress.reset(dsm.size());
  run(
      dsm.size(),
      [&dsm, &imgs_data, &texture](int idx) noexcept {
        auto dsm_unit = dsm[idx];
        std::cout << dsm_unit.point.x << " " << dsm_unit.point.y << " " << dsm_unit.point.z << std::endl;
        double                best_cos_angle = 0.0;
        ImgData*              best_img_ptr   = nullptr;
        std::pair<int, int>   best_pixel{-1, -1};
        std::array<double, 3> world_pt{dsm_unit.point.x, dsm_unit.point.y, dsm_unit.point.z};
        for(auto& img_data : imgs_data) {
          if(!img_data.is_valid()) {
            return;
          }
          cv::Mat camera_center = img_data.t_c2w();
          cv::Mat view_vec      = camera_center - dsm_unit.point;
          THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
          cv::normalize(view_vec, view_vec);
          double cos_angle = std::abs(view_vec.dot(dsm_unit.normal));
          if(cos_angle <= best_cos_angle || cos_angle < 0.5) {
            continue;
          }
          const auto& [width, height] = img_data.origin_img().get_size();
          auto point =
              world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), world_pt.data());
          auto pixel =
              camera2pixel(imgs_data.camera_array_raw().data(), imgs_data.distort_array_raw().data(), point.data());
          int img_x = static_cast<int>(std::round(pixel.x()));
          int img_y = static_cast<int>(std::round(pixel.y()));
          if(img_x < 0 || img_y < 0 || img_x >= width || img_y >= height) {
            return;
          }
          best_cos_angle = cos_angle;
          best_img_ptr   = &img_data;
          best_pixel     = {img_x, img_y};
        }
        if(best_img_ptr != nullptr) {
          cv::Mat   img                   = best_img_ptr->origin_img().get();
          cv::Vec3b best_color            = img.at<cv::Vec3b>(best_pixel.second, best_pixel.first);
          int       row                   = idx / dsm.cols();
          int       col                   = idx % dsm.cols();
          texture.at<cv::Vec3b>(row, col) = best_color;
        }
      },
      progress);
  return texture;
}

} // namespace Ortho

#endif // ORTHO_ALGO_STITCH_HPP
