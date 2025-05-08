#ifndef SKYMERGE_ALGO_STITCH_HPP
#define SKYMERGE_ALGO_STITCH_HPP

#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <unordered_map>

#include "algo/knn.hpp"
#include "ds/dsm.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace SkyMerge {

class DSMStitcher {
public:

  static auto stitch(ImgsData& imgs_data, DSM& dsm, Progress& progress) noexcept -> cv::Mat {
    if(imgs_data.empty() || dsm.empty()) {
      THIS_LOG_ERROR("empty imgs_data or dsm");
      return {};
    }
    THIS_MESSAGE("start stitching");
    std::unordered_map<int, std::vector<PixelSrc>> idx_map;
    std::mutex                                     mtx;
    auto knn = KNN<double>(8, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                return data.get_coord();
                              }) | std::views::common);
    progress.reset(dsm.size());
    run(
        dsm.size(),
        [&dsm, &imgs_data, &idx_map, &mtx, &knn](int idx) noexcept {
          auto                  world_pt_ = dsm[idx];
          std::array<double, 3> world_pt{world_pt_.x, world_pt_.y, world_pt_.z};
          BestPixel             best_pixel{.img_idx = -1, .pixel = {-1, -1}, .cos_angle = 0.0};
          cv::Mat               normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
          for(int idx : knn.find_nearest_neighbour(Point<double>{world_pt_.x, world_pt_.y})) {
            auto& img_data = imgs_data[idx];
            if(!img_data.is_valid()) {
              continue;
            }
            cv::Mat view_vec = img_data.t_c2w() - world_pt_;
            THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
            cv::normalize(view_vec, view_vec);
            double cos_angle = std::abs(view_vec.dot(normal));
            if(cos_angle <= best_pixel.cos_angle) {
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
              continue;
            }
            best_pixel.cos_angle = cos_angle;
            best_pixel.img_idx   = idx;
            best_pixel.pixel     = {img_x, img_y};
          }
          std::lock_guard<std::mutex> lock(mtx);
          if(best_pixel.img_idx != -1) {
            idx_map[best_pixel.img_idx].emplace_back(best_pixel.pixel, idx);
          }
        },
        progress);
    cv::Mat texture(dsm.rows(), dsm.cols(), CV_8UC3, cv::Scalar(0, 0, 0));
    for(const auto& [img_idx, pixels] : idx_map) {
      auto&   img_data = imgs_data[img_idx];
      cv::Mat img      = img_data.origin_img().get();
      for(const auto& [pixel, dsm_idx] : pixels) {
        int t_x                         = dsm_idx % dsm.cols();
        int t_y                         = dsm_idx / dsm.cols();
        texture.at<cv::Vec3b>(t_y, t_x) = img.at<cv::Vec3b>(pixel.y, pixel.x);
      }
    }
    cv::flip(texture, texture, -1);
    return texture;
  }

private:

  struct alignas(32) BestPixel {
    int        img_idx;
    Point<int> pixel;
    double     cos_angle;
  };

  struct alignas(16) PixelSrc {
    Point<int> pixel;
    int        dsm_idx;
  };
};
} // namespace SkyMerge

#endif // SKYMERGE_ALGO_STITCH_HPP
