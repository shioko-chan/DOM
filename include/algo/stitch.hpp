#ifndef SKYMERGE_ALGO_STITCH_HPP
#define SKYMERGE_ALGO_STITCH_HPP

#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>

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

  static auto stitch(ImgsData& imgs_data, DSM& dsm, Progress& progress, double target_resolution = 0.05) noexcept
      -> cv::Mat {
    if(imgs_data.empty() || dsm.empty()) {
      THIS_LOG_ERROR("empty imgs_data or dsm");
      return {};
    }
    if(target_resolution > dsm.resolution()) {
      dsm.downsample(target_resolution);
    }
    THIS_MESSAGE("start stitching");
    std::vector<std::vector<PixelSrc>> img_pixel_map(imgs_data.size());
    std::mutex                         mtx;
    auto knn = KNN<double>(8, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                return data.get_coord();
                              }) | std::views::common);
    progress.reset(dsm.size());
    run(
        dsm.size(),
        [&dsm, &imgs_data, &img_pixel_map, &mtx, &knn](int idx) noexcept {
          auto world_pt_ = dsm[idx];
          if(std::isnan(world_pt_.z)) {
            return;
          }
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
          if(best_pixel.img_idx != -1) {
            std::lock_guard<std::mutex> lock(mtx);
            img_pixel_map[best_pixel.img_idx].emplace_back(best_pixel.pixel, idx);
          }
        },
        progress);
    cv::Mat texture(
        static_cast<int>(dsm.rows() * dsm.resolution() / target_resolution),
        static_cast<int>(dsm.cols() * dsm.resolution() / target_resolution),
        CV_8UC3,
        cv::Scalar(0, 0, 0));
    run(
        img_pixel_map.size(),
        [&dsm, &img_pixel_map, &imgs_data, &texture, resolution_ratio = dsm.resolution() / target_resolution](
            int idx) noexcept {
          auto&   img_data = imgs_data[idx];
          auto&&  pixels   = img_pixel_map[idx];
          cv::Mat img      = img_data.origin_img().get();
          for(const auto& [pixel, dsm_idx] : pixels) {
            int     t_x     = dsm_idx % dsm.cols();
            int     t_y     = dsm_idx / dsm.cols();
            int     start_x = static_cast<int>(t_x * resolution_ratio);
            int     start_y = static_cast<int>(t_y * resolution_ratio);
            int     end_x   = static_cast<int>((t_x + 1) * resolution_ratio);
            int     end_y   = static_cast<int>((t_y + 1) * resolution_ratio);
            cv::Mat roi;
            cv::resize(
                img(cv::Rect(pixel.x, pixel.y, 1, 1)),
                roi,
                cv::Size(end_x - start_x, end_y - start_y),
                0,
                0,
                cv::INTER_LINEAR);
            roi.copyTo(texture(cv::Rect(start_x, start_y, end_x - start_x, end_y - start_y)));
          }
        },
        progress);
    cv::transpose(texture, texture);
    cv::flip(texture, texture, 0);
    cv::imshow("texture", texture);
    cv::waitKey(0);
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
