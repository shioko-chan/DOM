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
    struct PatchSrc {
      int img_idx;
      int dsm_idx;
      std::array<cv::Point2f, 4> img_corners;
      int start_x, start_y, end_x, end_y;
    };
    std::vector<PatchSrc> patch_src_map;
    auto knn = KNN<double>(8, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                return data.get_coord();
                              }) | std::views::common);
    progress.reset(dsm.size());
    run(
        dsm.size(),
        [&dsm, &imgs_data, &patch_src_map, &mtx, &knn, target_resolution](int idx) noexcept {
          auto world_pt_ = dsm[idx];
          if(std::isnan(world_pt_.z)) {
            return;
          }
          std::array<double, 3> world_pt{world_pt_.x, world_pt_.y, world_pt_.z};
          BestPixel best_pixel{.img_idx = -1, .pixel = {-1, -1}, .cos_angle = 0.0};
          cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
          int best_img_idx = -1;
          double best_cos_angle = 0.0;
          std::array<cv::Point2f, 4> best_img_corners;
          for(int img_idx : knn.find_nearest_neighbour(Point<double>{world_pt_.x, world_pt_.y})) {
            auto& img_data = imgs_data[img_idx];
            if(!img_data.is_valid()) continue;
            cv::Mat view_vec = img_data.t_c2w() - world_pt_;
            THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
            cv::normalize(view_vec, view_vec);
            double cos_angle = std::abs(view_vec.dot(normal));
            if(cos_angle <= best_cos_angle) continue;
            // 计算DSM区块四角点的世界坐标
            int t_x = idx % dsm.cols();
            int t_y = idx / dsm.cols();
            double res = dsm.resolution();
            double z = world_pt_.z;
            double cx = world_pt_.x;
            double cy = world_pt_.y;
            std::array<std::array<double, 3>, 4> world_corners = {
              std::array<double, 3>{cx - res/2, cy - res/2, z}, // 左上
              std::array<double, 3>{cx + res/2, cy - res/2, z}, // 右上
              std::array<double, 3>{cx + res/2, cy + res/2, z}, // 右下
              std::array<double, 3>{cx - res/2, cy + res/2, z}  // 左下
            };
            std::array<cv::Point2f, 4> img_corners;
            bool valid = true;
            for (int i = 0; i < 4; ++i) {
              auto cam_pt = world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), world_corners[i].data());
              auto px = camera2pixel(imgs_data.camera_array_raw().data(), imgs_data.distort_array_raw().data(), cam_pt.data());
              img_corners[i] = cv::Point2f(static_cast<float>(px.x()), static_cast<float>(px.y()));
              // 可选：检查是否在图像范围内
              const auto& [width, height] = img_data.origin_img().get_size();
              if(img_corners[i].x < 0 || img_corners[i].y < 0 || img_corners[i].x >= width || img_corners[i].y >= height) {
                valid = false;
              }
            }
            if (!valid) continue;
            best_cos_angle = cos_angle;
            best_img_idx = img_idx;
            best_img_corners = img_corners;
          }
          if(best_img_idx != -1) {
            int t_x = idx % dsm.cols();
            int t_y = idx / dsm.cols();
            double resolution_ratio = dsm.resolution() / target_resolution;
            int start_x = static_cast<int>(t_x * resolution_ratio);
            int start_y = static_cast<int>(t_y * resolution_ratio);
            int end_x   = static_cast<int>((t_x + 1) * resolution_ratio);
            int end_y   = static_cast<int>((t_y + 1) * resolution_ratio);
            std::lock_guard<std::mutex> lock(mtx);
            patch_src_map.push_back(PatchSrc{best_img_idx, idx, best_img_corners, start_x, start_y, end_x, end_y});
          }
        },
        progress);
    cv::Mat texture(
        static_cast<int>(dsm.rows() * dsm.resolution() / target_resolution),
        static_cast<int>(dsm.cols() * dsm.resolution() / target_resolution),
        CV_8UC3,
        cv::Scalar(0, 0, 0));
    run(
        patch_src_map.size(),
        [&imgs_data, &patch_src_map, &texture](int idx) noexcept {
          const auto& patch = patch_src_map[idx];
          auto& img_data = imgs_data[patch.img_idx];
          cv::Mat img = img_data.origin_img().get();
          int patch_w = patch.end_x - patch.start_x;
          int patch_h = patch.end_y - patch.start_y;
          std::vector<cv::Point2f> patch_corners = {
            cv::Point2f(0, 0),
            cv::Point2f(static_cast<float>(patch_w-1), 0),
            cv::Point2f(static_cast<float>(patch_w-1), static_cast<float>(patch_h-1)),
            cv::Point2f(0, static_cast<float>(patch_h-1))
          };
          cv::Mat H = cv::getPerspectiveTransform(patch.img_corners.data(), patch_corners.data());
          cv::Mat patch_img;
          cv::warpPerspective(img, patch_img, H, cv::Size(patch_w, patch_h), cv::INTER_LINEAR, cv::BORDER_REFLECT);
          cv::Mat roi;
          if (patch_img.size() != cv::Size(patch_w, patch_h)) {
            cv::resize(patch_img, roi, cv::Size(patch_w, patch_h), 0, 0, cv::INTER_LINEAR);
          } else {
            roi = patch_img;
          }
          roi.copyTo(texture(cv::Rect(patch.start_x, patch.start_y, patch_w, patch_h)));
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
