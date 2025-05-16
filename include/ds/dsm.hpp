#ifndef SKYMERGE_DSM_HPP
#define SKYMERGE_DSM_HPP

#include <pcl/common/common.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <cmath>
#include <cstddef>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/report_error.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

namespace SkyMerge {

class DSM {
public:

  DSM() noexcept = default;

  explicit DSM(const PointCloudPtr& cloud, Progress& progress, double resolution = 0.5) noexcept :
      resolution_(resolution) {
    if(cloud->empty()) {
      THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
      return;
    }
    pcl::PointXYZ min_pt_;
    pcl::PointXYZ max_pt_;
    pcl::getMinMax3D(*cloud, min_pt_, max_pt_);
    auto min_pt = min_pt_.getVector3fMap();
    auto max_pt = max_pt_.getVector3fMap();
    min_x       = min_pt.x();
    min_y       = min_pt.y();
    height_map  = calculate_dsm(cloud, min_x, min_y, max_pt.x(), max_pt.y(), progress, resolution);
  }

  [[nodiscard]] auto operator[](int idx) const noexcept -> Point3<double> {
    if(idx < 0 || idx >= height_map.rows * height_map.cols) {
      report_error("DSM Index out of range");
    }
    int row = idx / height_map.cols;
    int col = idx % height_map.cols;
    return {min_x + (col * resolution_), min_y + (row * resolution_), height_map.at<double>(row, col)};
  }

  [[nodiscard]] auto empty() const noexcept -> bool { return height_map.empty(); }

  [[nodiscard]] auto size() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map.rows * height_map.cols;
  }

  [[nodiscard]] auto cols() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map.cols;
  }

  [[nodiscard]] auto rows() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map.rows;
  }

  [[nodiscard]] auto resolution() const noexcept -> double { return resolution_; }

  void downsample(double target_resolution) noexcept {
    if(resolution_ >= target_resolution) {
      return;
    }
    auto scale = resolution_ / target_resolution;
    cv::resize(height_map, height_map, {}, scale, scale, cv::INTER_NEAREST);
    resolution_ = target_resolution;
  }

private:

  [[nodiscard]] static auto calculate_dsm(
      const PointCloudPtr& cloud,
      double               min_x,
      double               min_y,
      double               max_x,
      double               max_y,
      Progress&            progress,
      double               resolution = 0.5,
      int                  k_nn       = 16) noexcept -> cv::Mat {
    int cols = static_cast<int>(std::ceil((max_x - min_x) / resolution)) + 1;
    int rows = static_cast<int>(std::ceil((max_y - min_y) / resolution)) + 1;
    THIS_MESSAGE("DSM size: {}x{}, resolution: {}m", cols, rows, resolution);
    PointCloudPtr cloud_xy = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(const auto& point : *cloud) {
      auto pnt = point.getVector3fMap();
      cloud_xy->points.emplace_back(pnt.x(), pnt.y(), 0.0F);
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud_xy);
    cv::Mat      dsm(rows, cols, CV_64F, std::numeric_limits<double>::quiet_NaN());
    const double max_distance = compute_average_spacing(cloud_xy) * 2.0;
    run((static_cast<size_t>(rows * cols)),
        [rows, cols, min_x, min_y, resolution, k_nn, max_distance, &dsm, &kdtree, &cloud](int cnt) noexcept {
          int                row   = cnt / cols;
          int                col   = cnt % cols;
          double             cur_x = min_x + (col * resolution);
          double             cur_y = min_y + (row * resolution);
          pcl::PointXYZ      search_point{static_cast<float>(cur_x), static_cast<float>(cur_y), 0.0F};
          std::vector<int>   point_idx(k_nn);
          std::vector<float> point_dist(k_nn);
          if(kdtree.radiusSearch(search_point, max_distance, point_idx, point_dist, k_nn) > 0) {
            double sum_weights = 0.0;
            double sum_values  = 0.0;
            auto&  dsm_val     = dsm.at<double>(row, col);
            for(auto&& [idx, dist] : std::views::zip(point_idx, point_dist)) {
              if(dist < 1e-6) {
                dsm_val = cloud->points[idx].getVector3fMap().z();
                break;
              }
              double weight = 1.0 / std::pow(dist, 2);
              sum_weights += weight;
              sum_values += weight * cloud->points[idx].getVector3fMap().z();
            }
            if(std::isnan(dsm_val)) {
              dsm_val = sum_values / sum_weights;
            }
          }
        },
        progress);
    return dsm;
  }

  cv::Mat height_map;
  double  resolution_ = 0.0;
  double  min_x       = 0.0;
  double  min_y       = 0.0;
};

} // namespace SkyMerge

#endif // SKYMERGE_DSM_HPP