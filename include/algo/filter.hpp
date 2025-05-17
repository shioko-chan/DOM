#ifndef SKYMERGE_FILTER_HPP
#define SKYMERGE_FILTER_HPP

#include <thread>
#include <unordered_set>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>

#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

namespace SkyMerge {
class Filter {
public:

  static void filter_outliers_statistical(
      TrackPointVec* track_point_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
      const std::filesystem::path& pcd_output_path,
#endif
      int    mean_k      = 100,
      double std_dev_mul = 1.0) {
    if(track_point_vec->empty()) {
      THIS_LOG_WARN("[Filter] Empty input data, cannot filter outliers");
      return;
    }
    PointCloudPtr cloud = track_point_vec2point_cloud(*track_point_vec);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor{true};
    sor.setInputCloud(cloud);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_dev_mul);

    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    sor.filter(*filtered_cloud);

    const auto              indices_ptr = sor.getRemovedIndices();
    std::unordered_set<int> indices_to_remove(indices_ptr->begin(), indices_ptr->end());

    filter_by_idx(track_point_vec, indices_to_remove);

#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO(
        "[Filter] Statistical filtering: {} points removed from {} points", indices_to_remove.size(), cloud->size());
    export_pcd(pcd_output_path, filtered_cloud);
#endif
  }

  static void filter_outliers_radius(
      TrackPointVec* track_point_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
      const std::filesystem::path& pcd_output_path,
#endif
      double radius        = 5.0,
      int    min_neighbors = 2) {
    if(track_point_vec->empty()) {
      THIS_LOG_WARN("[Filter] Empty input data, cannot filter outliers");
      return;
    }

    PointCloudPtr cloud = track_point_vec2point_cloud(*track_point_vec);

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror{true};
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(min_neighbors);

    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ror.filter(*filtered_cloud);

    const auto              indices_ptr = ror.getRemovedIndices();
    std::unordered_set<int> indices_to_remove(indices_ptr->begin(), indices_ptr->end());

    filter_by_idx(track_point_vec, indices_to_remove);

#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO("[Filter] Radius filtering: {} points removed from {} points", indices_to_remove.size(), cloud->size());
    export_pcd(pcd_output_path, filtered_cloud);
#endif
  }

  static void smooth_surface(
      TrackPointVec* track_point_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
      const std::filesystem::path& pcd_output_path,
#endif
      int polynomial_order = 2) noexcept {
    if(track_point_vec->empty()) {
      THIS_LOG_WARN("[Filter] Empty input data, cannot smooth surface");
      return;
    }
    auto cloud    = track_point_vec2point_cloud(*track_point_vec);
    auto smoothed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto kdtree   = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    kdtree->setInputCloud(cloud);

    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    mls.setComputeNormals(false);
    mls.setInputCloud(cloud);
    mls.setNumberOfThreads(std::thread::hardware_concurrency());
    mls.setPolynomialOrder(polynomial_order);
    mls.setSearchRadius(10.0 * compute_average_spacing(cloud));
    mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::NONE);
    mls.setSearchMethod(kdtree);

    mls.process(*smoothed);

    auto corresponding_indices = mls.getCorrespondingIndices();
    THIS_ASSERTION_SHOULD_FALSE(!corresponding_indices || corresponding_indices->indices.empty());
    std::unordered_set<size_t> keep_indices;
    for(size_t i = 0; i < smoothed->size(); ++i) {
      int original_idx = corresponding_indices->indices[i];
      if(original_idx >= 0 && original_idx < track_point_vec->size()) {
        auto& point_origin   = (*track_point_vec)[original_idx].pnt3d;
        auto  point_smoothed = (*smoothed)[i].getVector3fMap();
        point_origin[0]      = static_cast<double>(point_smoothed.x());
        point_origin[1]      = static_cast<double>(point_smoothed.y());
        point_origin[2]      = static_cast<double>(point_smoothed.z());
        keep_indices.insert(original_idx);
      }
    }
    keep_by_idx(track_point_vec, keep_indices);

#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO("[Filter] Surface smoothing: {} points processed from {} points", smoothed->size(), cloud->size());
    export_pcd(pcd_output_path, smoothed);
#endif
  }

  static void
  filter_reprojection_error(TrackPointVec* track_point_vec, ImgsData& imgs_data, double threshold = 4.0) noexcept {
    if(track_point_vec->empty()) {
      THIS_LOG_WARN("[BA] Empty input data, cannot filter outliers");
      return;
    }
    for(auto& [pnt3d, pnt2d_idx_vec] : *track_point_vec) {
      std::erase_if(pnt2d_idx_vec, [&imgs_data, &pnt3d, threshold](const auto& pnt2d_idx) noexcept {
        const auto& img_data     = imgs_data[pnt2d_idx.img_idx];
        const auto& [ob_x, ob_y] = img_data.get_kpnts()[pnt2d_idx.pnt_idx];
        auto pnt                 = world2pixel(
            img_data.A_w2c_array_raw().data(),
            img_data.t_w2c_array_raw().data(),
            imgs_data.camera_array_raw().data(),
            imgs_data.distort_array_raw().data(),
            pnt3d.data());
        return std::hypot(pnt[0] - ob_x, pnt[1] - ob_y) > threshold;
      });
    }
  }

  static void filter_near_observations(TrackPointVec* track_point_vec, ImgsData& imgs_data, double threshold = 5.0) {
    for(auto& [pnt3d, pnt2d_idx_vec] : *track_point_vec) {
      std::erase_if(pnt2d_idx_vec, [&imgs_data, &pnt3d, threshold](const auto& pnt2d_idx) noexcept {
        const auto& img_data = imgs_data[pnt2d_idx.img_idx];
        auto pnt = world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), pnt3d.data());
        return pnt[2] < threshold;
      });
    }
  }

  static void filter_track_too_few_observations(TrackPointVec* track_point_vec, int min_points = 2) {
#ifdef ENABLE_VISUALIZE_OUTPUT
    int cnt = track_point_vec->size();
#endif
    std::erase_if(*track_point_vec, [min_points](const auto& tri_res) noexcept {
      return tri_res.pnt2d_idx_vec.size() < min_points;
    });
#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO(
        "[Filter] Too few observations filtering: {} points before, {} points after", cnt, track_point_vec->size());
#endif
  }

  static void filter_track_too_few_observations(Tracks* tracks, int min_points = 2) {
#ifdef ENABLE_VISUALIZE_OUTPUT
    int cnt = tracks->size();
#endif
    std::erase_if(*tracks, [min_points](const auto& tri_res) noexcept { return tri_res.size() < min_points; });
#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO("[Filter] Too few observations filtering: {} points before, {} points after", cnt, tracks->size());
#endif
  }

  static void filter_invalid_image(const TrackPointVec& track_point_vec, ImgsData& imgs_data) {
    std::unordered_set<int> img_id;
    for(const auto& tri_res : track_point_vec) {
      for(const auto& [idx, _] : tri_res.pnt2d_idx_vec) {
        img_id.insert(idx);
      }
    }
    for(int idx = 0; idx < imgs_data.size(); ++idx) {
      if(!img_id.contains(idx)) {
        imgs_data[idx].set_invalid();
      }
    }
  }

  static auto grid_downsample_2d(const PointCloudPtr& point_cloud, float distance_threshold = 0.5) noexcept
      -> PointCloudPtr {
    pcl::PointXYZ min_pnt;
    pcl::PointXYZ max_pnt;
    pcl::getMinMax3D(*point_cloud, min_pnt, max_pnt);
    float start_x        = min_pnt.getVector3fMap().x();
    float end_x          = max_pnt.getVector3fMap().x();
    float start_y        = min_pnt.getVector3fMap().y();
    float end_y          = max_pnt.getVector3fMap().y();
    auto  point_cloud_2d = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
    point_cloud_2d->reserve(point_cloud->size());
    for(const auto& point : *point_cloud) {
      auto point_ = point.getVector3fMap();
      point_cloud_2d->emplace_back(point_.x(), point_.y());
    }
    pcl::KdTreeFLANN<pcl::PointXY> kd_tree;
    kd_tree.setInputCloud(point_cloud_2d);
    const int x_steps    = static_cast<int>((end_x - start_x) / distance_threshold);
    const int y_steps    = static_cast<int>((end_y - start_y) / distance_threshold);
    auto      point_grid = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(int xi = 0; xi < x_steps; ++xi) {
      float x_pos = start_x + (static_cast<float>(xi) * distance_threshold);
      for(int yi = 0; yi < y_steps; ++yi) {
        float              y_pos = start_y + (static_cast<float>(yi) * distance_threshold);
        pcl::PointXY       search_point(x_pos, y_pos);
        std::vector<int>   indices;
        std::vector<float> distances;
        kd_tree.radiusSearch(search_point, distance_threshold, indices, distances);
        if(!indices.empty()) {
          auto point =
              point_cloud->points[*std::ranges::max_element(indices, [&point_cloud](int idx0, int idx1) noexcept {
                return point_cloud->points[idx0].getVector3fMap().z() < point_cloud->points[idx1].getVector3fMap().z();
              })];
          point_grid->emplace_back(x_pos, y_pos, point.getVector3fMap().z());
        }
      }
    }
    return point_grid;
  }
};
} // namespace SkyMerge

#endif // SKYMERGE_FILTER_HPP
