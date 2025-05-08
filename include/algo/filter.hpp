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
#include "types/common_types.hpp"

namespace SkyMerge {
class Filter {
public:

  static void filter_outliers_statistical(
      TriResVec* tri_res_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
      const fs::path& pcd_output_path,
#endif
      int    mean_k      = 100,
      double std_dev_mul = 1.0) {
    if(tri_res_vec->empty()) {
      THIS_LOG_WARN("[Filter] Empty input data, cannot filter outliers");
      return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = tri_res_vec2point_cloud(*tri_res_vec);

    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor{true};
    sor.setInputCloud(cloud);
    sor.setMeanK(mean_k);
    sor.setStddevMulThresh(std_dev_mul);

    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    sor.filter(*filtered_cloud);

    const auto              indices_ptr = sor.getRemovedIndices();
    std::unordered_set<int> indices_to_remove(indices_ptr->begin(), indices_ptr->end());

    filter_by_idx(tri_res_vec, indices_to_remove);

#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO(
        "[Filter] Statistical filtering: {} points removed from {} points", indices_to_remove.size(), cloud->size());
    export_pcd(pcd_output_path, filtered_cloud);
#endif
  }

  static void filter_outliers_radius(
      TriResVec* tri_res_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
      const fs::path& pcd_output_path,
#endif
      double radius        = 5.0,
      int    min_neighbors = 2) {
    if(tri_res_vec->empty()) {
      THIS_LOG_WARN("[Filter] Empty input data, cannot filter outliers");
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = tri_res_vec2point_cloud(*tri_res_vec);

    pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror{true};
    ror.setInputCloud(cloud);
    ror.setRadiusSearch(radius);
    ror.setMinNeighborsInRadius(min_neighbors);

    auto filtered_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    ror.filter(*filtered_cloud);

    const auto              indices_ptr = ror.getRemovedIndices();
    std::unordered_set<int> indices_to_remove(indices_ptr->begin(), indices_ptr->end());

    filter_by_idx(tri_res_vec, indices_to_remove);

#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO("[Filter] Radius filtering: {} points removed from {} points", indices_to_remove.size(), cloud->size());
    export_pcd(pcd_output_path, filtered_cloud);
#endif
  }

  static void smooth_surface(
      TriResVec* tri_res_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
      const fs::path& pcd_output_path,
#endif
      int polynomial_order = 2) {
    if(tri_res_vec->empty()) {
      THIS_LOG_WARN("[Filter] Empty input data, cannot smooth surface");
      return;
    }
    auto cloud    = tri_res_vec2point_cloud(*tri_res_vec);
    auto smoothed = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    auto kdtree   = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    kdtree->setInputCloud(cloud);

    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    mls.setComputeNormals(false);
    mls.setInputCloud(cloud);
    mls.setNumberOfThreads(std::thread::hardware_concurrency());
    mls.setPolynomialOrder(polynomial_order);
    mls.setSearchRadius(5.0 * compute_average_spacing(cloud));
    mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::NONE);
    mls.setSearchMethod(kdtree);

    mls.process(*smoothed);

    auto corresponding_indices = mls.getCorrespondingIndices();
    THIS_ASSERTION_SHOULD_FALSE(!corresponding_indices || corresponding_indices->indices.empty());
    std::unordered_set<size_t> keep_indices;
    for(size_t i = 0; i < smoothed->size(); ++i) {
      int original_idx = corresponding_indices->indices[i];
      if(original_idx >= 0 && original_idx < tri_res_vec->size()) {
        auto& point_origin   = (*tri_res_vec)[original_idx].pnt3d;
        auto  point_smoothed = (*smoothed)[i].getVector3fMap();
        point_origin[0]      = static_cast<double>(point_smoothed.x());
        point_origin[1]      = static_cast<double>(point_smoothed.y());
        point_origin[2]      = static_cast<double>(point_smoothed.z());
        keep_indices.insert(original_idx);
      }
    }
    keep_by_idx(tri_res_vec, keep_indices);

#ifdef ENABLE_VISUALIZE_OUTPUT
    THIS_LOG_INFO("[Filter] Surface smoothing: {} points processed from {} points", smoothed->size(), cloud->size());
    export_pcd(pcd_output_path, smoothed);
#endif
  }

  static void filter_too_few_points(TriResVec* tri_res_vec, int min_points = 2) {
    std::erase_if(*tri_res_vec, [min_points](const auto& tri_res) noexcept {
      return tri_res.pnt2d_idx_vec.size() < min_points;
    });
  }

  static void filter_near_observes(ImgsData& imgs_data, TriResVec* tri_res_vec, double threshold = 5.0) {
    for(auto& [pnt3d, pnt2d_idx_vec] : *tri_res_vec) {
      std::erase_if(pnt2d_idx_vec, [&imgs_data, &pnt3d, threshold](const auto& pnt2d_idx) noexcept {
        const auto& img_data = imgs_data[pnt2d_idx.img_idx];
        auto pnt = world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), pnt3d.data());
        return pnt[2] < threshold;
      });
    }
  }

  static void filter_invalid_image(const TriResVec& tri_res_vec, ImgsData& imgs_data) {
    std::unordered_set<int> img_id;
    for(const auto& tri_res : tri_res_vec) {
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
};
} // namespace SkyMerge

#endif // SKYMERGE_FILTER_HPP
