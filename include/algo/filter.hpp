#ifndef ORTHO_FILTER_HPP
#define ORTHO_FILTER_HPP

#include <thread>
#include <unordered_set>
#include <vector>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/common/distances.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/surface/mls.h>

#include "tools/log.hpp"
#include "tools/utility.hpp"
#include "types/common_types.hpp"

namespace Ortho {

inline auto compute_average_spacing(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int k_neighbors = 100) -> double {
  pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
  kdtree.setInputCloud(cloud);
  double total_distance = 0.0;
  for(const auto& point : *cloud) {
    std::vector<int>   indices(k_neighbors);
    std::vector<float> distances(k_neighbors);
    kdtree.nearestKSearch(point, k_neighbors, indices, distances);
    double distance = 0.0;
    for(double dist : distances) {
      distance += sqrt(dist);
    }
    total_distance += distance / k_neighbors;
  }
  return total_distance / static_cast<double>(cloud->size());
}

inline auto tri_res_vec2point_cloud(const TriResVec& tri_res_vec) -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud{new pcl::PointCloud<pcl::PointXYZ>};
  cloud->resize(tri_res_vec.size());
  for(int i = 0; i < tri_res_vec.size(); ++i) {
    const auto& point = tri_res_vec[i].pnt3d;
    (*cloud)[i].getVector3fMap() =
        Eigen::Vector3f{static_cast<float>(point[1]), static_cast<float>(point[0]), static_cast<float>(-point[2])};
  }
  return cloud;
}

inline void filter_outliers(
    TriResVec* tri_res_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
    const fs::path& pcd_output_path,
#endif
    int    mean_k      = 100,
    double std_dev_mul = 1.0) {
  if(tri_res_vec->empty()) {
    return;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = tri_res_vec2point_cloud(*tri_res_vec);

  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor{true};
  sor.setInputCloud(cloud);
  sor.setMeanK(mean_k);
  sor.setStddevMulThresh(std_dev_mul);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  sor.filter(*filtered_cloud);

  const auto              indices_ptr = sor.getRemovedIndices();
  std::unordered_set<int> indices_to_remove(indices_ptr->begin(), indices_ptr->end());

  filter_by_idx(tri_res_vec, indices_to_remove);

#ifdef ENABLE_VISUALIZE_OUTPUT
  THIS_MESSAGE("Original cloud size: {}", cloud->size());
  THIS_MESSAGE("Filtered cloud size: {}", filtered_cloud->size());
  THIS_MESSAGE("Removed indices: {}", indices_to_remove.size());
  export_pcd(pcd_output_path, filtered_cloud);
#endif
}

inline void smooth_surface(
    TriResVec* tri_res_vec,
#ifdef ENABLE_VISUALIZE_OUTPUT
    const fs::path& pcd_output_path,
#endif
    int  polynomial_order = 2,
    bool compute_normals  = false) {
  if(tri_res_vec->empty()) {
    return;
  }
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = tri_res_vec2point_cloud(*tri_res_vec);

  double avg_spacing = compute_average_spacing(cloud);

  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
  mls.setInputCloud(cloud);
  mls.setNumberOfThreads(std::thread::hardware_concurrency());

  mls.setSearchRadius(5.0 * avg_spacing);
  mls.setPolynomialOrder(polynomial_order);
  mls.setComputeNormals(compute_normals);
  mls.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::RANDOM_UNIFORM_DENSITY);
  mls.setUpsamplingRadius(0.7 * mls.getSearchRadius());
  mls.setUpsamplingStepSize(0.5 * avg_spacing);

  pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
  kdtree->setInputCloud(cloud);
  mls.setSearchMethod(kdtree);

  pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed{new pcl::PointCloud<pcl::PointXYZ>};
  mls.process(*smoothed);

  auto                       corresponding_indices = mls.getCorrespondingIndices();
  std::unordered_set<size_t> keep_indices;
  for(size_t i = 0; i < smoothed->size(); ++i) {
    int original_idx = corresponding_indices->indices[i];
    if(original_idx >= 0) {
      auto& point_origin   = (*tri_res_vec)[original_idx].pnt3d;
      auto  point_smoothed = (*smoothed)[i].getVector3fMap();
      point_origin[0]      = static_cast<double>(point_smoothed.y());
      point_origin[1]      = static_cast<double>(point_smoothed.x());
      point_origin[2]      = static_cast<double>(-point_smoothed.z());
      keep_indices.insert(original_idx);
    }
  }

  keep_by_idx(tri_res_vec, keep_indices);
#ifdef ENABLE_VISUALIZE_OUTPUT
  export_pcd(pcd_output_path, smoothed);
#endif
}
} // namespace Ortho

#endif // ORTHO_FILTER_HPP
