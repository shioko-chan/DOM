#ifndef ORTHO_FILTER_HPP
#define ORTHO_FILTER_HPP

#include <vector>
#include <utility>
#include <algorithm>
#include <unordered_set>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include "types/common_types.hpp"

namespace Ortho {
void
filter_outliers(
    TriReses* tri_reses,
    int mean_k = 50,
    double std_dev_mul = 1.0) {
    if (tri_reses->empty()) {
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(tri_reses->size());
    for(const auto& [point, _] : *tri_reses) {
        (*cloud)[i].x = static_cast<float>(point[0]);
        (*cloud)[i].y = static_cast<float>(point[1]);
        (*cloud)[i].z = static_cast<float>(point[2]);
    }
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    sor.setInputCloud(cloud);
    sor.setMeanK(mean_k);            
    sor.setStddevMulThresh(std_dev_mul); 
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    sor.filter(*filtered_cloud);
    std::vector<int> removed_indices;
    sor.getRemovedIndices(removed_indices);
    std::unordered_set<int> indices_to_remove(removed_indices.begin(), removed_indices.end());
    tri_reses->erase(std::remove_if(tri_reses->begin(), tri_reses->end(),
        [&indices_to_remove, idx = 0](const auto&) mutable noexcept {
            return indices_to_remove.find(idx++) != indices_to_remove.end();
        }), tri_reses->end());
}

void smooth_surface(
    TriReses* tri_reses,
    double search_radius = 0.03,
    bool polynomial_fit = true,
    int polynomial_order = 2,
    bool compute_normals = false) {
    if (tri_reses->empty()) {
        return;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->resize(tri_reses->size());
    for(const auto& [point, _] : *tri_reses) {
      (*cloud)[i].x = static_cast<float>(point[0]);
      (*cloud)[i].y = static_cast<float>(point[1]);
      (*cloud)[i].z = static_cast<float>(point[2]);
    }
    pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
    mls.setInputCloud(cloud);                    
    mls.setSearchRadius(search_radius);            
    mls.setPolynomialFit(polynomial_fit);          
    mls.setPolynomialOrder(polynomial_order);       
    mls.setComputeNormals(compute_normals);        
    pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed(new pcl::PointCloud<pcl::PointXYZ>);
    mls.process(*smoothed);
    for (size_t i = 0; i < tri_reses->size() && i < smoothed->size(); i++) {
        auto& point = (*tri_reses)[i].pnt3d;
        point[0] = static_cast<double>((*smoothed)[i].x);
        point[1] = static_cast<double>((*smoothed)[i].y);
        point[2] = static_cast<double>((*smoothed)[i].z);
    }
}
} // namespace Ortho

#endif // ORTHO_FILTER_HPP
