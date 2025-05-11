#ifndef SKYMERGE_ALGO_STITCH1_HPP
#define SKYMERGE_ALGO_STITCH1_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <pcl/common/centroid.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "algo/knn.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"
#include "types/common_types.hpp"
#include "types/cv_alias.hpp"

namespace SkyMerge {

class TriMeshStitcher {
private:

  using K             = CGAL::Exact_predicates_inexact_constructions_kernel;
  using Vb            = CGAL::Triangulation_vertex_base_with_info_2<int, K>;
  using Tds           = CGAL::Triangulation_data_structure_2<Vb>;
  using Triangulation = CGAL::Delaunay_triangulation_2<K, Tds>;

  using Triangle = std::array<int, 3>;

  using Triangles = std::vector<Triangle>;

  struct alignas(128) ImageOperation {
    cv::Mat  affine_transform;
    cv::Mat  mask;
    cv::Rect target_roi;
  };

public:

  static auto stitch(
      ImgsData&                                  imgs_data,
      const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud,
      Progress&                                  progress,
      double                                     resolution = 10.0) noexcept -> cv::Mat {
    if(imgs_data.empty()) {
      THIS_LOG_ERROR("empty imgs data");
      return {};
    }
    if(point_cloud->empty()) {
      THIS_LOG_ERROR("empty point cloud");
      return {};
    }
    auto            new_point_cloud = euclidean_cluster_xy(point_cloud, 0.4);
    Point3s<double> vertices;
    for(const auto& point : *new_point_cloud) {
      auto pnt = point.getVector3fMap();
      vertices.emplace_back(pnt.x(), pnt.y(), pnt.z());
    }
    THIS_MESSAGE("Start triangulation and stitching");
    auto point_2d_view = vertices | std::views::enumerate
                         | std::views::transform([](const auto& tuple) noexcept -> std::pair<K::Point_2, int> {
                             const auto& vertex = std::get<1>(tuple);
                             return {{vertex.x, vertex.y}, static_cast<int>(std::get<0>(tuple))};
                           });
    Triangulation triangulation;
    triangulation.insert(point_2d_view.begin(), point_2d_view.end());
    Triangles triangles;
    for(auto iter = triangulation.finite_faces_begin(); iter != triangulation.finite_faces_end(); ++iter) {
      auto triangle = *iter;
      auto v_1      = triangle.vertex(0)->info();
      auto v_2      = triangle.vertex(1)->info();
      auto v_3      = triangle.vertex(2)->info();
      triangles.push_back({v_1, v_2, v_3});
    }
    if(triangles.empty()) {
      THIS_LOG_ERROR("Delaunay triangulation failed");
      return {};
    }
    THIS_MESSAGE("triangulation completed, found " + std::to_string(triangles.size()) + " triangles");
    return render_textured_mesh(imgs_data, vertices, triangles, progress, resolution);
  }

private:

  static auto euclidean_cluster_xy(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, double eps = 0.5) noexcept
      -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
    if(point_cloud->empty()) {
      return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    }
    auto xy_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    xy_cloud->reserve(point_cloud->size());
    for(size_t i = 0; i < point_cloud->size(); ++i) {
      auto pnt = point_cloud->points[i].getVector3fMap();
      xy_cloud->points.emplace_back(pnt.x(), pnt.y(), 0.0F);
    }
    std::vector<pcl::PointIndices> cluster_indices;
    auto                           tree = std::make_shared<pcl::search::KdTree<pcl::PointXYZ>>();
    tree->setInputCloud(xy_cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> euclidean_cluster;
    euclidean_cluster.setClusterTolerance(eps);
    euclidean_cluster.setSearchMethod(tree);
    euclidean_cluster.setInputCloud(xy_cloud);
    euclidean_cluster.extract(cluster_indices);
    THIS_MESSAGE("聚类完成，找到 " + std::to_string(cluster_indices.size()) + " 个簇");
    auto result_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    for(const auto& indices : cluster_indices) {
      if(indices.indices.empty()) {
        continue;
      }
      auto cluster_cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
      for(const auto& idx : indices.indices) {
        cluster_cloud->push_back(point_cloud->points[idx]);
      }
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*cluster_cloud, centroid);
      float z_min = std::numeric_limits<float>::max();
      for(const auto& point : cluster_cloud->points) {
        z_min = std::min(z_min, point.getArray3fMap().z());
      }
      result_cloud->emplace_back(centroid[0], centroid[1], z_min);
    }
    return result_cloud;
  }

  static auto project_point(ImgData& img_data, ImgsData& imgs_data, const Point3<double>& world_pt_) noexcept
      -> Point<double> {
    std::array<double, 3> world_pt{world_pt_.x, world_pt_.y, world_pt_.z};
    auto cam_pt = world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), world_pt.data());
    auto p_x = camera2pixel(imgs_data.camera_array_raw().data(), imgs_data.distort_array_raw().data(), cam_pt.data());
    return {p_x.x(), p_x.y()};
  }

  static auto find_texture_source(ImgsData& imgs_data, const Point3<double>& vertex, KNN<double>& knn) noexcept
      -> std::vector<int> {
    if(std::isnan(vertex.z)) {
      return {};
    }
    std::array<double, 3>               world_pt{vertex.x, vertex.y, vertex.z};
    Point3<double>                      world_pt_{world_pt[0], world_pt[1], world_pt[2]};
    cv::Mat                             normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);
    std::vector<std::pair<double, int>> img_idxs;
    for(int img_idx : knn.find_nearest_neighbour(Point<double>{vertex.x, vertex.y})) {
      auto& img_data = imgs_data[img_idx];
      if(!img_data.is_valid()) {
        continue;
      }
      cv::Mat view_vec = img_data.t_c2w() - world_pt_;
      THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
      cv::normalize(view_vec, view_vec);
      double cos_angle            = std::abs(view_vec.dot(normal));
      auto   img_coord            = project_point(img_data, imgs_data, world_pt_);
      const auto& [width, height] = img_data.origin_img().get_size();
      if(img_coord.x < 0 || img_coord.y < 0 || img_coord.x >= width || img_coord.y >= height) {
        continue;
      }
      img_idxs.emplace_back(cos_angle, img_idx);
    }
    std::ranges::sort(img_idxs, [](const auto& lhs, const auto& rhs) noexcept { return lhs.first > rhs.first; });
    auto view = img_idxs | std::views::transform([](const auto& pair) noexcept { return pair.second; });
    return {view.begin(), view.end()};
  }

  static auto
  find_best_common_value(const std::vector<int>& v_1, const std::vector<int>& v_2, const std::vector<int>& v_3)
      -> std::optional<int> {
    std::unordered_map<int, int> a_pos;
    std::unordered_map<int, int> b_pos;
    std::unordered_map<int, int> c_pos;
    for(int i = 0; i < v_1.size(); ++i) {
      if(a_pos.find(v_1[i]) == a_pos.end()) {
        a_pos[v_1[i]] = i;
      }
    }
    for(int i = 0; i < v_2.size(); ++i) {
      if(b_pos.find(v_2[i]) == b_pos.end()) {
        b_pos[v_2[i]] = i;
      }
    }
    for(int i = 0; i < v_3.size(); ++i) {
      if(c_pos.find(v_3[i]) == c_pos.end()) {
        c_pos[v_3[i]] = i;
      }
    }
    int                min_sum = std::numeric_limits<int>::max();
    std::optional<int> result;
    for(const auto& [val, idx_a] : a_pos) {
      if(b_pos.find(val) != b_pos.end() && c_pos.find(val) != c_pos.end()) {
        int total_idx = idx_a + b_pos[val] + c_pos[val];
        if(total_idx < min_sum) {
          min_sum = total_idx;
          result  = val;
        }
      }
    }
    return result;
  }

  static auto find_best_texture_for_each_triangle(
      ImgsData&                    imgs_data,
      const Point3s<double>&       vertices,
      const std::vector<Triangle>& triangles,
      Progress&                    progress) -> std::vector<std::vector<int>> {
    std::vector<std::vector<int>> img_tri_pairs{imgs_data.size()};
    std::vector<std::mutex>       mtxs{imgs_data.size()};
    auto knn = KNN<double>(16, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                 return data.get_coord();
                               }) | std::views::common);
    run(
        triangles.size(),
        [&mtxs, &imgs_data, &triangles, &vertices, &img_tri_pairs, &knn](int idx) noexcept {
          auto best1 = find_texture_source(imgs_data, vertices[triangles[idx][0]], knn);
          auto best2 = find_texture_source(imgs_data, vertices[triangles[idx][1]], knn);
          auto best3 = find_texture_source(imgs_data, vertices[triangles[idx][2]], knn);
          auto res   = find_best_common_value(best1, best2, best3);
          if(!res) {
            THIS_LOG_WARN("No common texture source found for triangle {}", idx);
            return;
          }
          int                         img_idx = *res;
          std::lock_guard<std::mutex> lock(mtxs[img_idx]);
          img_tri_pairs[img_idx].push_back(idx);
        },
        progress);
    return img_tri_pairs;
  }

  static auto put_triangle_texture(
      cv::Mat*                texture,
      ImgData&                img_data,
      ImgsData&               imgs_data,
      const std::vector<int>& triangle_indices,
      const Triangles&        triangles,
      const Point3s<double>&  vertices,
      double                  min_x,
      double                  min_y,
      double                  resolution = 10.0) noexcept {
    auto get_pixel = [&img_data, &imgs_data](const Point3<double>& world_pt_) noexcept {
      return project_point(img_data, imgs_data, world_pt_);
    };
    auto img = img_data.origin_img().get();
    run(triangle_indices.size(),
        [&img, &triangles, &triangle_indices, &vertices, &get_pixel, texture, resolution, min_x, min_y](
            int tri_idx) noexcept {
          auto tri_vert = triangles[triangle_indices[tri_idx]]
                          | std::views::transform([&vertices](int idx) noexcept { return vertices[idx]; });
          auto large_tex_view =
              tri_vert | std::views::transform([resolution, min_x, min_y](const auto& vertex) noexcept -> Point<double> {
                return {(vertex.x - min_x) * resolution, (vertex.y - min_y) * resolution};
              });
          auto img_view =
              tri_vert | std::views::transform([&get_pixel](const auto& vertex) noexcept { return get_pixel(vertex); });
          Points<double> large_tex_poly{large_tex_view.begin(), large_tex_view.end()};
          Points<double> img_poly{img_view.begin(), img_view.end()};
          auto           large_tex_roi = bounding_rect(large_tex_poly);
          auto           img_roi       = bounding_rect(img_poly);
          Points<float>  local_large_tex_poly{
              large_tex_poly[0] - large_tex_roi.tl(),
              large_tex_poly[1] - large_tex_roi.tl(),
              large_tex_poly[2] - large_tex_roi.tl()};
          Points<float> local_img_poly{img_poly[0] - img_roi.tl(), img_poly[1] - img_roi.tl(), img_poly[2] - img_roi.tl()};
          cv::Mat affine;
          affine = cv::getAffineTransform(local_img_poly, local_large_tex_poly);
          cv::Rect large_tex_roi_int{
              static_cast<int>(std::floor(large_tex_roi.x)),
              static_cast<int>(std::floor(large_tex_roi.y)),
              static_cast<int>(std::ceil(large_tex_roi.width)),
              static_cast<int>(std::ceil(large_tex_roi.height))};
          cv::Rect img_roi_int{
              static_cast<int>(std::floor(img_roi.x)),
              static_cast<int>(std::floor(img_roi.y)),
              static_cast<int>(std::ceil(img_roi.width)),
              static_cast<int>(std::ceil(img_roi.height))};
          cv::Mat     local_large_tex_mask = cv::Mat::zeros(large_tex_roi_int.size(), CV_8UC1);
          Points<int> local_large_tex_poly_int;
          {
            auto view = convert_arithmetic_type<int>(local_large_tex_poly);
            local_large_tex_poly_int.assign(view.begin(), view.end());
          }
          cv::fillConvexPoly(local_large_tex_mask, local_large_tex_poly_int, cv::Scalar(255));
          cv::Mat new_tex_local = cv::Mat::zeros(large_tex_roi_int.size(), CV_8UC3);
          cv::warpAffine(
              img(img_roi_int), new_tex_local, affine, large_tex_roi_int.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
          new_tex_local.copyTo((*texture)(large_tex_roi_int), local_large_tex_mask);
        });
  }

  static auto render_textured_mesh(
      ImgsData&                    imgs_data,
      const Point3s<double>&       vertices,
      const std::vector<Triangle>& triangles,
      Progress&                    progress,
      double                       resolution = 10.0) noexcept -> cv::Mat {
    THIS_MESSAGE("Start finding best texture for each triangle");
    auto img_tri_pairs = find_best_texture_for_each_triangle(imgs_data, vertices, triangles, progress);

    double min_x  = SkyMerge::min_x(vertices);
    double min_y  = SkyMerge::min_y(vertices);
    double max_x  = SkyMerge::max_x(vertices);
    double max_y  = SkyMerge::max_y(vertices);
    int    width  = static_cast<int>(std::ceil((max_x - min_x) * resolution));
    int    height = static_cast<int>(std::ceil((max_y - min_y) * resolution));

    cv::Mat texture{height, width, CV_8UC3, cv::Scalar(0, 0, 0)};
    THIS_MESSAGE("Start putting triangle texture");
    progress.reset(static_cast<int>(imgs_data.size()));
    for(auto&& [img_data, tris] : std::views::zip(imgs_data, img_tri_pairs)) {
      if(!img_data.is_valid()) {
        continue;
      }
      put_triangle_texture(&texture, img_data, imgs_data, tris, triangles, vertices, min_x, min_y, resolution);
      progress.update();
    }
    return texture;
  }
};

} // namespace SkyMerge

#endif