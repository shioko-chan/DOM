#ifndef SKYMERGE_ALGO_STITCH1_HPP
#define SKYMERGE_ALGO_STITCH1_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>

#include <opencv2/core.hpp>

#include "algo/knn.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/report_error.hpp"
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
    auto            new_point_cloud = grid_downsample_2d(point_cloud, 1.0);
    Point3s<double> vertices;
    for(const auto& point : *new_point_cloud) {
      auto pnt = point.getVector3fMap();
      vertices.emplace_back(pnt.x(), pnt.y(), pnt.z());
    }
    THIS_MESSAGE("开始三角剖分");
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
      THIS_LOG_ERROR("Delaunay三角剖分失败");
      return {};
    }
    THIS_MESSAGE("三角剖分完成，找到 " + std::to_string(triangles.size()) + " 个三角形");
    auto texture = render_textured_mesh(imgs_data, vertices, triangles, progress, resolution);
    cv::flip(texture, texture, 0);
    return texture;
  }

private:

  static auto
  grid_downsample_2d(const pcl::PointCloud<pcl::PointXYZ>::Ptr& point_cloud, float distance_threshold = 0.5) noexcept
      -> pcl::PointCloud<pcl::PointXYZ>::Ptr {
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
            THIS_LOG_INFO("No common texture source found for triangle {}", idx);
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
    auto img = img_data.origin_img().get();
    auto get_pixel = [&img_data, &imgs_data, size = img.size()](const Point3<double>& world_pt_) noexcept -> Point<int> {
      auto pnt = project_point(img_data, imgs_data, world_pt_);
      return {
          static_cast<int>(std::clamp(pnt.x, 0.0, static_cast<double>(size.width - 1))),
          static_cast<int>(std::clamp(pnt.y, 0.0, static_cast<double>(size.height - 1)))};
    };
    auto get_large_tex = [resolution, min_x, min_y, size = texture->size()](const auto& vertex) noexcept -> Point<int> {
      return {
          static_cast<int>(std::clamp((vertex.x - min_x) * resolution, 0.0, static_cast<double>(size.width - 1))),
          static_cast<int>(std::clamp((vertex.y - min_y) * resolution, 0.0, static_cast<double>(size.height - 1)))};
    };
    run(triangle_indices.size(),
        [&img, &triangles, &triangle_indices, &vertices, &get_pixel, &get_large_tex, texture, resolution](
            int tri_idx) noexcept {
          auto tri_vert = triangles[triangle_indices[tri_idx]]
                          | std::views::transform([&vertices](int idx) noexcept { return vertices[idx]; });
          Points<int> large_tex_poly;
          {
            auto view = tri_vert | std::views::transform(get_large_tex);
            large_tex_poly.assign(view.begin(), view.end());
          }
          Points<int> img_poly;
          {
            auto view = tri_vert | std::views::transform(get_pixel);
            img_poly.assign(view.begin(), view.end());
          }
          auto        large_tex_roi = cv::boundingRect(large_tex_poly);
          auto        img_roi       = cv::boundingRect(img_poly);
          Points<int> local_large_tex_poly{
              large_tex_poly[0] - large_tex_roi.tl(),
              large_tex_poly[1] - large_tex_roi.tl(),
              large_tex_poly[2] - large_tex_roi.tl()};
          Points<int> local_img_poly{img_poly[0] - img_roi.tl(), img_poly[1] - img_roi.tl(), img_poly[2] - img_roi.tl()};
          cv::Mat affine;
          {
            auto local_img_poly_f       = convert_arithmetic_type_point<float>(local_img_poly);
            auto local_large_tex_poly_f = convert_arithmetic_type_point<float>(local_large_tex_poly);
            affine                      = cv::getAffineTransform(local_img_poly_f, local_large_tex_poly_f);
          }
          cv::Mat local_large_tex_mask = cv::Mat::zeros(large_tex_roi.size(), CV_8UC1);
          try {
            //
            // cv::polylines(img(img_roi), local_img_poly, true, cv::Scalar(255, 255, 255), 1);
            //
            cv::fillConvexPoly(local_large_tex_mask, local_large_tex_poly, cv::Scalar(255));
            cv::Mat new_tex_local = cv::Mat::zeros(large_tex_roi.size(), CV_8UC3);
            cv::warpAffine(
                img(img_roi), new_tex_local, affine, large_tex_roi.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
            new_tex_local.copyTo((*texture)(large_tex_roi), local_large_tex_mask);
          } catch(const cv::Exception& e) {
            std::stringstream ss;
            ss << img.size() << " " << img_roi << " " << texture->size() << " " << large_tex_roi;
            report_error(e, "{}", ss.str());
            return;
          }
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