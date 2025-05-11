#ifndef SKYMERGE_ALGO_STITCH1_HPP
#define SKYMERGE_ALGO_STITCH1_HPP

#include <mutex>
#include <optional>
#include <ranges>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <pcl/impl/point_types.hpp>

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

  using BestPixel = std::pair<int, Point<double>>;

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
    Point3s<double> vertices;
    for(const auto& point : *point_cloud) {
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

  static auto project_point(ImgData& img_data, ImgsData& imgs_data, const Point3<double>& world_pt_) noexcept
      -> Point<double> {
    std::array<double, 3> world_pt{world_pt_.x, world_pt_.y, world_pt_.z};
    auto cam_pt = world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), world_pt.data());
    auto p_x = camera2pixel(imgs_data.camera_array_raw().data(), imgs_data.distort_array_raw().data(), cam_pt.data());
    return {p_x.x(), p_x.y()};
  }

  static auto find_best_texture_source(ImgsData& imgs_data, const Point3<double>& vertex, KNN<double>& knn) noexcept
      -> std::optional<BestPixel> {
    if(std::isnan(vertex.z)) {
      return std::nullopt;
    }
    std::array<double, 3> world_pt{vertex.x, vertex.y, vertex.z};
    Point3<double>        world_pt_{world_pt[0], world_pt[1], world_pt[2]};
    BestPixel             best_pixel{-1, {0, 0}};
    double                best_cos_angle = 0.0;
    cv::Mat               normal         = (cv::Mat_<double>(3, 1) << 0, 0, -1);
    for(int img_idx : knn.find_nearest_neighbour(Point<double>{vertex.x, vertex.y})) {
      auto& img_data = imgs_data[img_idx];
      if(!img_data.is_valid()) {
        continue;
      }
      cv::Mat view_vec = img_data.t_c2w() - world_pt_;
      THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
      cv::normalize(view_vec, view_vec);
      double cos_angle = std::abs(view_vec.dot(normal));
      if(cos_angle <= best_cos_angle) {
        continue;
      }
      auto img_coord              = project_point(img_data, imgs_data, world_pt_);
      const auto& [width, height] = img_data.origin_img().get_size();
      if(img_coord.x < 0 || img_coord.y < 0 || img_coord.x >= width || img_coord.y >= height) {
        continue;
      }
      best_cos_angle = cos_angle;
      best_pixel     = {img_idx, img_coord};
    }
    return best_pixel.first != -1 ? std::make_optional(best_pixel) : std::nullopt;
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
          auto tri_vert =
              triangles[idx] | std::views::transform([&vertices](int idx) noexcept { return vertices[idx]; });
          auto tri_center = avg(tri_vert);
          auto res        = find_best_texture_source(imgs_data, tri_center, knn);
          if(!res) {
            THIS_LOG_WARN("find_best_texture_source failed");
            return;
          }
          auto [img_idx, _] = *res;
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
                return {std::max(0.0, (vertex.x - min_x)) * resolution, std::max(0.0, (vertex.y - min_y)) * resolution};
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
          cv::Mat     affine               = cv::getAffineTransform(local_img_poly, local_large_tex_poly);
          cv::Rect    large_tex_roi_int    = large_tex_roi;
          cv::Rect    img_roi_int          = img_roi;
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
    int    width  = static_cast<int>((max_x - min_x) * resolution);
    int    height = static_cast<int>((max_y - min_y) * resolution);

    cv::Mat texture{height, width, CV_8UC3, cv::Scalar(0, 0, 0)};
    for(auto&& [idx, pair] : std::views::zip(imgs_data, img_tri_pairs) | std::views::enumerate) {
      const auto& [img_data, tris] = pair;
      if(!img_data.is_valid()) {
        continue;
      }
      THIS_MESSAGE("Start stitching image {}/{}", idx, imgs_data.size());
      put_triangle_texture(&texture, img_data, imgs_data, tris, triangles, vertices, min_x, min_y, resolution);
    }
    return texture;
  }
};

} // namespace SkyMerge

#endif