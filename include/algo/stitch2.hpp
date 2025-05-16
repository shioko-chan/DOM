#ifndef SKYMERGE_ALGO_STITCH_HPP
#define SKYMERGE_ALGO_STITCH_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>
#include <vector>

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "algo/filter.hpp"
#include "algo/knn.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/report_error.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

#include "ds/dsm.hpp"

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
      int                        img_idx;
      int                        dsm_idx;
      std::array<cv::Point2f, 4> img_corners;
      int                        start_x, start_y, end_x, end_y;
    };

    std::mutex            mtx;
    std::vector<PatchSrc> patch_src_map;
    auto                  knn = KNN<double>(8, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
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
          std::array<double, 3>      world_pt{world_pt_.x, world_pt_.y, world_pt_.z};
          BestPixel                  best_pixel{.img_idx = -1, .pixel = {-1, -1}, .cos_angle = 0.0};
          cv::Mat                    normal         = (cv::Mat_<double>(3, 1) << 0, 0, -1);
          int                        best_img_idx   = -1;
          double                     best_cos_angle = 0.0;
          std::array<cv::Point2f, 4> best_img_corners;
          for(int img_idx : knn.find_nearest_neighbour(Point<double>{world_pt_.x, world_pt_.y})) {
            auto&   img_data = imgs_data[img_idx];
            cv::Mat view_vec = img_data.t_c2w() - world_pt_;
            THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
            cv::normalize(view_vec, view_vec);
            double cos_angle = std::abs(view_vec.dot(normal));
            if(cos_angle <= best_cos_angle)
              continue;
            int                                  t_x           = idx % dsm.cols();
            int                                  t_y           = idx / dsm.cols();
            double                               res           = dsm.resolution();
            double                               z             = world_pt_.z;
            double                               cx            = world_pt_.x;
            double                               cy            = world_pt_.y;
            std::array<std::array<double, 3>, 4> world_corners = {
                std::array<double, 3>{cx - res / 2, cy - res / 2, z}, // 左上
                std::array<double, 3>{cx + res / 2, cy - res / 2, z}, // 右上
                std::array<double, 3>{cx + res / 2, cy + res / 2, z}, // 右下
                std::array<double, 3>{cx - res / 2, cy + res / 2, z}  // 左下
            };
            std::array<cv::Point2f, 4> img_corners;
            bool                       valid = true;
            for(int i = 0; i < 4; ++i) {
              auto cam_pt = world2camera(
                  img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), world_corners[i].data());
              auto px =
                  camera2pixel(imgs_data.camera_array_raw().data(), imgs_data.distort_array_raw().data(), cam_pt.data());
              img_corners[i] = cv::Point2f(static_cast<float>(px.x()), static_cast<float>(px.y()));
              // 可选：检查是否在图像范围内
              const auto& [width, height] = img_data.origin_img().get_size();
              if(img_corners[i].x < 0 || img_corners[i].y < 0 || img_corners[i].x >= width
                 || img_corners[i].y >= height) {
                valid = false;
              }
            }
            if(!valid)
              continue;
            best_cos_angle   = cos_angle;
            best_img_idx     = img_idx;
            best_img_corners = img_corners;
          }
          if(best_img_idx != -1) {
            int                         t_x              = idx % dsm.cols();
            int                         t_y              = idx / dsm.cols();
            double                      resolution_ratio = dsm.resolution() / target_resolution;
            int                         start_x          = static_cast<int>(t_x * resolution_ratio);
            int                         start_y          = static_cast<int>(t_y * resolution_ratio);
            int                         end_x            = static_cast<int>((t_x + 1) * resolution_ratio);
            int                         end_y            = static_cast<int>((t_y + 1) * resolution_ratio);
            std::lock_guard<std::mutex> lock(mtx);
            patch_src_map.push_back(PatchSrc{
                .img_idx     = best_img_idx,
                .dsm_idx     = idx,
                .img_corners = best_img_corners,
                .start_x     = start_x,
                .start_y     = start_y,
                .end_x       = end_x,
                .end_y       = end_y});
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
          const auto&              patch    = patch_src_map[idx];
          auto&                    img_data = imgs_data[patch.img_idx];
          cv::Mat                  img      = img_data.origin_img().get();
          int                      patch_w  = patch.end_x - patch.start_x;
          int                      patch_h  = patch.end_y - patch.start_y;
          std::vector<cv::Point2f> patch_corners =
              {cv::Point2f(0, 0),
               cv::Point2f(static_cast<float>(patch_w - 1), 0),
               cv::Point2f(static_cast<float>(patch_w - 1), static_cast<float>(patch_h - 1)),
               cv::Point2f(0, static_cast<float>(patch_h - 1))};
          cv::Mat H = cv::getPerspectiveTransform(patch.img_corners.data(), patch_corners.data());
          cv::Mat patch_img;
          cv::warpPerspective(img, patch_img, H, cv::Size(patch_w, patch_h), cv::INTER_NEAREST, cv::BORDER_REFLECT);
          cv::Mat roi;
          if(patch_img.size() != cv::Size(patch_w, patch_h)) {
            cv::resize(patch_img, roi, cv::Size(patch_w, patch_h), 0, 0, cv::INTER_NEAREST);
          } else {
            roi = patch_img;
          }
          roi.copyTo(texture(cv::Rect(patch.start_x, patch.start_y, patch_w, patch_h)));
        },
        progress);
    cv::transpose(texture, texture);
    cv::flip(texture, texture, 0);
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

  static auto grid_sample_2d(const PointCloudPtr& point_cloud, float distance_threshold = 0.5) noexcept
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
        kd_tree.nearestKSearch(search_point, 100, indices, distances);
        if(!indices.empty()) {
          double z      = 0;
          double factor = std::accumulate(distances.begin(), distances.end(), 0.0, [](double a, double b) {
            return a + (1.0 / (b * b));
          });
          for(auto&& [idx, dist] : std::views::zip(indices, distances)) {
            auto point = point_cloud->points[idx].getVector3fMap();
            z += point.z() / (dist * dist);
          }
          point_grid->emplace_back(x_pos, y_pos, z / factor);
        }
      }
    }
    return point_grid;
  }

  static auto
  stitch(ImgsData& imgs_data, const PointCloudPtr& point_cloud, Progress& progress, double resolution = 10.0) noexcept
      -> cv::Mat {
    if(imgs_data.empty()) {
      THIS_LOG_ERROR("empty imgs data");
      return {};
    }
    if(point_cloud->empty()) {
      THIS_LOG_ERROR("empty point cloud");
      return {};
    }
    auto new_point_cloud = time_function(grid_sample_2d, point_cloud, GRID_LENGTH);
    // auto            new_point_cloud = point_cloud;
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
    // cv::transpose(texture, texture);
    // cv::flip(texture, texture, 0);
    return texture;
  }

  static auto
  stitch(ImgsData& imgs_data, const TrackPointVec& point_cloud, Progress& progress, double resolution = 10.0) noexcept
      -> cv::Mat {
    if(imgs_data.empty()) {
      THIS_LOG_ERROR("empty imgs data");
      return {};
    }
    if(point_cloud.empty()) {
      THIS_LOG_ERROR("empty point cloud");
      return {};
    }
    THIS_MESSAGE("开始三角剖分");
    auto point_2d_view = point_cloud | std::views::enumerate
                         | std::views::transform([](const auto& tuple) noexcept -> std::pair<K::Point_2, int> {
                             const auto& pnt3d = std::get<1>(tuple).pnt3d;
                             return {{pnt3d[0], pnt3d[1]}, static_cast<int>(std::get<0>(tuple))};
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
    auto texture = render_textured_mesh(imgs_data, point_cloud, triangles, progress, resolution);
    // cv::transpose(texture, texture);
    // cv::flip(texture, texture, 0);
    return texture;
  }

private:

  static auto project_point(ImgData& img_data, ImgsData& imgs_data, const Point3<double>& world_pt_) noexcept
      -> Point<double> {
    Eigen::Vector3d world{world_pt_.x, world_pt_.y, world_pt_.z};
    auto            pixel = world2pixel(
        img_data.A_w2c_array_raw().data(),
        img_data.t_w2c_array_raw().data(),
        imgs_data.camera_array_raw().data(),
        imgs_data.distort_array_raw().data(),
        world.data());
    return {pixel.x(), pixel.y()};
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

  static auto group_triangles_by_image(
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

  static void paint_triangles_to_texture(
      cv::Mat*                                                          texture,
      const cv::Mat&                                                    img,
      const std::vector<int>&                                           triangle_indices,
      const Triangles&                                                  triangles,
      const Point3s<double>&                                            vertices,
      NoexceptCallableWithRet<Point<int>, const Point3<double>&> auto&& get_img_pixel,
      NoexceptCallableWithRet<Point<int>, const Point3<double>&> auto&& get_texture_pixel,
      int                                                               idx) noexcept {
    run(triangle_indices.size(),
        [idx, &img, &triangles, &triangle_indices, &vertices, &get_img_pixel, &get_texture_pixel, texture](
            int tri_idx) noexcept {
          auto tri_vert = triangles[triangle_indices[tri_idx]]
                          | std::views::transform([&vertices](int idx) noexcept { return vertices[idx]; });
          Points<int> large_tex_poly;
          {
            auto view = tri_vert | std::views::transform(get_texture_pixel);
            large_tex_poly.assign(view.begin(), view.end());
          }
          Points<int> img_poly;
          {
            auto view = tri_vert | std::views::transform(get_img_pixel);
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
          Point<int> av = avg(large_tex_poly);
          cv::putText(*texture, std::format("{}", idx), av, 0, 0.5, cv::Scalar{255, 255, 255});
          cv::polylines(*texture, large_tex_poly, true, cv::Scalar(255, 255, 255), 1);
        });
  }

  static void paint_triangles_to_texture(
      cv::Mat*                                                          texture,
      const cv::Mat&                                                    img,
      const std::vector<int>&                                           triangle_indices,
      const Triangles&                                                  triangles,
      const Point3s<double>&                                            vertices,
      NoexceptCallableWithRet<Point<int>, const Point3<double>&> auto&& get_img_pixel,
      NoexceptCallableWithRet<Point<int>, const Point3<double>&> auto&& get_texture_pixel) noexcept {
    run(triangle_indices.size(),
        [&img, &triangles, &triangle_indices, &vertices, &get_img_pixel, &get_texture_pixel, texture](
            int tri_idx) noexcept {
          auto tri_vert = triangles[triangle_indices[tri_idx]]
                          | std::views::transform([&vertices](int idx) noexcept { return vertices[idx]; });
          Points<int> large_tex_poly;
          {
            auto view = tri_vert | std::views::transform(get_texture_pixel);
            large_tex_poly.assign(view.begin(), view.end());
          }
          Points<int> img_poly;
          {
            auto view = tri_vert | std::views::transform(get_img_pixel);
            img_poly.assign(view.begin(), view.end());
          }
          auto        large_tex_roi = cv::boundingRect(large_tex_poly);
          auto        img_roi       = cv::boundingRect(img_poly);
          Points<int> local_large_tex_poly{
              large_tex_poly[0] - large_tex_roi.tl(),
              large_tex_poly[1] - large_tex_roi.tl(),
              large_tex_poly[2] - large_tex_roi.tl()};
          Points<int> local_img_poly{img_poly[0] - img_roi.tl(), img_poly[1] - img_roi.tl(), img_poly[2] - img_roi.tl()};
          try {
            auto    local_img_poly_f       = convert_arithmetic_type_point<float>(local_img_poly);
            auto    local_large_tex_poly_f = convert_arithmetic_type_point<float>(local_large_tex_poly);
            cv::Mat affine                 = cv::getAffineTransform(local_img_poly_f, local_large_tex_poly_f);
            cv::Mat local_large_tex_mask   = cv::Mat::zeros(large_tex_roi.size(), CV_8UC1);
            cv::fillConvexPoly(local_large_tex_mask, local_large_tex_poly, cv::Scalar(255));
            cv::Mat new_tex_local = cv::Mat::zeros(large_tex_roi.size(), CV_8UC3);
            cv::warpAffine(
                img(img_roi), new_tex_local, affine, large_tex_roi.size(), cv::INTER_LANCZOS4, cv::BORDER_CONSTANT);
            new_tex_local.copyTo((*texture)(large_tex_roi), local_large_tex_mask);
          } catch(const cv::Exception& except) {
            report_error(except, "[TriMeshStitcher] CV ERROR");
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
    auto tri_groups_by_img = group_triangles_by_image(imgs_data, vertices, triangles, progress);

    double min_x = SkyMerge::min_x(vertices);
    double min_y = SkyMerge::min_y(vertices);
    double max_x = SkyMerge::max_x(vertices);
    double max_y = SkyMerge::max_y(vertices);

    int width  = static_cast<int>(std::ceil((max_x - min_x) * resolution));
    int height = static_cast<int>(std::ceil((max_y - min_y) * resolution));

    cv::Mat texture{height, width, CV_8UC3, cv::Scalar(0, 0, 0)};
    cv::Mat text{height, width, CV_8UC3, cv::Scalar(0, 0, 0)};

    THIS_MESSAGE("Start putting triangle texture");
    progress.reset(static_cast<int>(imgs_data.size()));
    int idx = -1;
    for(auto&& [img_data, tris] : std::views::zip(imgs_data, tri_groups_by_img)) {
      ++idx;
      if(!img_data.is_valid()) {
        continue;
      }
      auto img = img_data.origin_img().get();
      paint_triangles_to_texture(
          &text,
          img,
          tris,
          triangles,
          vertices,
          [&img_data, &imgs_data, size = img.size()](const Point3<double>& world) noexcept -> Point<int> {
            auto pixel = project_point(img_data, imgs_data, world);
            return {
                static_cast<int>(std::round(std::clamp(pixel.x, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(std::round(std::clamp(pixel.y, 0.0, static_cast<double>(size.height - 1))))};
          },
          [resolution, min_x, min_y, size = texture.size()](const Point3<double>& world) noexcept -> Point<int> {
            return {
                static_cast<int>(
                    std::round(std::clamp((world.x - min_x) * resolution, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(
                    std::round(std::clamp((world.y - min_y) * resolution, 0.0, static_cast<double>(size.height - 1))))};
          },
          idx);
      paint_triangles_to_texture(
          &texture,
          img,
          tris,
          triangles,
          vertices,
          [&img_data, &imgs_data, size = img.size()](const Point3<double>& world) noexcept -> Point<int> {
            auto pixel = project_point(img_data, imgs_data, world);
            return {
                static_cast<int>(std::round(std::clamp(pixel.x, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(std::round(std::clamp(pixel.y, 0.0, static_cast<double>(size.height - 1))))};
          },
          [resolution, min_x, min_y, size = texture.size()](const Point3<double>& world) noexcept -> Point<int> {
            return {
                static_cast<int>(
                    std::round(std::clamp((world.x - min_x) * resolution, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(
                    std::round(std::clamp((world.y - min_y) * resolution, 0.0, static_cast<double>(size.height - 1))))};
          });
      progress.update();
    }
    cv::imwrite("text.png", text);
    return texture;
  }

  static auto group_triangles_by_image(
      size_t                       img_num,
      const TrackPointVec&         vertices,
      const std::vector<Triangle>& triangles,
      Progress&                    progress) -> std::vector<std::vector<int>> {
    std::vector<std::vector<int>> img_tri_pairs{img_num};
    std::vector<std::mutex>       mtxs{img_num};
    auto                          find_texture_source = [&vertices](int idx) noexcept -> std::vector<int> {
      auto view = vertices[idx].pnt2d_idx_vec
                  | std::views::transform([](const auto& pnt2d_idx) noexcept { return pnt2d_idx.img_idx; });
      return {view.begin(), view.end()};
    };
    run(
        triangles.size(),
        [&triangles, &vertices, &img_tri_pairs, &mtxs, &find_texture_source](int idx) noexcept {
          auto best1 = find_texture_source(triangles[idx][0]);
          auto best2 = find_texture_source(triangles[idx][1]);
          auto best3 = find_texture_source(triangles[idx][2]);
          auto res   = find_best_common_value(best1, best2, best3);
          if(!res) {
            std::stringstream ss;
            ss << "best1:";
            for(const auto& i : best1) {
              ss << i << ",";
            }
            ss << "best2:";
            for(const auto& i : best2) {
              ss << i << ",";
            }
            ss << "best3:";
            for(const auto& i : best3) {
              ss << i << ",";
            }
            THIS_LOG_INFO("No common texture source found for triangle {} {}", idx, ss.str());
            return;
          }
          int                         img_idx = *res;
          std::lock_guard<std::mutex> lock(mtxs[img_idx]);
          img_tri_pairs[img_idx].push_back(idx);
        },
        progress);
    return img_tri_pairs;
  }

  static auto render_textured_mesh(
      ImgsData&                    imgs_data,
      const TrackPointVec&         vertices,
      const std::vector<Triangle>& triangles,
      Progress&                    progress,
      double                       resolution = 10.0) noexcept -> cv::Mat {
    THIS_MESSAGE("Start finding best texture for each triangle");
    auto            tri_groups_by_img = group_triangles_by_image(imgs_data.size(), vertices, triangles, progress);
    auto            vertices_v        = vertices | std::views::transform([](const auto& pnt) noexcept {
                        return Point3<double>{pnt.pnt3d[0], pnt.pnt3d[1], pnt.pnt3d[2]};
                      });
    Point3s<double> vertices_(vertices_v.begin(), vertices_v.end());
    double          min_x  = SkyMerge::min_x(vertices_);
    double          min_y  = SkyMerge::min_y(vertices_);
    double          max_x  = SkyMerge::max_x(vertices_);
    double          max_y  = SkyMerge::max_y(vertices_);
    int             width  = static_cast<int>(std::ceil((max_x - min_x) * resolution));
    int             height = static_cast<int>(std::ceil((max_y - min_y) * resolution));
    cv::Mat         texture{height, width, CV_8UC3, cv::Scalar(0, 0, 0)};
    cv::Mat         text{height, width, CV_8UC3, cv::Scalar(0, 0, 0)};
    THIS_MESSAGE("Start putting triangle texture");
    progress.reset(static_cast<int>(imgs_data.size()));
    int idx = -1;
    for(auto&& [img_data, tris] : std::views::zip(imgs_data, tri_groups_by_img)) {
      ++idx;
      if(!img_data.is_valid()) {
        continue;
      }
      auto img = img_data.origin_img().get();
      paint_triangles_to_texture(
          &text,
          img,
          tris,
          triangles,
          vertices_,
          [&img_data, &imgs_data, size = img.size()](const Point3<double>& world) noexcept -> Point<int> {
            auto pixel = project_point(img_data, imgs_data, world);
            return {
                static_cast<int>(std::round(std::clamp(pixel.x, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(std::round(std::clamp(pixel.y, 0.0, static_cast<double>(size.height - 1))))};
          },
          [resolution, min_x, min_y, size = texture.size()](const Point3<double>& world) noexcept -> Point<int> {
            return {
                static_cast<int>(
                    std::round(std::clamp((world.x - min_x) * resolution, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(
                    std::round(std::clamp((world.y - min_y) * resolution, 0.0, static_cast<double>(size.height - 1))))};
          },
          idx);
      paint_triangles_to_texture(
          &texture,
          img,
          tris,
          triangles,
          vertices_,
          [&img_data, &imgs_data, size = img.size()](const Point3<double>& world) noexcept -> Point<int> {
            auto pixel = project_point(img_data, imgs_data, world);
            return {
                static_cast<int>(std::round(std::clamp(pixel.x, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(std::round(std::clamp(pixel.y, 0.0, static_cast<double>(size.height - 1))))};
          },
          [resolution, min_x, min_y, size = texture.size()](const Point3<double>& world) noexcept -> Point<int> {
            return {
                static_cast<int>(
                    std::round(std::clamp((world.x - min_x) * resolution, 0.0, static_cast<double>(size.width - 1)))),
                static_cast<int>(
                    std::round(std::clamp((world.y - min_y) * resolution, 0.0, static_cast<double>(size.height - 1))))};
          });
      progress.update();
    }
    cv::imwrite("text.png", text);
    return texture;
  }
};

} // namespace SkyMerge

#endif