#ifndef SKYMERGE_ALGO_STITCH1_HPP
#define SKYMERGE_ALGO_STITCH1_HPP

#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <ranges>
#include <unordered_map>
#include <vector>

#include "algo/knn.hpp"
#include "ds/dsm.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace SkyMerge {

class TriMeshStitcher {
public:
  // 三角形结构体，存储三个顶点索引
  struct Triangle {
    int v1, v2, v3;  // 顶点索引
  };

  // 3D顶点结构体
  struct Vertex {
    double x, y, z;
    int best_img_idx = -1;  // 最佳图像索引
    cv::Point2f img_coord;  // 在最佳图像中的坐标

    // 转换为OpenCV点
    cv::Point2f to_cv_point() const {
      return cv::Point2f(static_cast<float>(x), static_cast<float>(y));
    }
  };

  static auto stitch(ImgsData& imgs_data, const std::vector<Vertex>& point_cloud, Progress& progress) noexcept -> cv::Mat {
    if(imgs_data.empty() || point_cloud.empty()) {
      THIS_LOG_ERROR("empty imgs_data or point_cloud");
      return {};
    }

    THIS_MESSAGE("start triangulation and stitching");

    // 第一步：Delaunay三角化
    std::vector<Triangle> triangles;
    auto triangulation_success = delaunay_triangulation(point_cloud, triangles);
    if(!triangulation_success) {
      THIS_LOG_ERROR("Delaunay triangulation failed");
      return {};
    }

    THIS_MESSAGE("triangulation completed, found " + std::to_string(triangles.size()) + " triangles");

    // 第二步：计算每个顶点的最佳纹理源
    std::vector<Vertex> vertices = point_cloud;  // 复制一份，因为我们会修改它
    compute_best_texture_source(imgs_data, vertices, progress);

    // 第三步：渲染纹理化的三角网格
    cv::Mat texture = render_textured_mesh(imgs_data, vertices, triangles, progress);

    cv::imshow("texture", texture);
    cv::waitKey(0);
    return texture;
  }

private:
  // Delaunay三角化
  static bool delaunay_triangulation(const std::vector<Vertex>& point_cloud, std::vector<Triangle>& triangles) noexcept {
    // 将3D点云投影到2D平面
    std::vector<cv::Point2f> points_2d;
    points_2d.reserve(point_cloud.size());
    for(const auto& vertex : point_cloud) {
      points_2d.push_back(vertex.to_cv_point());
    }

    // 使用OpenCV的Subdiv2D进行Delaunay三角化
    cv::Rect rect = cv::boundingRect(points_2d);
    // 扩大边界矩形以确保所有点都包含在内
    rect.x -= rect.width * 0.1;
    rect.y -= rect.height * 0.1;
    rect.width *= 1.2;
    rect.height *= 1.2;

    cv::Subdiv2D subdiv(rect);
    try {
      for(size_t i = 0; i < points_2d.size(); ++i) {
        subdiv.insert(points_2d[i]);
      }
    } catch(const cv::Exception& e) {
      THIS_LOG_ERROR("Error in Delaunay triangulation: " + std::string(e.what()));
      return false;
    }

    // 获取三角形列表
    std::vector<cv::Vec6f> triangleList;
    subdiv.getTriangleList(triangleList);

    // 将OpenCV的三角形转换为我们的Triangle结构
    std::unordered_map<int, int> point_to_vertex;
    for(size_t i = 0; i < points_2d.size(); ++i) {
      point_to_vertex[i] = i;
    }

    triangles.clear();
    for(const auto& t : triangleList) {
      cv::Point2f pt1(t[0], t[1]);
      cv::Point2f pt2(t[2], t[3]);
      cv::Point2f pt3(t[4], t[5]);

      // 检查三角形的所有顶点是否在我们的点集中
      int v1 = -1, v2 = -1, v3 = -1;
      for(size_t i = 0; i < points_2d.size(); ++i) {
        if(std::fabs(pt1.x - points_2d[i].x) < 1e-5 && std::fabs(pt1.y - points_2d[i].y) < 1e-5) {
          v1 = i;
        }
        if(std::fabs(pt2.x - points_2d[i].x) < 1e-5 && std::fabs(pt2.y - points_2d[i].y) < 1e-5) {
          v2 = i;
        }
        if(std::fabs(pt3.x - points_2d[i].x) < 1e-5 && std::fabs(pt3.y - points_2d[i].y) < 1e-5) {
          v3 = i;
        }
      }

      // 只添加所有顶点都在点集中的三角形
      if(v1 != -1 && v2 != -1 && v3 != -1) {
        triangles.push_back({v1, v2, v3});
      }
    }

    return !triangles.empty();
  }

  // 计算每个顶点的最佳纹理源
  static void compute_best_texture_source(ImgsData& imgs_data, std::vector<Vertex>& vertices, Progress& progress) noexcept {
    std::mutex mtx;
    auto knn = KNN<double>(8, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                return data.get_coord();
                              }) | std::views::common);
    progress.reset(vertices.size());

    run(
        vertices.size(),
        [&imgs_data, &vertices, &mtx, &knn](int idx) noexcept {
          auto& vertex = vertices[idx];
          if(std::isnan(vertex.z)) {
            return;
          }

          std::array<double, 3> world_pt{vertex.x, vertex.y, vertex.z};
          int best_img_idx = -1;
          double best_cos_angle = 0.0;
          cv::Point2f best_img_coord;

          cv::Mat normal = (cv::Mat_<double>(3, 1) << 0, 0, -1);  // 假设法向量指向z轴负方向
          for(int img_idx : knn.find_nearest_neighbour(Point<double>{vertex.x, vertex.y})) {
            auto& img_data = imgs_data[img_idx];
            if(!img_data.is_valid()) continue;

            cv::Mat view_vec = img_data.t_c2w() - world_pt_;
            THIS_ASSERTION_SHOULD_LEQ(1e-6, cv::norm(view_vec));
            cv::normalize(view_vec, view_vec);
            double cos_angle = std::abs(view_vec.dot(normal));
            
            if(cos_angle <= best_cos_angle) continue;

            // 计算该点在图像中的投影坐标
            auto cam_pt = world2camera(img_data.A_w2c_array_raw().data(), img_data.t_w2c_array_raw().data(), world_pt.data());
            auto px = camera2pixel(imgs_data.camera_array_raw().data(), imgs_data.distort_array_raw().data(), cam_pt.data());
            cv::Point2f img_coord(static_cast<float>(px.x()), static_cast<float>(px.y()));

            // 检查投影点是否在图像范围内
            const auto& [width, height] = img_data.origin_img().get_size();
            if(img_coord.x < 0 || img_coord.y < 0 || img_coord.x >= width || img_coord.y >= height) {
              continue;
            }

            best_cos_angle = cos_angle;
            best_img_idx = img_idx;
            best_img_coord = img_coord;
          }

          // 更新顶点的最佳纹理源
          if(best_img_idx != -1) {
            std::lock_guard<std::mutex> lock(mtx);
            vertex.best_img_idx = best_img_idx;
            vertex.img_coord = best_img_coord;
          }
        },
        progress);
  }

  // 渲染纹理化的三角网格
  static cv::Mat render_textured_mesh(ImgsData& imgs_data, const std::vector<Vertex>& vertices, 
                                      const std::vector<Triangle>& triangles, Progress& progress) noexcept {
    // 计算渲染区域的边界
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    for(const auto& vertex : vertices) {
      min_x = std::min(min_x, static_cast<float>(vertex.x));
      min_y = std::min(min_y, static_cast<float>(vertex.y));
      max_x = std::max(max_x, static_cast<float>(vertex.x));
      max_y = std::max(max_y, static_cast<float>(vertex.y));
    }

    // 创建渲染结果图像
    int width = static_cast<int>(max_x - min_x) * 20;  // 放大系数可以调整
    int height = static_cast<int>(max_y - min_y) * 20;
    cv::Mat texture(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // 在图像上渲染每个三角形
    progress.reset(triangles.size());
    run(
        triangles.size(),
        [&texture, &vertices, &triangles, &imgs_data, min_x, min_y, width, height](int idx) noexcept {
          const auto& tri = triangles[idx];
          const auto& v1 = vertices[tri.v1];
          const auto& v2 = vertices[tri.v2];
          const auto& v3 = vertices[tri.v3];

          // 如果三角形的任一顶点没有纹理，则跳过
          if(v1.best_img_idx == -1 || v2.best_img_idx == -1 || v3.best_img_idx == -1) {
            return;
          }

          // 对于每个三角形，我们现在简单地使用第一个顶点的图像作为纹理源
          // 实际应用中，可能需要更复杂的策略，如投票或混合
          int img_idx = v1.best_img_idx;
          auto& img_data = imgs_data[img_idx];
          cv::Mat img = img_data.origin_img().get();

          // 将世界坐标映射到渲染图像坐标
          cv::Point2f p1((v1.x - min_x) * width / (max_x - min_x), 
                         (v1.y - min_y) * height / (max_y - min_y));
          cv::Point2f p2((v2.x - min_x) * width / (max_x - min_x), 
                         (v2.y - min_y) * height / (max_y - min_y));
          cv::Point2f p3((v3.x - min_x) * width / (max_x - min_x), 
                         (v3.y - min_y) * height / (max_y - min_y));

          // 计算纹理坐标
          std::vector<cv::Point2f> tri_vertices = {p1, p2, p3};
          std::vector<cv::Point2f> tex_coords = {v1.img_coord, v2.img_coord, v3.img_coord};

          // 创建仿射变换矩阵
          cv::Mat affine_transform = cv::getAffineTransform(tex_coords.data(), tri_vertices.data());
          
          // 使用仿射变换将图像区域映射到三角形
          cv::Mat warped_triangle;
          cv::warpAffine(img, warped_triangle, affine_transform, texture.size(), 
                        cv::INTER_NEAREST, cv::BORDER_REFLECT);

          // 创建三角形掩码
          cv::Mat mask = cv::Mat::zeros(texture.size(), CV_8UC1);
          std::vector<cv::Point> polygon = {
            cv::Point(static_cast<int>(p1.x), static_cast<int>(p1.y)),
            cv::Point(static_cast<int>(p2.x), static_cast<int>(p2.y)),
            cv::Point(static_cast<int>(p3.x), static_cast<int>(p3.y))
          };
          cv::fillConvexPoly(mask, polygon, cv::Scalar(255));

          // 将变换后的三角形复制到结果图像
          cv::Mat masked_warped_triangle;
          warped_triangle.copyTo(masked_warped_triangle, mask);
          
          // 将结果叠加到纹理图像上
          texture = texture + masked_warped_triangle;
        },
        progress);

    return texture;
  }
};

} // namespace SkyMerge

#endif // SKYMERGE_ALGO_STITCH1_HPP 