#ifndef SKYMERGE_ALGO_STITCH1_HPP
#define SKYMERGE_ALGO_STITCH1_HPP

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "algo/knn.hpp"
#include "ds/imgdata.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/report.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

namespace SkyMerge {

class Stitcher {
private:

  using PixelPair  = std::pair<Point<int>, Point<int>>;
  using PixelPairs = std::vector<PixelPair>;

public:

  static auto
  stitch(ImgsData& imgs_data, const PointCloudPtr& point_cloud, Progress& progress, float grid_length = 0.05F) noexcept
      -> cv::Mat {
    if(imgs_data.empty() || point_cloud->empty()) {
      THIS_LOG_ERROR("empty imgs_data or point cloud, cannot stitch");
      return {};
    }
    THIS_MESSAGE("start stitching");
    auto [start_x, end_x, start_y, end_y] = get_min_max_xy(point_cloud);
    auto width                            = static_cast<int>(std::ceil((end_x - start_x) / grid_length));
    auto height                           = static_cast<int>(std::ceil((end_y - start_y) / grid_length));
    auto height_map = height_map_by_grid(point_cloud, width, height, start_x, start_y, grid_length, progress);
#ifdef ENABLE_VISUALIZE_OUTPUT
    double min_val;
    double max_val;
    cv::minMaxLoc(height_map, &min_val, &max_val);
    cv::imwrite("height_map.png", (max_val - height_map) * 255.0 / (max_val - min_val));
#endif
    auto pixel_img_map = find_pixel_map(imgs_data, height_map, width, height, start_x, start_y, grid_length, progress);
    cv::Mat texture    = cv::Mat::zeros(height, width, CV_8UC3);
    run(
        imgs_data.size(),
        [&](int idx) noexcept {
          auto& img_data  = imgs_data[idx];
          auto& pixel_map = pixel_img_map[idx];
          if(!img_data.is_valid() || pixel_img_map.empty()) {
            return;
          }
          auto img = img_data.origin_img().get();
          for(const auto& [texture_pixel, img_pixel] : pixel_map) {
            texture.at<cv::Vec3b>(texture_pixel.y, texture_pixel.x) = img.at<cv::Vec3b>(img_pixel.y, img_pixel.x);
          }
        },
        progress);
    return texture;
  }

private:

  static auto height_map_by_grid(
      PointCloudPtr point_cloud,
      int           width,
      int           height,
      float         start_x,
      float         start_y,
      float         grid_length,
      Progress&     progress) noexcept -> cv::Mat {
    auto calculate_weight = [](float dist) { return 1.0F / (dist * dist * dist); };
    auto calculate_z      = [&point_cloud, calculate_weight](std::vector<int> indices, std::vector<float> distances) {
      float z_sum   = 0.0;
      float divisor = 0.0;
      for(auto&& [idx, dist] : std::views::zip(indices, distances)) {
        auto point = point_cloud->points[idx].getVector3fMap();
        if(dist < 1e-6F) {
          return point.z();
        }
        auto weight = calculate_weight(dist);
        z_sum += point.z() * weight;
        divisor += weight;
      }
      return z_sum / divisor;
    };
    auto point_cloud_2d = std::make_shared<pcl::PointCloud<pcl::PointXY>>();
    point_cloud_2d->reserve(point_cloud->size());
    for(const auto& point : *point_cloud) {
      auto point_ = point.getVector3fMap();
      point_cloud_2d->emplace_back(point_.x(), point_.y());
    }
    cv::Mat height_map = cv::Mat::zeros(height, width, CV_32FC1);
    // run(
    //     width,
    //     [start_x, start_y, grid_length, width, height, &point_cloud_2d, &height_map, &calculate_z](int x_i) noexcept {
    //       pcl::KdTreeFLANN<pcl::PointXY> kd_tree;
    //       kd_tree.setInputCloud(point_cloud_2d);
    //       float x_pos = start_x + (static_cast<float>(x_i) * grid_length);
    //       for(int y_i = 0; y_i < height; ++y_i) {
    //         float              y_pos = start_y + (static_cast<float>(y_i) * grid_length);
    //         pcl::PointXY       search_point(x_pos, y_pos);
    //         std::vector<int>   indices;
    //         std::vector<float> distances;
    //         kd_tree.nearestKSearch(search_point, 100, indices, distances);
    //         if(indices.empty()) {
    //           continue;
    //         }
    //         height_map.at<float>(y_i, x_i) = calculate_z(indices, distances);
    //       }
    //     },
    //     progress);
    return height_map;
  }

  static auto find_pixel_map(
      ImgsData&      imgs_data,
      const cv::Mat& height_map,
      int            width,
      int            height,
      float          start_x,
      float          start_y,
      float          grid_length,
      Progress&      progress) noexcept -> std::vector<PixelPairs> {
    auto knn = KNN<double>(16, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                 return data.get_coord();
                               }) | std::views::common);
    std::vector<PixelPairs> pixel_img_map{imgs_data.size()};
    std::vector<std::mutex> mtxs{imgs_data.size()};
    cv::Mat                 texture_source = cv::Mat::zeros(height, width, CV_32SC4);
    run(
        static_cast<std::int64_t>(width) * height,
        [&](int idx) noexcept {
          int   x_i   = idx / height;
          int   y_i   = idx % height;
          float x_pos = start_x + (static_cast<float>(x_i) * grid_length);
          float y_pos = start_y + (static_cast<float>(y_i) * grid_length);
          Point3<double> world_pt{static_cast<double>(x_pos), static_cast<double>(y_pos), height_map.at<float>(y_i, x_i)};
          double     best_dist = std::numeric_limits<double>::max();
          int        best_idx  = -1;
          Point<int> best_pixel{-1, -1};
          for(int img_idx : knn.find_nearest_neighbour({world_pt.x, world_pt.y})) {
            if(!imgs_data[img_idx].is_valid()) {
              continue;
            }
            auto& img_data = imgs_data[img_idx];
            auto  pixel_   = project_point(img_data, imgs_data, world_pt);
            auto  pixel    = Point<int>{static_cast<int>(std::round(pixel_.x)), static_cast<int>(std::round(pixel_.y))};
            auto [width, height] = img_data.origin_img().get_size();
            if(pixel.x < 0 || pixel.y < 0 || pixel.x >= width || pixel.y >= height) {
              continue;
            }
            Point<double> img_center{static_cast<double>(width) / 2.0, static_cast<double>(height) / 2.0};
            double        dist = std::hypot(pixel.x - img_center.x, pixel.y - img_center.y);
            if(dist < best_dist) {
              best_dist  = dist;
              best_idx   = img_idx;
              best_pixel = pixel;
            }
          }
          if(best_idx != -1) {
            std::lock_guard lock{mtxs[best_idx]};
            pixel_img_map[best_idx].emplace_back(Point<int>{x_i, y_i}, best_pixel);
          }
        },
        progress);
    return pixel_img_map;
  }

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

  static auto get_min_max_xy(const PointCloudPtr& point_cloud) noexcept -> std::tuple<float, float, float, float> {
    pcl::PointXYZ min_pnt;
    pcl::PointXYZ max_pnt;
    pcl::getMinMax3D(*point_cloud, min_pnt, max_pnt);
    auto start_x = min_pnt.getVector3fMap().x();
    auto end_x   = max_pnt.getVector3fMap().x();
    auto start_y = min_pnt.getVector3fMap().y();
    auto end_y   = max_pnt.getVector3fMap().y();
    return {start_x, end_x, start_y, end_y};
  }
};

} // namespace SkyMerge

#endif