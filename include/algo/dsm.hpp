#ifndef ORTHO_DSM_HPP
#define ORTHO_DSM_HPP

#include <pcl/common/common.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/report_error.hpp"

namespace Ortho {

class DSM {
public:
  DSM() noexcept = default;

  DSM(cv::Mat height_map, double resolution, float min_x, float min_y) noexcept :
      height_map_(std::move(height_map)),
      resolution_(resolution),
      min_x_(min_x),
      min_y_(min_y) {}

  [[nodiscard]] double resolution() const noexcept { return resolution_; }
  [[nodiscard]] float min_x() const noexcept { return min_x_; }
  [[nodiscard]] float min_y() const noexcept { return min_y_; }
  [[nodiscard]] float max_x() const noexcept { return min_x_ + (cols() - 1) * resolution_; }
  [[nodiscard]] float max_y() const noexcept { return min_y_ + (rows() - 1) * resolution_; }
  [[nodiscard]] int rows() const noexcept { return height_map_.rows; }
  [[nodiscard]] int cols() const noexcept { return height_map_.cols; }
  [[nodiscard]] bool empty() const noexcept { return height_map_.empty(); }
  [[nodiscard]] cv::Size size() const noexcept { return height_map_.size(); }

  [[nodiscard]] float get_height(int row, int col) const noexcept {
    return height_map_.at<float>(row, col);
  }

  [[nodiscard]] float get_height_world(float world_x, float world_y) const noexcept {
    int col = static_cast<int>((world_x - min_x_) / resolution_);
    int row = static_cast<int>((world_y - min_y_) / resolution_);
    
    if (row < 0 || row >= rows() || col < 0 || col >= cols()) {
      return std::numeric_limits<float>::quiet_NaN();
    }
    
    return get_height(row, col);
  }

  [[nodiscard]] cv::Point2i world_to_grid(float world_x, float world_y) const noexcept {
    int col = static_cast<int>((world_x - min_x_) / resolution_);
    int row = static_cast<int>((world_y - min_y_) / resolution_);
    return {col, row};
  }

  [[nodiscard]] cv::Point2f grid_to_world(int row, int col) const noexcept {
    float world_x = min_x_ + col * resolution_;
    float world_y = min_y_ + row * resolution_;
    return {world_x, world_y};
  }
  
  [[nodiscard]] cv::Point3f grid_to_world_3d(int row, int col) const noexcept {
    float world_x = min_x_ + col * resolution_;
    float world_y = min_y_ + row * resolution_;
    float height = get_height(row, col);
    return {world_x, world_y, height};
  }
  
  [[nodiscard]] const cv::Mat& height_map() const noexcept { return height_map_; }
  
  void save_as_image(const std::string& output_path, bool normalize = true) const {
    save_dsm_as_image(height_map_, output_path, normalize);
  }

private:
  cv::Mat height_map_;
  double resolution_ = 0.0;
  float min_x_ = 0.0f;
  float min_y_ = 0.0f;
};

inline auto pointcloud_to_dsm(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) noexcept -> DSM {
  if(cloud->empty()) {
    THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
    return {};
  }
  pcl::PointXYZ min_pt_;
  pcl::PointXYZ max_pt_;
  pcl::getMinMax3D(*cloud, min_pt_, max_pt_);
  auto min_pt = min_pt_.getVector3fMap();
  auto max_pt = max_pt_.getVector3fMap();
  std::cout << std::format("max {} {} min {} {}", max_pt.x(), max_pt.y(), min_pt.x(), min_pt.y());

  int cols = static_cast<int>(std::ceil((max_pt.x() - min_pt.x()) / resolution)) + 1;
  int rows = static_cast<int>(std::ceil((max_pt.y() - min_pt.y()) / resolution)) + 1;
  THIS_MESSAGE("DSM size: {}x{}, resolution: {}m", cols, rows, resolution);
  cv::Mat dsm(rows, cols, CV_64F, std::numeric_limits<double>::quiet_NaN());
  for(const auto& point_ : *cloud) {
    auto point = point_.getVector3fMap();
    int  col   = static_cast<int>((point.x() - min_pt.x()) / resolution);
    int  row   = static_cast<int>((point.y() - min_pt.y()) / resolution);
    if(col >= 0 && col < cols && row >= 0 && row < rows) {
      auto z_value = static_cast<double>(point.z());
      if(std::isnan(dsm.at<double>(row, col)) || z_value > dsm.at<double>(row, col)) {
        dsm.at<double>(row, col) = z_value;
      }
    }
  }
  cv::Mat valid_mask = cv::Mat(rows, cols, CV_8U);
  cv::Mat dsm_temp;
  dsm.convertTo(dsm_temp, CV_32F);
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      valid_mask.at<uchar>(i, j) = std::isnan(dsm.at<double>(i, j)) ? 0 : 255;
      if(std::isnan(dsm.at<double>(i, j))) {
        dsm_temp.at<float>(i, j) = 0.0F;
      }
    }
  }
  cv::Mat inpainted;
  cv::inpaint(dsm_temp, ~valid_mask, inpainted, 5, cv::INPAINT_NS);
  
  return DSM(inpainted, resolution, min_pt.x(), min_pt.y());
}

inline auto save_dsm_as_image(const cv::Mat& dsm, const std::string& output_path, bool normalize = true) -> bool {
  if(dsm.empty()) {
    THIS_MESSAGE("DSM is empty, cannot save");
    return false;
  }
  try {
    cv::Mat display_dsm;
    if(normalize) {
      cv::Mat mask = cv::Mat(dsm.size(), CV_8U, 255);
      double  min_val{};
      double  max_val{};
      cv::minMaxLoc(dsm, &min_val, &max_val, nullptr, nullptr, mask);
      cv::normalize(dsm, display_dsm, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
    } else {
      dsm.convertTo(display_dsm, CV_8U);
    }
    cv::imwrite(output_path, display_dsm);
    THIS_MESSAGE("DSM has been saved to: {}", output_path);
    return true;
  } catch(const cv::Exception& e) {
    report_error(e, "Failed to save DSM image");
    return false;
  }
}

} // namespace Ortho

#endif // ORTHO_DSM_HPP
