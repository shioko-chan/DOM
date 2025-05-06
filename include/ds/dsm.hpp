#ifndef ORTHO_DSM_HPP
#define ORTHO_DSM_HPP

#include <pcl/common/common.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/report_error.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

class DSM {
public:

  DSM() noexcept = default;

  DSM(cv::Mat height_map, double resolution, double min_x, double min_y) noexcept :
      height_map_(std::move(height_map)), resolution_(resolution), min_x_(min_x), min_y_(min_y) {}

  [[nodiscard]] auto resolution() const noexcept -> double { return resolution_; }

  [[nodiscard]] auto min_x() const noexcept -> double { return min_x_; }

  [[nodiscard]] auto min_y() const noexcept -> double { return min_y_; }

  [[nodiscard]] auto max_x() const noexcept -> double { return min_x_ + ((cols() - 1) * resolution_); }

  [[nodiscard]] auto max_y() const noexcept -> double { return min_y_ + ((rows() - 1) * resolution_); }

  [[nodiscard]] auto rows() const noexcept -> int { return height_map_.rows; }

  [[nodiscard]] auto cols() const noexcept -> int { return height_map_.cols; }

  [[nodiscard]] auto empty() const noexcept -> bool { return height_map_.empty(); }

  [[nodiscard]] auto size() const noexcept -> cv::Size { return height_map_.size(); }

  [[nodiscard]] auto get_height(int row, int col) const noexcept -> double { return height_map_.at<double>(row, col); }

  [[nodiscard]] auto get_height_world(double world_x, double world_y) const noexcept -> double {
    int col = static_cast<int>((world_x - min_x_) / resolution_);
    int row = static_cast<int>((world_y - min_y_) / resolution_);

    if(row < 0 || row >= rows() || col < 0 || col >= cols()) {
      return std::numeric_limits<double>::quiet_NaN();
    }

    return get_height(row, col);
  }

  [[nodiscard]] auto world_to_grid(double world_x, double world_y) const noexcept -> Point<int> {
    int col = static_cast<int>((world_x - min_x_) / resolution_);
    int row = static_cast<int>((world_y - min_y_) / resolution_);
    return {col, row};
  }

  [[nodiscard]] auto grid_to_world(int row, int col) const noexcept -> Point<double> {
    double world_x = min_x_ + (col * resolution_);
    double world_y = min_y_ + (row * resolution_);
    return {world_x, world_y};
  }

  [[nodiscard]] auto grid_to_world_3d(int row, int col) const noexcept -> Point3<double> {
    double world_x = min_x_ + (col * resolution_);
    double world_y = min_y_ + (row * resolution_);
    double height  = get_height(row, col);
    return {world_x, world_y, height};
  }

  [[nodiscard]] auto height_map() const noexcept -> const cv::Mat& { return height_map_; }

  void save_as_image(const std::string& output_path, bool normalize = true) const {
    if(height_map_.empty()) {
      THIS_MESSAGE("DSM is empty, cannot save");
      return;
    }
    try {
      cv::Mat display_dsm;
      if(normalize) {
        cv::Mat mask = cv::Mat(height_map_.size(), CV_8U, 255);
        double  min_val{};
        double  max_val{};
        cv::minMaxLoc(height_map_, &min_val, &max_val, nullptr, nullptr, mask);
        cv::normalize(height_map_, display_dsm, 0, 255, cv::NORM_MINMAX, CV_8U, mask);
      } else {
        height_map_.convertTo(display_dsm, CV_8U);
      }
      cv::imwrite(output_path, display_dsm);
      THIS_MESSAGE("DSM has been saved to: {}", output_path);
      return;
    } catch(const cv::Exception& e) {
      report_error(e, "Failed to save DSM image");
      return;
    }
  }

private:

  cv::Mat height_map_;
  double  resolution_ = 0.0;
  double  min_x_      = 0.0;
  double  min_y_      = 0.0;
};

inline auto pointcloud_to_dsm(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) noexcept
    -> DSM {
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
        dsm_temp.at<double>(i, j) = 0.0F;
      }
    }
  }
  cv::Mat inpainted;
  cv::inpaint(dsm_temp, ~valid_mask, inpainted, 5, cv::INPAINT_NS);
  return {inpainted, resolution, min_pt.x(), min_pt.y()};
}

} // namespace Ortho

#endif // ORTHO_DSM_HPP
