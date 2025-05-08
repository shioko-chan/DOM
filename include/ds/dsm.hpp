#ifndef SKYMERGE_DSM_HPP
#define SKYMERGE_DSM_HPP

#include <pcl/common/common.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/log.hpp"
#include "tools/report_error.hpp"
#include "types/cv_alias.hpp"

namespace SkyMerge {

class DSM {
public:

  DSM() noexcept = default;

  explicit DSM(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) noexcept :
      resolution_(resolution) {
    if(cloud->empty()) {
      THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
      return;
    }
    height_map = calculate_dsm(cloud);
  }

  [[nodiscard]] auto operator[](int idx) const noexcept -> Point3<double> {
    if(idx < 0 || idx >= height_map.rows * height_map.cols) {
      report_error("DSM Index out of range");
    }
    int row = idx / height_map.cols;
    int col = idx % height_map.cols;
    return {min_x + (col * resolution_), min_y + (row * resolution_), height_map.at<double>(row, col)};
  }

  [[nodiscard]] auto empty() const noexcept -> bool { return height_map.empty(); }

  [[nodiscard]] auto size() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map.rows * height_map.cols;
  }

  [[nodiscard]] auto cols() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map.cols;
  }

  [[nodiscard]] auto rows() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map.rows;
  }

private:

  auto calculate_dsm(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) noexcept -> cv::Mat {
    pcl::PointXYZ min_pt_;
    pcl::PointXYZ max_pt_;
    pcl::getMinMax3D(*cloud, min_pt_, max_pt_);
    auto min_pt = min_pt_.getVector3fMap();
    auto max_pt = max_pt_.getVector3fMap();
    min_x       = min_pt.x();
    min_y       = min_pt.y();
    int cols    = static_cast<int>(std::ceil((max_pt.x() - min_x) / resolution_)) + 1;
    int rows    = static_cast<int>(std::ceil((max_pt.y() - min_y) / resolution_)) + 1;
    THIS_MESSAGE("DSM size: {}x{}, resolution: {}m", cols, rows, resolution_);
    cv::Mat dsm(rows, cols, CV_64F, std::numeric_limits<double>::quiet_NaN());
    for(const auto& point_ : *cloud) {
      auto point = point_.getVector3fMap();
      int  col   = static_cast<int>((point.x() - min_x) / resolution_);
      int  row   = static_cast<int>((point.y() - min_y) / resolution_);
      if(col >= 0 && col < cols && row >= 0 && row < rows) {
        auto z_value = static_cast<double>(point.z());
        if(std::isnan(dsm.at<double>(row, col)) || z_value < dsm.at<double>(row, col)) {
          dsm.at<double>(row, col) = z_value;
        }
      }
    }

    cv::Mat dsm_temp;
    dsm.convertTo(dsm_temp, CV_32F);
    cv::Mat invalid_mask;
    cv::compare(dsm, dsm, invalid_mask, cv::CMP_NE);
    dsm_temp.setTo(0, invalid_mask);
    cv::Mat inpainted;
    cv::inpaint(dsm_temp, invalid_mask, inpainted, 5, cv::INPAINT_NS);
    inpainted.convertTo(inpainted, CV_64F);

    cv::normalize(inpainted, inpainted, 0.0, 1.0, cv::NORM_MINMAX);
    inpainted.convertTo(inpainted, CV_8UC1, 255.0);
    cv::imshow("in", inpainted);
    cv::waitKey();
    return inpainted;
  }

  cv::Mat height_map;
  cv::Mat normals_;
  double  resolution_ = 0.0;
  double  min_x       = 0.0;
  double  min_y       = 0.0;
};

} // namespace SkyMerge

#endif // SKYMERGE_DSM_HPP
