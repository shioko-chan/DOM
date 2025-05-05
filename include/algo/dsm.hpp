#ifndef ORTHO_DSM_HPP
#define ORTHO_DSM_HPP

#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/report_error.hpp"
#include "tools/utility.hpp"
#include "types/common_types.hpp"

namespace Ortho {

inline auto pointcloud_to_dsm(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) -> cv::Mat {
  if(cloud->empty()) {
    THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
    return {};
  }
  pcl::PointXYZ min_pt;
  pcl::PointXYZ max_pt;
  pcl::getMinMax3D(*cloud, min_pt, max_pt);
  int cols = static_cast<int>(std::ceil((max_pt.x - min_pt.x) / resolution)) + 1;
  int rows = static_cast<int>(std::ceil((max_pt.y - min_pt.y) / resolution)) + 1;
  THIS_MESSAGE("DSM size: {}x{}, resolution: {}m", cols, rows, resolution);
  cv::Mat dsm(rows, cols, CV_64F, std::numeric_limits<double>::quiet_NaN());
  for(const auto& point : *cloud) {
    int col = static_cast<int>((point.x - min_pt.x) / resolution);
    int row = static_cast<int>((point.y - min_pt.y) / resolution);
    if(col >= 0 && col < cols && row >= 0 && row < rows) {
      auto z_value = static_cast<double>(point.z);
      if(std::isnan(dsm.at<double>(row, col)) || z_value > dsm.at<double>(row, col)) {
        dsm.at<double>(row, col) = z_value;
      }
    }
  }
  cv::Mat valid_mask = cv::Mat(rows, cols, CV_8U);
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      valid_mask.at<uchar>(i, j) = std::isnan(dsm.at<double>(i, j)) ? 0 : 255;
    }
  }
  cv::Mat dsm_temp;
  dsm.convertTo(dsm_temp, CV_64F);
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      if(std::isnan(dsm.at<double>(i, j))) {
        dsm_temp.at<double>(i, j) = 0.0;
      }
    }
  }
  cv::Mat inpainted;
  cv::inpaint(dsm_temp, ~valid_mask, inpainted, 5, cv::INPAINT_NS);
  return inpainted;
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
