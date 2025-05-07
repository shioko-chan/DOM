#ifndef SKYMERGE_DSM_HPP
#define SKYMERGE_DSM_HPP

#include <pcl/common/common.h>

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

  struct alignas(128) DSMUnit {
    Point3<double> point;
    cv::Mat        normal;
  };

  DSM() noexcept = default;

  explicit DSM(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) noexcept :
      resolution_(resolution) {
    if(cloud->empty()) {
      THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
      return;
    }
    calculate_dsm(cloud);
    calculate_normals();
  }

  auto operator[](int idx) noexcept -> DSMUnit {
    if(idx < 0 || idx >= height_map.rows * height_map.cols) {
      report_error("DSM Index out of range");
    }
    int       row    = idx / height_map.cols;
    int       col    = idx % height_map.cols;
    cv::Vec3f normal = normals_.at<cv::Vec3f>(row, col);
    return {
        .point =
            Point3<double>{min_x + (col * resolution_), min_y + (row * resolution_), height_map.at<double>(row, col)},
        .normal = (cv::Mat_<double>(3, 1) << normal[0], normal[1], normal[2]),
    };
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

  void calculate_dsm(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud) noexcept {
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
    cv::Mat valid_mask = cv::Mat(rows, cols, CV_8U, cv::Scalar(0));
    cv::Mat dsm_temp;
    dsm.convertTo(dsm_temp, CV_32F);
    for(int i = 0; i < rows; i++) {
      for(int j = 0; j < cols; j++) {
        if(std::isnan(dsm.at<double>(i, j))) {
          valid_mask.at<uchar>(i, j) = 255;
          dsm_temp.at<float>(i, j)   = 0;
        }
      }
    }
    cv::Mat inpainted;
    cv::inpaint(dsm_temp, valid_mask, inpainted, 5, cv::INPAINT_NS);
    inpainted.convertTo(height_map, CV_64F);
    cv::normalize(inpainted, inpainted, 0.0, 1.0, cv::NORM_MINMAX);
    inpainted.convertTo(inpainted, CV_8UC1, 255.0);
    // cv::imshow("in", inpainted);
    // cv::waitKey();
  }

  void calculate_normals() noexcept {
    int rows             = height_map.rows;
    int cols             = height_map.cols;
    normals_             = cv::Mat(rows, cols, CV_64FC3, cv::Vec3d(0.0, 0.0, 1.0));
    const double inv_res = 1.0 / resolution_;
    // run(rows * cols, [this](int idx) noexcept {
    //   int row = idx / cols;
    //   int col = idx % cols;
    //   if(row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
    //     normals_.at<cv::Vec3f>(row, col) = cv::Vec3f(0.0, 0.0, -1.0);
    //   }
    // });
    for(int row = 1; row < rows - 1; ++row) {
      for(int col = 1; col < cols - 1; ++col) {
        double d_x = (height_map.at<double>(row - 1, col + 1) + (2 * height_map.at<double>(row, col + 1))
                      + height_map.at<double>(row + 1, col + 1) - height_map.at<double>(row - 1, col - 1)
                      - (2 * height_map.at<double>(row, col - 1)) - height_map.at<double>(row + 1, col - 1))
                     * inv_res * 0.125;
        double d_y = (height_map.at<double>(row + 1, col - 1) + (2 * height_map.at<double>(row + 1, col))
                      + height_map.at<double>(row + 1, col + 1) - height_map.at<double>(row - 1, col - 1)
                      - (2 * height_map.at<double>(row - 1, col)) - height_map.at<double>(row - 1, col + 1))
                     * inv_res * 0.125;
        cv::Vec3d normal(-d_x, -d_y, 1.0);
        double    norm = std::sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) + (normal[2] * normal[2]));
        if(norm > 1e-8) {
          normal[0] /= norm;
          normal[1] /= norm;
          normal[2] /= norm;
        } else {
          normal = cv::Vec3d(0.0, 0.0, -1.0);
        }

        normals_.at<cv::Vec3d>(row, col) = normal;
      }
    }

    for(int row = 0; row < rows; ++row) {
      normals_.at<cv::Vec3f>(row, 0) =
          (row > 0 && row < rows - 1) ? normals_.at<cv::Vec3f>(row, 1) : normals_.at<cv::Vec3f>(1, 1);
      normals_.at<cv::Vec3f>(row, cols - 1) =
          (row > 0 && row < rows - 1) ? normals_.at<cv::Vec3f>(row, cols - 2) : normals_.at<cv::Vec3f>(1, cols - 2);
    }

    for(int col = 0; col < cols; ++col) {
      normals_.at<cv::Vec3f>(0, col) =
          (col > 0 && col < cols - 1) ? normals_.at<cv::Vec3f>(1, col) : normals_.at<cv::Vec3f>(1, 1);
      normals_.at<cv::Vec3f>(rows - 1, col) = (col > 0 && col < cols - 1) ? normals_.at<cv::Vec3f>(rows - 2, col)
                                                                          : normals_.at<cv::Vec3f>(rows - 2, cols - 2);
    }
  }

  cv::Mat height_map;
  cv::Mat normals_;
  double  resolution_ = 0.0;
  double  min_x       = 0.0;
  double  min_y       = 0.0;
};

} // namespace SkyMerge

#endif // SKYMERGE_DSM_HPP
