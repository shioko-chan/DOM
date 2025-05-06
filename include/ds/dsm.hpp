#ifndef ORTHO_DSM_HPP
#define ORTHO_DSM_HPP

#include <pcl/common/common.h>

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "tools/log.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

class DSM {
public:

  struct alignas(128) DSMUnit {
    Point3<double> point;
    cv::Mat        normal;
    cv::Vec3b&     tex_ref;
  };

  class Iterator {
  public:

    Iterator(DSM& dsm, int row, int col) : dsm_(dsm), row_(row), col_(col) {}

    auto operator++() -> Iterator& {
      ++col_;
      if(col_ >= dsm_.height_map_.cols) {
        col_ = 0;
        ++row_;
      }
      return *this;
    }

    auto operator++(int) -> Iterator {
      Iterator temp = *this;
      ++(*this);
      return temp;
    }

    auto operator*() -> DSMUnit {
      double    world_x = dsm_.min_x_ + (col_ * dsm_.resolution_);
      double    world_y = dsm_.min_y_ + (row_ * dsm_.resolution_);
      double    height  = dsm_.height_map_.at<double>(row_, col_);
      cv::Vec3f normal  = dsm_.normals_.at<cv::Vec3f>(row_, col_);
      return {
          .point   = Point3<double>{world_x, world_y, height},
          .normal  = (cv::Mat_<double>(3, 1) << normal[0], normal[1], normal[2]),
          .tex_ref = dsm_.texture_.at<cv::Vec3b>(row_, col_)};
    }

    auto operator==(const Iterator& other) const -> bool {
      return (&dsm_ == &other.dsm_) && (row_ == other.row_) && (col_ == other.col_);
    }

    auto operator!=(const Iterator& other) const -> bool { return !(*this == other); }

  private:

    DSM& dsm_;
    int  row_;
    int  col_;
  };

  DSM() noexcept = default;

  explicit DSM(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) noexcept :
      resolution_(resolution) {
    if(cloud->empty()) {
      THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
      return;
    }

    pcl::PointXYZ min_pt_;
    pcl::PointXYZ max_pt_;
    pcl::getMinMax3D(*cloud, min_pt_, max_pt_);
    auto min_pt = min_pt_.getVector3fMap();
    auto max_pt = max_pt_.getVector3fMap();

    min_x_ = min_pt.x();
    min_y_ = min_pt.y();

    std::cout << std::format("max {} {} min {} {}", max_pt.x(), max_pt.y(), min_x_, min_y_);

    int cols = static_cast<int>(std::ceil((max_pt.x() - min_x_) / resolution_)) + 1;
    int rows = static_cast<int>(std::ceil((max_pt.y() - min_y_) / resolution_)) + 1;

    THIS_MESSAGE("DSM size: {}x{}, resolution: {}m", cols, rows, resolution_);

    cv::Mat dsm(rows, cols, CV_64F, std::numeric_limits<double>::quiet_NaN());

    for(const auto& point_ : *cloud) {
      auto point = point_.getVector3fMap();
      int  col   = static_cast<int>((point.x() - min_x_) / resolution_);
      int  row   = static_cast<int>((point.y() - min_y_) / resolution_);

      if(col >= 0 && col < cols && row >= 0 && row < rows) {
        auto z_value = static_cast<double>(point.z());
        if(std::isnan(dsm.at<double>(row, col)) || z_value < dsm.at<double>(row, col)) {
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
          dsm_temp.at<float>(i, j) = 0.0;
        }
      }
    }
    cv::Mat inpainted;
    cv::inpaint(dsm_temp, ~valid_mask, inpainted, 5, cv::INPAINT_NS);

    height_map_ = std::move(inpainted);
    texture_    = cv::Mat::zeros(height_map_.rows, height_map_.cols, CV_8UC3);

    calculate_normals();
  }

  auto begin() -> Iterator { return {*this, 0, 0}; }

  auto end() -> Iterator { return {*this, height_map_.rows, 0}; }

  [[nodiscard]] auto empty() const noexcept -> bool { return height_map_.empty(); }

  [[nodiscard]] auto export_texture(const std::string& filepath) const {
    if(empty() || texture_.empty()) {
      THIS_MESSAGE("DSM或纹理为空，无法导出");
      return;
    }

    try {
      cv::imwrite(filepath, texture_);
      THIS_MESSAGE("已将DSM纹理导出至: {}", filepath);
      return;
    } catch(const cv::Exception& e) {
      THIS_MESSAGE("导出纹理失败: {}", e.what());
      return;
    }
  }

  [[nodiscard]] auto size() const -> int {
    if(empty()) {
      return 0;
    }
    return height_map_.rows * height_map_.cols;
  }

private:

  void calculate_normals() {
    int rows             = height_map_.rows;
    int cols             = height_map_.cols;
    normals_             = cv::Mat(rows, cols, CV_64FC3, cv::Vec3d(0.0, 0.0, 1.0));
    const double inv_res = 1.0 / resolution_;
    for(int row = 1; row < rows - 1; ++row) {
      for(int col = 1; col < cols - 1; ++col) {
        double d_x = (height_map_.at<double>(row - 1, col + 1) + (2 * height_map_.at<double>(row, col + 1))
                      + height_map_.at<double>(row + 1, col + 1) - height_map_.at<double>(row - 1, col - 1)
                      - (2 * height_map_.at<double>(row, col - 1)) - height_map_.at<double>(row + 1, col - 1))
                     * inv_res * 0.125;
        double d_y = (height_map_.at<double>(row + 1, col - 1) + (2 * height_map_.at<double>(row + 1, col))
                      + height_map_.at<double>(row + 1, col + 1) - height_map_.at<double>(row - 1, col - 1)
                      - (2 * height_map_.at<double>(row - 1, col)) - height_map_.at<double>(row - 1, col + 1))
                     * inv_res * 0.125;
        cv::Vec3d normal(-d_x, -d_y, 1.0);
        double    norm = std::sqrt((normal[0] * normal[0]) + (normal[1] * normal[1]) + (normal[2] * normal[2]));
        if(norm > 1e-8) {
          normal[0] /= norm;
          normal[1] /= norm;
          normal[2] /= norm;
        } else {
          normal = cv::Vec3d(0.0, 0.0, 1.0);
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

  cv::Mat height_map_;
  cv::Mat texture_;
  cv::Mat normals_;
  double  resolution_ = 0.0;
  double  min_x_      = 0.0;
  double  min_y_      = 0.0;
};

} // namespace Ortho

#endif // ORTHO_DSM_HPP
