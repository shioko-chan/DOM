#ifndef ORTHO_DSM_HPP
#define ORTHO_DSM_HPP

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <queue>

#include "tools/utility.hpp"
#include "tools/report_error.hpp"
#include "types/common_types.hpp"

namespace Ortho {

inline void fill_dsm_holes(cv::Mat& dsm, const cv::Mat& has_points, int max_search_radius = 10, double power = 2.0) {
    int rows = dsm.rows;
    int cols = dsm.cols;
    
    std::vector<std::pair<int, int>> holes;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (has_points.at<uchar>(i, j) == 0) {
                holes.push_back({i, j});
            }
        }
    }
    
    THIS_MESSAGE("Filling {} empty cells with IDW interpolation", holes.size());
    
    const int dx[8] = {0, 1, 0, -1, 1, 1, -1, -1};
    const int dy[8] = {-1, 0, 1, 0, -1, 1, 1, -1};
    
    for (const auto& [y, x] : holes) {
        double sum_weights = 0.0;
        double sum_weighted_values = 0.0;
        bool found_neighbors = false;
        
        for (int radius = 1; radius <= max_search_radius && !found_neighbors; radius++) {
            for (int i = -radius; i <= radius; i++) {
                for (int j = -radius; j <= radius; j++) {
                    if (std::abs(i) < radius && std::abs(j) < radius) continue;
                    
                    int ni = y + i;
                    int nj = x + j;
                    
                    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
                        if (has_points.at<uchar>(ni, nj) == 1) {
                            double distance = std::sqrt(i*i + j*j);
                            double weight = 1.0 / std::pow(distance, power);
                            
                            sum_weights += weight;
                            sum_weighted_values += weight * dsm.at<double>(ni, nj);
                            found_neighbors = true;
                        }
                    }
                }
            }
        }
        
        if (found_neighbors) {
            dsm.at<double>(y, x) = sum_weighted_values / sum_weights;
        } else {
            dsm.at<double>(y, x) = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

inline cv::Mat pointcloud_to_dsm(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double resolution = 0.5) {
    if (cloud->empty()) {
        THIS_MESSAGE("Empty point cloud data, cannot generate DSM.");
        return cv::Mat();
    }
    
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    
    for (const auto& point : *cloud) {
        min_x = std::min(min_x, point.x);
        min_y = std::min(min_y, point.y);
        max_x = std::max(max_x, point.x);
        max_y = std::max(max_y, point.y);
    }
    
    int cols = static_cast<int>(std::ceil((max_x - min_x) / resolution)) + 1;
    int rows = static_cast<int>(std::ceil((max_y - min_y) / resolution)) + 1;
    
    THIS_MESSAGE("DSM size: {}x{}, resolution: {}m", cols, rows, resolution);
    
    cv::Mat dsm = cv::Mat(rows, cols, CV_64F, -std::numeric_limits<double>::infinity());
    cv::Mat has_points = cv::Mat::zeros(rows, cols, CV_8U);
    
    for (const auto& point : *cloud) {
        int col = static_cast<int>((point.x - min_x) / resolution);
        int row = static_cast<int>((point.y - min_y) / resolution);
        
        if (col >= 0 && col < cols && row >= 0 && row < rows) {
            double z_value = static_cast<double>(point.z);
            if (z_value > dsm.at<double>(row, col)) {
                dsm.at<double>(row, col) = z_value;
            }
            has_points.at<uchar>(row, col) = 1;
        }
    }
    
    fill_dsm_holes(dsm, has_points);
    
    return dsm;
}

inline cv::Mat tri_res_vec_to_dsm(const TriResVec& tri_res_vec, double resolution = 0.5) {
    auto cloud = tri_res_vec2point_cloud(tri_res_vec);
    return pointcloud_to_dsm(cloud, resolution);
}

inline bool save_dsm_as_image(const cv::Mat& dsm, const std::string& output_path, bool normalize = true) {
    if (dsm.empty()) {
        THIS_MESSAGE("DSM is empty, cannot save");
        return false;
    }
    cv::Mat display_dsm;
    if (normalize) {
        cv::Mat mask = cv::Mat(dsm.size(), CV_8U);
        for (int i = 0; i < dsm.rows; i++) {
            for (int j = 0; j < dsm.cols; j++) {
                mask.at<uchar>(i, j) = std::isnan(dsm.at<double>(i, j)) ? 0 : 255;
            }
        }
        double min_val, max_val;
        cv::Point min_loc, max_loc;
        cv::minMaxLoc(dsm, &min_val, &max_val, &min_loc, &max_loc, mask);
        
        dsm.convertTo(display_dsm, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
        
        for (int i = 0; i < dsm.rows; i++) {
            for (int j = 0; j < dsm.cols; j++) {
                if (!mask.at<uchar>(i, j)) {
                    display_dsm.at<uchar>(i, j) = 0;
                }
            }
        }
    } else {
        display_dsm = dsm.clone();
    }
    try {
        cv::imwrite(output_path, display_dsm);
        THIS_MESSAGE("DSM has been saved to: {}", output_path);
        return true;
    } catch (const cv::Exception& e) {
        report_error(e, "Failed to save DSM image");
        return false;
    }
}

} // namespace Ortho

#endif // ORTHO_DSM_HPP
