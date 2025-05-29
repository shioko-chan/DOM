#ifndef SKYMERGE_TRI_HPP
#define SKYMERGE_TRI_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <map>
#include <mutex>
#include <vector>

#include <Eigen/Dense>

#include <Eigen/src/Core/util/Constants.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

namespace SkyMerge {

static const struct alignas(32) {
  int    min_track_length            = 2;
  double min_depth                   = 0.1;
  double min_baseline_length         = 0.4;
  double min_triangulation_angle_rad = 0.2;
} robust_triangulation_config;

inline auto is_point_in_front_of_camera(
    const std::array<double, 3>& point_3d,
    const double*                axisangle,
    const double*                translation,
    double                       min_depth = 0.01) -> bool {
  auto point_cam = world2camera(axisangle, translation, point_3d.data());
  return point_cam.z() > min_depth;
}

inline auto compute_baseline_length(const ImgData& lhs_img, const ImgData& rhs_img) -> double {
  cv::Mat camera_center1 = lhs_img.t_c2w();
  cv::Mat camera_center2 = rhs_img.t_c2w();
  return std::hypot(
      camera_center1.at<double>(0, 0) - camera_center2.at<double>(0, 0),
      camera_center1.at<double>(1, 0) - camera_center2.at<double>(1, 0),
      camera_center1.at<double>(2, 0) - camera_center2.at<double>(2, 0));
}

inline auto compute_baseline_angle(
    const ImgData&  img_data1,
    const ImgData&  img_data2,
    const PointIdx& obs1,
    const PointIdx& obs2,
    ImgsData&       imgs_data) -> double {
  const auto& kpnt1      = img_data1.get_kpnts()[obs1.pnt_idx];
  const auto& kpnt2      = img_data2.get_kpnts()[obs2.pnt_idx];
  auto        kpnt1_norm = mat2point(imgs_data.M().inv() * kpnt1);
  auto        kpnt2_norm = mat2point(imgs_data.M().inv() * kpnt2);
  cv::Mat     ray1       = (cv::Mat_<double>(3, 1) << kpnt1_norm.x, kpnt1_norm.y, 1.0);
  cv::Mat     ray2       = (cv::Mat_<double>(3, 1) << kpnt2_norm.x, kpnt2_norm.y, 1.0);
  cv::Mat     world_ray1 = img_data1.R_c2w() * ray1;
  cv::Mat     world_ray2 = img_data2.R_c2w() * ray2;
  cv::normalize(world_ray1, world_ray1);
  cv::normalize(world_ray2, world_ray2);
  double dot_product = world_ray1.dot(world_ray2);
  dot_product        = std::max(-1.0, std::min(1.0, dot_product));
  return std::acos(std::abs(dot_product));
}

inline auto evaluate_observation_pair_quality(const PointIdx& obs1, const PointIdx& obs2, ImgsData& imgs_data)
    -> double {
  auto&  lhs_img        = imgs_data[obs1.img_idx];
  auto&  rhs_img        = imgs_data[obs2.img_idx];
  double baseline       = compute_baseline_length(lhs_img, rhs_img);
  double angle          = compute_baseline_angle(lhs_img, rhs_img, obs1, obs2, imgs_data);
  double baseline_score = std::min(1.0, baseline / robust_triangulation_config.min_baseline_length);
  double angle_score    = std::min(1.0, angle / robust_triangulation_config.min_triangulation_angle_rad);
  return (0.6 * baseline_score) + (0.4 * angle_score);
}

inline auto
simple_triangulation_with_validation(const PointIdxs& track, ImgsData& imgs_data, std::array<double, 3>& best_point_3d)
    -> bool {
  int64_t               rows     = 2 * static_cast<int64_t>(track.size());
  int64_t               cols     = 3;
  Eigen::MatrixXd       A_matrix = Eigen::MatrixXd::Zero(rows, cols);
  Eigen::VectorXd       b_vector(rows);
  std::vector<ImgData*> img_data_ptrs;
  for(int64_t i = 0; i < static_cast<int64_t>(track.size()); ++i) {
    const auto& [img_idx, pnt_idx] = track[i];
    auto&       img_data           = imgs_data[img_idx];
    const auto& kpnt               = img_data.get_kpnts()[pnt_idx];
    img_data_ptrs.push_back(&img_data);
    auto            kpnt_uni = mat2point(imgs_data.M().inv() * kpnt);
    double          x_uni    = kpnt_uni.x;
    double          y_uni    = kpnt_uni.y;
    cv::Mat         R_mat    = img_data.R_w2c();
    cv::Mat         t_mat    = img_data.t_w2c();
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R_mat, R_eigen);
    double t_x                           = t_mat.at<double>(0, 0);
    double t_y                           = t_mat.at<double>(1, 0);
    double t_z                           = t_mat.at<double>(2, 0);
    A_matrix.block<1, 3>(i * 2, 0)       = R_eigen.row(0) - x_uni * R_eigen.row(2);
    A_matrix.block<1, 3>((i * 2) + 1, 0) = R_eigen.row(1) - y_uni * R_eigen.row(2);
    b_vector(i * 2)                      = x_uni * t_z - t_x;
    b_vector((i * 2) + 1)                = y_uni * t_z - t_y;
  }
  Eigen::VectorXd x_vector = A_matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b_vector);
  if(!x_vector.array().isFinite().all()) {
    return false;
  }
  std::array<double, 3> candidate_point = {x_vector(0), x_vector(1), x_vector(2)};
  for(auto& img_data : img_data_ptrs) {
    if(!is_point_in_front_of_camera(
           candidate_point,
           img_data->A_w2c_array_raw().data(),
           img_data->t_w2c_array_raw().data(),
           robust_triangulation_config.min_depth)) {
      return false;
    }
  }
  best_point_3d = candidate_point;
  return true;
}

inline auto
select_best_observations_and_triangulate(const PointIdxs& track, ImgsData& imgs_data, std::array<double, 3>& best_point_3d)
    -> bool {
  if(track.size() < static_cast<size_t>(robust_triangulation_config.min_track_length)) {
    return false;
  }
  if(track.size() == 2) {
    return simple_triangulation_with_validation(track, imgs_data, best_point_3d);
  }
  double                best_quality = 0.0;
  std::vector<PointIdx> best_pair;
  for(size_t i = 0; i < track.size(); ++i) {
    for(size_t j = i + 1; j < track.size(); ++j) {
      double quality = evaluate_observation_pair_quality(track[i], track[j], imgs_data);
      if(quality > best_quality) {
        best_quality = quality;
        best_pair    = {track[i], track[j]};
      }
    }
  }
  if(best_quality < 0.1) {
    return false;
  }
  PointIdxs best_track = {best_pair[0], best_pair[1]};
  return simple_triangulation_with_validation(best_track, imgs_data, best_point_3d);
}

inline auto
weighted_triangulation_with_pair_quality(const PointIdxs& track, ImgsData& imgs_data, std::array<double, 3>& best_point_3d)
    -> bool {
  if(track.size() < static_cast<size_t>(robust_triangulation_config.min_track_length)) {
    return false;
  }

  if(track.size() == 2) {
    return simple_triangulation_with_validation(track, imgs_data, best_point_3d);
  }

  std::map<std::pair<size_t, size_t>, double> pair_weights;
  for(size_t i = 0; i < track.size(); ++i) {
    for(size_t j = i + 1; j < track.size(); ++j) {
      double quality = evaluate_observation_pair_quality(track[i], track[j], imgs_data);
      if(quality > 0.01) {
        pair_weights[{i, j}] = quality;
      }
    }
  }

  if(pair_weights.empty()) {
    return false;
  }

  // 第二步：将pair-wise权重分摊到各个观测点
  std::vector<double> observation_weights(track.size(), 0.0);
  for(const auto& [pair, weight] : pair_weights) {
    observation_weights[pair.first] += weight;
    observation_weights[pair.second] += weight;
  }

  // 点对数量归一化：避免参与更多对的点获得过高权重
  std::vector<int> pair_counts(track.size(), 0);
  for(const auto& [pair, weight] : pair_weights) {
    pair_counts[pair.first]++;
    pair_counts[pair.second]++;
  }

  for(size_t i = 0; i < track.size(); ++i) {
    if(pair_counts[i] > 0) {
      observation_weights[i] /= pair_counts[i]; // 平均每对贡献
    }
  }

  // 第三步：归一化权重
  double total_weight = 0.0;
  for(double w : observation_weights) {
    total_weight += w;
  }

  if(total_weight < 1e-8) {
    return false;
  }

  for(double& w : observation_weights) {
    w /= total_weight;
  }

  // 调试信息：显示权重分摊效果（仅前几个轨迹）
  static std::atomic<int> debug_count{0};
  if(debug_count.load() < 3) {
    std::string weight_info = "Track " + std::to_string(debug_count.load()) + " final_weights: ";
    std::string count_info  = "pair_counts: ";
    for(size_t i = 0; i < observation_weights.size(); ++i) {
      weight_info += std::to_string(i) + ":" + std::to_string(observation_weights[i]) + " ";
      count_info += std::to_string(i) + ":" + std::to_string(pair_counts[i]) + " ";
    }
    THIS_LOG_INFO("{}", weight_info);
    THIS_LOG_INFO("{} (total_pairs: {})", count_info, pair_weights.size());
    debug_count++;
  }

  int64_t         total_rows = 2 * static_cast<int64_t>(track.size());
  Eigen::MatrixXd A_matrix   = Eigen::MatrixXd::Zero(total_rows, 4);

  std::vector<ImgData*> img_data_ptrs;

  for(int64_t i = 0; i < static_cast<int64_t>(track.size()); ++i) {
    const auto& [img_idx, pnt_idx] = track[i];
    auto&       img_data           = imgs_data[img_idx];
    const auto& kpnt               = img_data.get_kpnts()[pnt_idx];

    img_data_ptrs.push_back(&img_data);

    auto   kpnt_uni = mat2point(imgs_data.M().inv() * kpnt);
    double x_uni    = kpnt_uni.x;
    double y_uni    = kpnt_uni.y;

    cv::Mat R_mat = img_data.R_w2c();
    cv::Mat t_mat = img_data.t_w2c();

    Eigen::Matrix<double, 1, 4> P1;
    Eigen::Matrix<double, 1, 4> P2;
    Eigen::Matrix<double, 1, 4> P3;
    P1 << R_mat.at<double>(0, 0), R_mat.at<double>(0, 1), R_mat.at<double>(0, 2), t_mat.at<double>(0, 0);
    P2 << R_mat.at<double>(1, 0), R_mat.at<double>(1, 1), R_mat.at<double>(1, 2), t_mat.at<double>(1, 0);
    P3 << R_mat.at<double>(2, 0), R_mat.at<double>(2, 1), R_mat.at<double>(2, 2), t_mat.at<double>(2, 0);

    Eigen::Matrix<double, 1, 4> constraint1 = x_uni * P3 - P1;
    Eigen::Matrix<double, 1, 4> constraint2 = y_uni * P3 - P2;

    double weight_sqrt        = std::sqrt(observation_weights[i]);
    A_matrix.row(i * 2)       = weight_sqrt * constraint1;
    A_matrix.row((i * 2) + 1) = weight_sqrt * constraint2;
  }

  Eigen::Matrix4d AtA = A_matrix.transpose() * A_matrix;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eigen_solver(AtA);
  if(eigen_solver.info() != Eigen::Success) {
    return false;
  }

  Eigen::Vector4d solution = eigen_solver.eigenvectors().col(0);

  if(!solution.array().isFinite().all() || std::abs(solution(3)) < 1e-8) {
    return false;
  }

  std::array<double, 3> candidate_point =
      {solution(0) / solution(3), solution(1) / solution(3), solution(2) / solution(3)};

  for(auto& img_data : img_data_ptrs) {
    if(!is_point_in_front_of_camera(
           candidate_point,
           img_data->A_w2c_array_raw().data(),
           img_data->t_w2c_array_raw().data(),
           robust_triangulation_config.min_depth)) {
      return false;
    }
  }

  best_point_3d = candidate_point;
  return true;
}

inline auto triangulation(
    Tracks&   tracks,
    ImgsData& imgs_data,
    Progress& progress
#ifdef ENABLE_VISUALIZE_OUTPUT
    ,
    const std::filesystem::path& pcd_output_dir
#endif
    ) noexcept -> TrackPointVec {
  if(imgs_data.empty()) {
    THIS_LOG_WARN("No input!");
    return {};
  }
  THIS_MESSAGE("Start Weighted Triangulation: Using pair-wise geometric quality for observation weighting.");
  TrackPointVec all_res;
  std::mutex    res_mtx;

#ifdef ENABLE_VISUALIZE_OUTPUT
  Point3s<double> points;
  std::mutex      points_mtx;
#endif
  THIS_LOG_INFO("Processing {} tracks with weighted triangulation...", tracks.size());
  run(
      tracks.size(),
      [&](int idx) noexcept {
        auto&                 track = tracks[idx];
        std::array<double, 3> point_3d{};
        if(weighted_triangulation_with_pair_quality(track, imgs_data, point_3d)) {
#ifdef ENABLE_VISUALIZE_OUTPUT
          {
            std::lock_guard<std::mutex> lock(points_mtx);
            points.emplace_back(point_3d[0], point_3d[1], point_3d[2]);
          }
#endif
          {
            std::lock_guard lock{res_mtx};
            all_res.emplace_back(TrackPoint{.pnt3d = point_3d, .pnt2d_idx_vec = track});
          }
        }
      },
      progress);

#ifdef ENABLE_VISUALIZE_OUTPUT
  export_pcd(pcd_output_dir / "weighted_tri.pcd", points);
#endif

  THIS_LOG_INFO(
      "[Weighted Triangulation] Generated {} 3D points from {} tracks (success rate: {:.1f}%)",
      all_res.size(),
      tracks.size(),
      tracks.empty() ? 0.0 : (100.0 * all_res.size() / tracks.size()));

  THIS_MESSAGE("Weighted Triangulation Finished");
  return all_res;
}

} // namespace SkyMerge

#endif
