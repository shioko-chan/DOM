#ifndef ORTHO_TRI_HPP
#define ORTHO_TRI_HPP

#include <array>
#include <cmath>
#include <mutex>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "algo/tracks.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"
#include "types/common_types.hpp"

namespace Ortho {

inline auto triangulation(
    const MatchPairs& match_img_pairs,
    ImgsData&         imgs_data,
    Progress&         progress
#ifdef ENABLE_VISUALIZE_OUTPUT
    ,
    const fs::path& pcd_output_dir
#endif
    ) noexcept -> TriResVec {
  if(imgs_data.empty() || match_img_pairs.empty()) {
    THIS_LOG_WARN("No input!");
    return {};
  }
  THIS_MESSAGE("Build tracks.");
  progress.reset(static_cast<int>(match_img_pairs.size()));
  TracksMaintainer tracks_maintainer;
  for(const auto& match_img_pair : match_img_pairs) {
    if(!match_img_pair.valid) {
      continue;
    }
    for(const auto& [lhs, rhs, score] : match_img_pair.matches) {
      tracks_maintainer.append_match(
          PointIdx{.img_idx = match_img_pair.first, .pnt_idx = lhs},
          PointIdx{.img_idx = match_img_pair.second, .pnt_idx = rhs},
          score);
    }
    progress.update();
  }
  std::vector<PointIdxs> pntidx_vecs = tracks_maintainer.get_tracks();

  THIS_MESSAGE("Start Triangulating.");
  TriResVec  all_res;
  std::mutex res_mtx;
#ifdef ENABLE_VISUALIZE_OUTPUT
  Point3s<double> points;
  std::mutex      points_mtx;
  run(
      pntidx_vecs.size(),
      [&all_res, &res_mtx, &pntidx_vecs, &imgs_data, &points, &points_mtx](int idx) noexcept {
#else
  run(
      pntidx_vecs.size(),
      [&all_res, &res_mtx, &pntidx_vecs, &imgs_data](int idx) noexcept {
#endif
        auto& pntidx_vec = pntidx_vecs[idx];
        auto  len        = static_cast<int64_t>(pntidx_vec.size());
        if(len <= 1) {
          return;
        }
        int64_t         rows     = 2 * len;
        int64_t         cols     = 3;
        Eigen::MatrixXd A_matrix = Eigen::MatrixXd::Zero(rows, cols);
        Eigen::VectorXd b_vector(rows);
        for(int64_t i = 0; i < len; ++i) {
          const auto& [img_idx, pnt_idx] = pntidx_vec[i];
          const auto&     img_data       = imgs_data[img_idx];
          const auto&     kpnt           = img_data.get_kpnts().get(pnt_idx);
          auto            kpnt_uni       = mat2point(imgs_data.M().inv() * kpnt);
          double          x_uni          = kpnt_uni.x;
          double          y_uni          = kpnt_uni.y;
          cv::Mat         R_mat          = img_data.R_w2c();
          Eigen::Matrix3d R_eigen;
          cv::cv2eigen(R_mat, R_eigen);
          cv::Mat t_mat                        = img_data.t_w2c();
          double  t_x                          = t_mat.at<double>(0, 0);
          double  t_y                          = t_mat.at<double>(1, 0);
          double  t_z                          = t_mat.at<double>(2, 0);
          A_matrix.block<1, 3>(i * 2, 0)       = R_eigen.row(0) - x_uni * R_eigen.row(2);
          A_matrix.block<1, 3>((i * 2) + 1, 0) = R_eigen.row(1) - y_uni * R_eigen.row(2);
          b_vector(i * 2)                      = x_uni * t_z - t_x;
          b_vector((i * 2) + 1)                = y_uni * t_z - t_y;
        }
        Eigen::VectorXd x_vector = A_matrix.colPivHouseholderQr().solve(b_vector);
        THIS_ASSERTION_SHOULD_TRUE(x_vector.array().isFinite().all());
        if(!x_vector.array().isFinite().all()) {
          return;
        }

#ifdef ENABLE_VISUALIZE_OUTPUT
        {
          std::lock_guard<std::mutex> lock(points_mtx);
          points.emplace_back(x_vector(0), x_vector(1), x_vector(2));
        }
#endif
        {
          std::lock_guard lock{res_mtx};
          all_res.emplace_back(std::array<double, 3>{x_vector(0), x_vector(1), x_vector(2)}, std::move(pntidx_vec));
        }
      },
      progress);
#ifdef ENABLE_VISUALIZE_OUTPUT
  export_pcd(pcd_output_dir / "tri.pcd", points);
#endif
  THIS_MESSAGE("Triangulation Finished");
  return all_res;
}
} // namespace Ortho

#endif
