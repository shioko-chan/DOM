#ifndef ORTHO_TRI_HPP
#define ORTHO_TRI_HPP

#include <array>
#include <fstream>
#include <mutex>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <ostream>
#include <vector>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "algo/cost.hpp"
#include "algo/tracks.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace Ortho {

#ifdef ENABLE_VISUALIZE_OUTPUT
inline void export_pcd(const fs::path& path, const Point3s<double>& points) {
  std::ofstream file(path);
  file << "# .PCD v7 - Point Cloud Data\n";
  file << "VERSION .7\n";
  file << "FIELDS x y z\n";
  file << "SIZE 4 4 4\n";
  file << "TYPE F F F\n";
  file << "COUNT 1 1 1\n";
  file << "WIDTH " << points.size() << "\n";
  file << "HEIGHT 1\n";
  file << "VIEWPOINT 0 0 0 1 0 0 0\n";
  file << "POINTS " << points.size() << "\n";
  file << "DATA ascii\n";
  for(const auto& point : points) {
    file << std::fixed << std::setprecision(6) << point.y << " " << point.x << " " << -point.z << "\n";
  }
  file.close();
}
#endif
struct alignas(64) TriRes {
  std::array<double, 3> pnt3d;
  PointIdxs             pnt2d_idx_vec;
};

inline auto triangulation(const MatchPairs& match_img_pairs, ImgsData& imgs_data, Progress& progress) noexcept
    -> std::vector<TriRes> {
  THIS_MESSAGE("Build tracks");
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
  std::vector<TriRes>    all_res;
  std::mutex             mtx;
#ifdef ENABLE_VISUALIZE_OUTPUT
  Point3s<double> points1;
  Point3s<double> points2;
  std::mutex      mtx1;
  std::mutex      mtx2;
  run(
      pntidx_vecs.size(),
      [&all_res, &pntidx_vecs, &imgs_data, &mtx, &points1, &points2, &mtx1, &mtx2](int idx) noexcept {
#else
  run(
      pntidx_vecs.size(),
      [&all_res, &pntidx_vecs, &imgs_data, &mtx](int idx) noexcept {
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
          auto            kpnt_uni       = mat2point(img_data.M().inv() * kpnt);
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
        std::array<double, 3> world_point{x_vector(0), x_vector(1), x_vector(2)};
#ifdef ENABLE_VISUALIZE_OUTPUT
        {
          std::lock_guard<std::mutex> lock(mtx1);
          points1.emplace_back(world_point[0], world_point[1], world_point[2]);
        }
#endif
        ceres::Problem problem;
        add_parameter_block(problem, world_point);
        for(const auto& pntidx : pntidx_vec) {
          const auto& img_data = imgs_data[pntidx.img_idx];
          try {
            problem.AddResidualBlock(
                SimpReprojectionError::create(
                    img_data.get_kpnts().get(pntidx.pnt_idx),
                    img_data.Q_w2c_array_raw(),
                    img_data.camera_array_raw(),
                    img_data.t_w2c_array_raw()),
                new ceres::HuberLoss(1.0),
                world_point.data());
          } catch(const std::exception& e) {
            report_error(e, "Bad allocation");
          }
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;

        options.check_gradients                   = false;
        options.gradient_check_relative_precision = 1e-2;

        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations           = 1000;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if(summary.IsSolutionUsable()) {
          std::lock_guard lock{mtx};
          all_res
              .emplace_back(std::array<double, 3>{world_point[0], world_point[1], world_point[2]}, std::move(pntidx_vec));
#ifdef ENABLE_VISUALIZE_OUTPUT
          {
            std::lock_guard<std::mutex> lock(mtx2);
            points2.emplace_back(world_point[0], world_point[1], world_point[2]);
          }
#endif
        } else {
          std::cout << 12324982743 << '\n';
          THIS_LOG_WARN("Triangulation solution is unusable. The report is as below: \n{}", summary.FullReport());
        }
      },
      progress);

#ifdef ENABLE_VISUALIZE_OUTPUT
  export_pcd("tri1.pcd", points1);
#endif

#ifdef ENABLE_VISUALIZE_OUTPUT
  export_pcd("tri2.pcd", points2);
#endif

  return all_res;
}
} // namespace Ortho

#endif
