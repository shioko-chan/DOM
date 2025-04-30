#ifndef ORTHO_TRI_HPP
#define ORTHO_TRI_HPP

#include <array>
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

namespace Ortho {

struct alignas(128) TriReprojectionError {
public:

  TriReprojectionError(
      Point<double>         img_pnt,
      const RotateQArray&   quaternion,
      const CameraArray&    camera,
      const TranslateArray& transpose) noexcept : pnt2d(img_pnt), q(quaternion), c(camera), t(transpose) {}

  template <typename T>
  auto operator()(const T* const pnt3d, T* residuals) const -> bool {
    std::array<T, 3>   pnt0;
    std::array<T, 3>   pnt1;
    std::array<T, 4>   quaternion;
    std::span<T>       residuals_span{residuals, 2};
    std::span<const T> pnt3d_span{pnt3d, 3};
    for(size_t i = 0; i < 3; ++i) {
      pnt0[i] = pnt3d_span[i] + T(t[i]);
    }
    for(size_t i = 0; i < 4; ++i) {
      quaternion[i] = T(q[i]);
    }
    ceres::QuaternionRotatePoint(quaternion.data(), pnt0.data(), pnt1.data());
    T p1_z = pnt1[2];
    if(ceres::abs(p1_z) < 1e-6) {
      return false;
    }
    residuals_span[0] = T(c[0]) * pnt1[0] / p1_z + T(c[2]) - T(pnt2d.x);
    residuals_span[1] = T(c[1]) * pnt1[1] / p1_z + T(c[3]) - T(pnt2d.y);
    return true;
  }

  static auto create(Point<double> img_pnt, RotateQArray quaternion, CameraArray camera, TranslateArray transpose) noexcept
      -> ceres::CostFunction* {
    TriReprojectionError* error_ptr{};
    try {
      error_ptr = new TriReprojectionError(Point<double>(img_pnt), quaternion, camera, transpose);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    try {
      return new ceres::AutoDiffCostFunction<TriReprojectionError, 2, 3>(error_ptr);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
  }

private:

  Point<double> pnt2d;

  RotateQArray   q;
  CameraArray    c;
  TranslateArray t;
};

struct alignas(64) TriRes {
  std::array<double, 3> pnt3d;
  PointIdxs             pnt2d_idx_vec;
};

inline auto triangulation(const MatchPairs& match_img_pairs, ImgsData& imgs_data, Progress& progress) noexcept
    -> std::vector<TriRes> {
  THIS_MESSAGE("Build tracks");
  progress.reset(static_cast<int>(match_img_pairs.size()));
  TracksMaintainer tracks_maintainer;
  time_function([&] noexcept {
    for(const auto& match_img_pair : match_img_pairs) {
      for(const auto& [lhs, rhs, score] : match_img_pair.matches) {
        tracks_maintainer.append_match(
            PointIdx{.img_idx = match_img_pair.first, .pnt_idx = lhs},
            PointIdx{.img_idx = match_img_pair.second, .pnt_idx = rhs},
            score);
      }
      progress.update();
    }
  });

  std::vector<PointIdxs> pntidx_vecs = tracks_maintainer.get_tracks();
  std::vector<TriRes>    all_res;
  std::mutex             mtx;
  run(
      pntidx_vecs.size(),
      [&all_res, &pntidx_vecs, &imgs_data, &mtx](int idx) noexcept {
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
          const auto& img                = imgs_data[img_idx];
          const auto& kpnt               = img.get_kpnts().get(pnt_idx);

          // std::cout << "Image " << img_idx << ", Point " << pnt_idx << ": "
          //           << "kpnt=[" << kpnt << "], R=" << img.R_proj() << ", t=" << img.t_proj() << std::endl;

          cv::Mat kpnt_mat = (img.K_bproj() * kpnt);
          double  u_pix    = kpnt_mat.at<double>(0, 0);
          double  v_pix    = kpnt_mat.at<double>(1, 0);
          // std::cout << "u=" << u << ", v=" << v << std::endl;
          cv::Mat         R_mat = img.R_proj();
          Eigen::Matrix3d R_eigen;
          cv::cv2eigen(R_mat, R_eigen);
          cv::Mat t_mat                        = img.t_proj();
          double  t_x                          = t_mat.at<double>(0, 0);
          double  t_y                          = t_mat.at<double>(1, 0);
          double  t_z                          = t_mat.at<double>(2, 0);
          A_matrix.block<1, 3>(i * 2, 0)       = u_pix * R_eigen.row(2) - R_eigen.row(0);
          A_matrix.block<1, 3>((i * 2) + 1, 0) = v_pix * R_eigen.row(2) - R_eigen.row(1);
          b_vector(i * 2)                      = t_x - u_pix * t_z;
          b_vector((i * 2) + 1)                = t_y - v_pix * t_z;
        }
        Eigen::VectorXd x_vector = A_matrix.colPivHouseholderQr().solve(b_vector);
        THIS_ASSERTION_SHOULD_TRUE(x_vector.array().isFinite().all());
        std::array<double, 3> world_point{x_vector(0), x_vector(1), x_vector(2)};
        ceres::Problem        problem;
        problem.AddParameterBlock(world_point.data(), world_point.size());
        for(const auto& pntidx : pntidx_vec) {
          const auto& img = imgs_data[pntidx.img_idx];
          try {
            problem.AddResidualBlock(
                TriReprojectionError::create(
                    img.get_kpnts().get(pntidx.pnt_idx),
                    rotate2qarray(img.R_proj()),
                    camera2array(img.K_proj()),
                    translate2array(img.t_proj())),
                new ceres::HuberLoss(1.0),
                world_point.data());
          } catch(const std::exception& e) {
            report_error(e, "Bad allocation");
          }
        }
        ceres::Solver::Options options;
        options.linear_solver_type           = ceres::DENSE_QR;
        options.check_gradients              = false;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations           = 1000;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << '\n';
        if(summary.IsSolutionUsable()) {
          std::lock_guard lock{mtx};
          all_res
              .emplace_back(std::array<double, 3>{world_point[0], world_point[1], world_point[2]}, std::move(pntidx_vec));
        }
      },
      progress);
  return all_res;
}
} // namespace Ortho

#endif
