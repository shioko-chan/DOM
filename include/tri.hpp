#ifndef ORTHO_TRI_HPP
#define ORTHO_TRI_HPP

#include <array>
#include <mutex>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "progress.hpp"
#include "tracks.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {

struct TriReprojectionError {
public:

  TriReprojectionError(Point<double> img_pnt, const RotateQArray& q, const CameraArray& c, const TranslateArray& t) :
      pnt2d(std::move(img_pnt)), q(std::move(q)), c(std::move(c)), t(std::move(t)) {}

  template <typename T>
  bool operator()(const T* const pnt3d, T* residuals) const {
    T p0[3];
    for(size_t i = 0; i < 3; ++i) {
      p0[i] = pnt3d[i] + T(t[i]);
    }
    T p1[3], q[4];
    for(size_t i = 0; i < 4; ++i) {
      q[i] = T(this->q[i]);
    }
    ceres::QuaternionRotatePoint(q, p0, p1);
    residuals[0] = T(c[0]) * p1[0] / p1[2] + T(c[2]) - T(pnt2d.x);
    residuals[1] = T(c[1]) * p1[1] / p1[2] + T(c[3]) - T(pnt2d.y);
    return true;
  }

  static ceres::CostFunction* create(const Point<float>& img_pnt, RotateQArray q, CameraArray c, TranslateArray t) {
    return new ceres::AutoDiffCostFunction<TriReprojectionError, 2, 3>(
        new TriReprojectionError(Point<double>(img_pnt), std::move(q), std::move(c), std::move(t)));
  }

private:

  Point<double>  pnt2d;
  RotateQArray   q;
  CameraArray    c;
  TranslateArray t;
};

struct TriRes {
  std::array<double, 3> pnt3d;
  PointIdxs             pnt2d_idx_vec;
};

std::vector<TriRes> triangulation(const MatchPairs& match_img_pairs, ImgsData& imgs_data, Progress& progress) {
  TracksMaintainer tracks_maintainer;
  for(const auto& match_img_pair : match_img_pairs) {
    for(const auto& [lhs, rhs, score] : match_img_pair.matches) {
      tracks_maintainer.append_match(PointIdx{match_img_pair.first, lhs}, PointIdx{match_img_pair.second, rhs}, score);
    }
  }
  std::vector<PointIdxs> pntidx_vecs = tracks_maintainer.get_tracks();
  std::vector<TriRes>    res;
  std::mutex             mtx;
  run(
      pntidx_vecs.size(),
      [&res, &pntidx_vecs, &imgs_data, &mtx](int idx) {
        auto&  pntidx_vec = pntidx_vecs[idx];
        size_t n          = pntidx_vec.size();
        assert(n > 1);
        size_t          rows = 2 * n, cols = 3;
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
        Eigen::VectorXd b(rows);
        for(size_t i = 0; i < n; ++i) {
          const auto& [img_idx, pnt_idx] = pntidx_vec[i];
          const auto& img                = imgs_data[img_idx];
          const auto& kpnt               = img.get_kpnts().get(pnt_idx);

          // std::cout << "Image " << img_idx << ", Point " << pnt_idx << ": "
          //           << "kpnt=[" << kpnt << "], R=" << img.R_proj() << ", t=" << img.t_proj() << std::endl;

          cv::Mat kpnt_mat = img.K_bproj() * kpnt;
          double  u = kpnt_mat.at<float>(0, 0), v = kpnt_mat.at<float>(1, 0);
          std::cout << "u=" << u << ", v=" << v << std::endl;
          cv::Mat R = img.R_proj();
          R.convertTo(R, CV_64F);
          Eigen::Matrix3d R_eigen;
          cv::cv2eigen(R, R_eigen);
          cv::Mat t  = img.t_proj();
          double  tx = t.at<float>(0, 0), ty = t.at<float>(1, 0), tz = t.at<float>(2, 0);
          A.block<1, 3>(i * 2, 0)     = u * R_eigen.row(2) - R_eigen.row(0);
          A.block<1, 3>(i * 2 + 1, 0) = v * R_eigen.row(2) - R_eigen.row(1);
          b(i * 2)                    = tx - u * tz;
          b(i * 2 + 1)                = ty - v * tz;
        }
        Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
        assert(x.array().isFinite().all());
        std::array<double, 3> wp{x(0), x(1), x(2)};
        ceres::Problem        problem;
        problem.AddParameterBlock(wp.data(), wp.size());
        for(const auto& pntidx : pntidx_vec) {
          const auto&          img  = imgs_data[pntidx.img_idx];
          ceres::CostFunction* cost = TriReprojectionError::create(
              img.get_kpnts().get(pntidx.pnt_idx),
              rotate2qarray(img.R_proj()),
              camera2array(img.K_proj()),
              translate2array(img.t_proj()));
          problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), wp.data());
        }
        ceres::Solver::Options options;
        options.linear_solver_type           = ceres::DENSE_QR;
        options.check_gradients              = false;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations           = 1000;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        if(summary.IsSolutionUsable()) {
          std::lock_guard _{mtx};
          res.emplace_back(std::array<double, 3>{wp[0], wp[1], wp[2]}, std::move(pntidx_vec));
        }
      },
      progress);
  return res;
}
} // namespace Ortho

#endif
