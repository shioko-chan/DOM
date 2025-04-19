#ifndef ORTHO_TRI_HPP
#define ORTHO_TRI_HPP

#include <vector>
#include <array>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "tracks.hpp"
#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {
struct ReprojectionError {
public:
  ReprojectionError(Point<double> img_pnt, const std::array<double, 4>& q, const std::array<double, 4>& c, const std::array<double, 3>& t) : pnt2d(std::move(img_pnt)), q(std::move(q)), c(std::move(c)), t(std::move(t)) {}
  template <typename T>
  bool operator()(const T* const pnt3d, T* residuals) const {
    T p0[3];
    for (size_t i = 0; i < 3; ++i) {
      p0[i] = pnt3d[i] + T(t[i]);
    }
    T p1[3], q[4];
    for (size_t i = 0; i < 4; ++i) {
      q[i] = T(this->q[i]);
    }
    ceres::QuaternionRotatePoint(q, p0, p1);
    residuals[0] = T(c[0]) * p1[0] / p1[2] + T(c[2]) - T(pnt2d.x);
    residuals[1] = T(c[1]) * p1[1] / p1[2] + T(c[3]) - T(pnt2d.y);
    return true;
  }
  static ceres::CostFunction* create(const Point<float>& img_pnt, std::array<double, 4> q, std::array<double, 4> c, std::array<double, 3> t) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3>(
        new ReprojectionError(Point<double>(img_pnt), std::move(q), std::move(c), std::move(t)));
  }
private:
  Point<double> pnt2d;
  std::array<double, 4> q, c;
  std::array<double, 3> t;
};

struct TriRes {
  Point3<float> pnt3d;
  PointIdxs     pnt2d_idx_vec;
};

std::vector<TriRes> triangulation(const MatchPairs& match_img_pairs, ImgsData& imgs_data) {
  TracksMaintainer tracks_maintainer;
  for (const auto& match_img_pair : match_img_pairs) {
    for (const auto& [lhs, rhs, score] : match_img_pair.matches) {
      tracks_maintainer.append_match(
          PointIdx { match_img_pair.first, lhs }, PointIdx { match_img_pair.second, rhs }, score);
    }
  }
  std::vector<PointIdxs> pntidx_vecs = tracks_maintainer.get_tracks();

  auto get_v_c2w = [&imgs_data](const PointIdx& idx) {
    const auto& [img_idx, pnt_idx] = idx;
    auto& img = imgs_data[img_idx];
    cv::Mat v_m = img.R().t() * img.K().inv() * img.kpnts.get(pnt_idx);
    v_m.convertTo(v_m, CV_64F);
    Eigen::Vector3d v;
    cv::cv2eigen(v_m, v);
    return v;
    };
  auto get_t_c2w = [&imgs_data](const PointIdx& idx) {
    const auto& [img_idx, _] = idx;
    auto& img = imgs_data[img_idx];
    cv::Mat t_m = -img.t();
    t_m.convertTo(t_m, CV_64F);
    Eigen::Vector3d t;
    cv::cv2eigen(t_m, t);
    return t;
    };
  std::vector<TriRes> res;
  for (auto& pntidx_vec : pntidx_vecs) {
    size_t n = pntidx_vec.size();
    assert(n > 1);
    size_t          rows = 3 * n, cols = n + 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::VectorXd b(rows);
    for (size_t i = 0; i < n; ++i) {
      Eigen::Vector3d v = get_v_c2w(pntidx_vec[i]), t = get_t_c2w(pntidx_vec[i]);
      A.block<3, 3>(i * 3, 0) = -Eigen::Matrix3d::Identity();
      A.block<3, 1>(i * 3, 3) = v;
      b.segment<3>(i * 3) = t;
    }
    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    std::array<double, 3> wp { x(0), x(1), x(2) };
    ceres::Problem problem;
    problem.AddParameterBlock(wp.data(), wp.size());
    for (const auto& pntidx : pntidx_vec) {
      ImgData& img = imgs_data[pntidx.img_idx];
      ceres::CostFunction* cost = ReprojectionError::create(img.kpnts[pntidx.pnt_idx], quaternion(img.R()), get_camera_params(img.K()),
       get_transpose_params(img.t())
      );
      problem.AddResidualBlock(cost, new ceres::HuberLoss(1.0), wp.data());
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.check_gradients = false;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 1000;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << std::endl;
    if (summary.IsSolutionUsable()) {
      res.emplace_back(Point3<float>(wp[0], wp[1], wp[2]), std::move(pntidx_vec));
    }
  }
}
} // namespace Ortho

#endif
