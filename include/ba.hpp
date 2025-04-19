#ifndef ORTHO_BA_HPP
#define ORTHO_BA_HPP

#include <array>
#include <cassert>
#include <thread>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {
namespace Ba {
struct ReprojectionError {
public:

  ReprojectionError(Point<double> img_pnt) : pnt2d(std::move(img_pnt)) {}

  template <typename T>
  bool operator()(const T* const q, const T* const t, const T* const c, const T* const pnt3d, T* residuals) const {
    T p0[3];
    for (size_t i = 0; i < 3; ++i) {
      p0[i] = pnt3d[i] + t[i];
    }
    T p1[3];
    ceres::rotate2qarrayRotatePoint(q, p0, p1);
    residuals[0] = c[0] * p1[0] / p1[2] + c[2] - T(pnt2d.x);
    residuals[1] = c[1] * p1[1] / p1[2] + c[3] - T(pnt2d.y);
    return true;
  }

  static ceres::CostFunction* create(const Point<float>& img_pnt) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 4, 3>(
        new ReprojectionError(Point<double>(img_pnt)));
  }

private:

  Point<double> pnt2d;
};

void ba(ImgsData& imgs_data, std::vector<Tri::TriRes>& res) {
  ceres::Problem problem;

  auto add_parameter_block = [&problem](auto& param) { problem.AddParameterBlock(param.data(), param.size()); };
  auto add_parameter_block_rotate2qarray = [&problem](auto& param) {
    problem.AddParameterBlock(param.data(), param.size(), new ceres::rotate2qarrayManifold());
    };

  auto q_lhs = rotate2qarray(img_lhs.R()), q_rhs = rotate2qarray(img_rhs.R());
  add_parameter_block_rotate2qarray(q_lhs), add_parameter_block_rotate2qarray(q_rhs);

  auto t_lhs = transpose2array(img_lhs.t()), t_rhs = transpose2array(img_rhs.t());
  add_parameter_block(t_lhs), add_parameter_block(t_rhs);

  auto camera_lhs = intrinsic2array(img_lhs.K()), camera_rhs = intrinsic2array(img_rhs.K());
  add_parameter_block(camera_lhs), add_parameter_block(camera_rhs);

  auto set_lower_bound = [&problem](auto& param, size_t idx, double lower_bound = 0.0) {
    problem.SetParameterLowerBound(param.data(), idx, lower_bound);
    };

  set_lower_bound(camera_lhs, 0);
  set_lower_bound(camera_lhs, 1);

  set_lower_bound(camera_rhs, 0);
  set_lower_bound(camera_rhs, 1);

  std::vector<std::array<double, 3>> optimized_pnts3d(lhs_pnts.size());
  for (int i = 0; i < input_pnts3d.size(); ++i) {
    optimized_pnts3d[i] = { input_pnts3d[i].x, input_pnts3d[i].y, input_pnts3d[i].z };
    add_parameter_block(optimized_pnts3d[i]);
    ceres::CostFunction* cost_function = ReprojectionError::create(lhs_pnts[i], rhs_pnts[i]);
    problem.AddResidualBlock(
        cost_function,
        new ceres::HuberLoss(1.0),
        q_lhs.data(),
        q_rhs.data(),
        t_lhs.data(),
        t_rhs.data(),
        camera_lhs.data(),
        camera_rhs.data(),
        optimized_pnts3d[i].data());
  }

  ceres::Solver::Options options;
  options.num_threads = std::thread::hardware_concurrency();
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.check_gradients = false;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations = 1000;

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  auto v = std::views::transform(optimized_pnts3d, [](const auto& point) {
    return Point3<float>{static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2])};
  });
  auto q2mat = [](const std::array<double, 4>& q) {
    Eigen::rotate2qarrayd rotate2qarray(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d    m = rotate2qarray.toRotationMatrix();
    cv::Mat            R;
    cv::eigen2cv(m, R);
    R.convertTo(R, CV_32F);
    return R;
    };
  auto t2mat = [](const std::array<double, 3>& t) {
    cv::Mat T(3, 1, CV_32F);
    T.at<float>(0, 0) = t[0];
    T.at<float>(1, 0) = t[1];
    T.at<float>(2, 0) = t[2];
    return T;
    };
  auto k2mat = [](const std::array<double, 4>& k) {
    cv::Mat K = (cv::Mat_<float>(3, 3) << k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1);
    K.at<float>(0, 0) = k[0];
    K.at<float>(1, 1) = k[1];
    K.at<float>(0, 2) = k[2];
    K.at<float>(1, 2) = k[3];
    return K;
    };
  return BAResults {
      .R_lhs = q2mat(q_lhs),
      .R_rhs = q2mat(q_rhs),
      .t_lhs = t2mat(t_lhs),
      .t_rhs = t2mat(t_rhs),
      .K_lhs = k2mat(camera_lhs),
      .K_rhs = k2mat(camera_rhs),
      .points3d = std::vector<Point3<float>>{v.begin(), v.end()} };
}
}
} // namespace Ortho
#endif
