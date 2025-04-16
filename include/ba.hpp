#ifndef ORTHO_BA_HPP
#define ORTHO_BA_HPP

#include <array>
#include <thread>
#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "utility.hpp"

namespace Ortho {

struct ReprojectionError {
  ReprojectionError(Point<double> img_pnt_lhs, Point<double> img_pnt_rhs) :
      img_pnt_lhs(img_pnt_lhs), img_pnt_rhs(img_pnt_rhs) {}

  template <typename T>
  bool operator()(
      const T* const q_lhs,
      const T* const q_rhs,
      const T* const t_lhs,
      const T* const t_rhs,
      const T* const camera_lhs,
      const T* const camera_rhs,
      const T* const point3d,
      T*             residuals) const {
    auto calculate = [this, point3d](T* residuals, const T* const q, const T* const t, const T* const c, T u, T v) {
      T p[3];
      ceres::QuaternionRotatePoint(q, point3d, p);
      for(size_t i = 0; i < 3; ++i) {
        p[i] += t[i];
      }
      T predicted_x = c[0] * p[0] / p[2] + c[2];
      T predicted_y = c[1] * p[1] / p[2] + c[3];
      residuals[0]  = predicted_x - u;
      residuals[1]  = predicted_y - v;
    };
    calculate(residuals, q_lhs, t_lhs, camera_lhs, T(img_pnt_lhs.x), T(img_pnt_lhs.y));
    calculate(residuals + 2, q_rhs, t_rhs, camera_rhs, T(img_pnt_rhs.x), T(img_pnt_rhs.y));
    return true;
  }

  static ceres::CostFunction* create(const Point<float>& img_pnt_lhs, const Point<float>& img_pnt_rhs) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 4, 4, 4, 3, 3, 4, 4, 3>(
        new ReprojectionError(Point<double>(img_pnt_lhs.x, img_pnt_lhs.y), Point<double>(img_pnt_rhs.x, img_pnt_rhs.y)));
  }

  Point<double> img_pnt_lhs, img_pnt_rhs;
};

struct BAResults {
  cv::Mat                    R_lhs, R_rhs;
  cv::Mat                    t_lhs, t_rhs;
  cv::Mat                    K_lhs, K_rhs;
  std::vector<Point3<float>> points3d;
};

BAResults
ba(const Points<float>&              lhs_pnts,
   const Points<float>&              rhs_pnts,
   ImgData&                          img_lhs,
   ImgData&                          img_rhs,
   const std::vector<Point3<float>>& pnts3d) {
  ceres::Problem problem;

  // Camera parameters: [rotation(4), translation(3), camera(4)(fx(1), fy(1), cx(1), cy(1))]
  auto add_parameter_block = [&problem](auto& param) { problem.AddParameterBlock(param.data(), param.size()); };
  auto add_parameter_block_quaternion = [&problem](auto& param) {
    problem.AddParameterBlock(param.data(), param.size(), new ceres::QuaternionManifold());
  };

  auto quaternion = [](const cv::Mat& R) -> std::array<double, 4> {
    Eigen::Matrix3d m;
    cv::cv2eigen(R, m);
    Eigen::Quaterniond q(m);
    return {q.w(), q.x(), q.y(), q.z()};
  };
  auto q_lhs = quaternion(img_lhs.R()), q_rhs = quaternion(img_rhs.R());
  add_parameter_block_quaternion(q_lhs), add_parameter_block_quaternion(q_rhs);

  auto get_transpose_params = [](const cv::Mat& t) -> std::array<double, 3> {
    return {t.at<float>(0), t.at<float>(1), t.at<float>(2)};
  };
  auto t_lhs = get_transpose_params(img_lhs.t()), t_rhs = get_transpose_params(img_rhs.t());
  add_parameter_block(t_lhs), add_parameter_block(t_rhs);

  auto get_camera_params = [](const cv::Mat& K) -> std::array<double, 4> {
    return {K.at<float>(0, 0), K.at<float>(1, 1), K.at<float>(0, 2), K.at<float>(1, 2)};
  };
  auto camera_lhs = get_camera_params(img_lhs.K()), camera_rhs = get_camera_params(img_rhs.K());
  add_parameter_block(camera_lhs), add_parameter_block(camera_rhs);

  auto set_lower_bound = [&problem](auto& param, double lower_bound = 0.0) {
    for(size_t i = 0; i < param.size(); ++i) {
      problem.SetParameterLowerBound(param.data(), i, lower_bound);
    }
  };
  set_lower_bound(camera_lhs);
  set_lower_bound(camera_rhs);

  problem.SetParameterBlockConstant(q_lhs.data());
  problem.SetParameterBlockConstant(q_rhs.data());
  problem.SetParameterBlockConstant(t_lhs.data());
  problem.SetParameterBlockConstant(t_rhs.data());
  problem.SetParameterBlockConstant(camera_lhs.data());
  problem.SetParameterBlockConstant(camera_rhs.data());

  std::vector<std::array<double, 3>> points3d(lhs_pnts.size());
  for(int i = 0; i < pnts3d.size(); ++i) {
    points3d[i] = {pnts3d[i].x, pnts3d[i].y, pnts3d[i].z};
    add_parameter_block(points3d[i]);
    ceres::CostFunction* cost_function = ReprojectionError::create(lhs_pnts[i], rhs_pnts[i]);
    // problem.SetParameterBlockConstant(points3d[i].data());
    problem.AddResidualBlock(
        cost_function,
        new ceres::HuberLoss(1.0),
        q_lhs.data(),
        q_rhs.data(),
        t_lhs.data(),
        t_rhs.data(),
        camera_lhs.data(),
        camera_rhs.data(),
        points3d[i].data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type           = ceres::DENSE_QR;
  options.check_gradients              = false;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations           = 500;
  options.num_threads                  = std::thread::hardware_concurrency();

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  auto v     = std::views::transform(points3d, [](const auto& point) {
    return Point3<float>{static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2])};
  });
  auto q2mat = [](const std::array<double, 4>& q) {
    Eigen::Quaterniond quaternion(q[0], q[1], q[2], q[3]);
    Eigen::Matrix3d    m = quaternion.toRotationMatrix();
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
    cv::Mat K         = (cv::Mat_<float>(3, 3) << k[0], 0, k[2], 0, k[1], k[3], 0, 0, 1);
    K.at<float>(0, 0) = k[0];
    K.at<float>(1, 1) = k[1];
    K.at<float>(0, 2) = k[2];
    K.at<float>(1, 2) = k[3];
    return K;
  };
  return BAResults{
      .R_lhs    = q2mat(q_lhs),
      .R_rhs    = q2mat(q_rhs),
      .t_lhs    = t2mat(t_lhs),
      .t_rhs    = t2mat(t_rhs),
      .K_lhs    = k2mat(camera_lhs),
      .K_rhs    = k2mat(camera_rhs),
      .points3d = std::vector<Point3<float>>{v.begin(), v.end()}};
}
} // namespace Ortho
#endif