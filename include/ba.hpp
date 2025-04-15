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
  ReprojectionError(Point<float> img_pnt_lhs, Point<float> img_pnt_rhs) :
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
    auto calculate =
        [this, point3d](const T* const q, const T* const t, const T* const camera, Point<float> img_pnt, T* residuals) {
          const T* translation = t;
          const T& fx          = camera[0];
          const T& fy          = camera[1];
          const T& cx          = camera[2];
          const T& cy          = camera[3];
          T        p[3];
          ceres::QuaternionRotatePoint(q, point3d, p);
          p[0] += translation[0];
          p[1] += translation[1];
          p[2] += translation[2];
          T xp          = p[0] / p[2];
          T yp          = p[1] / p[2];
          T predicted_x = fx * xp + cx;
          T predicted_y = fy * yp + cy;
          residuals[0]  = predicted_x - T(img_pnt.x);
          residuals[1]  = predicted_y - T(img_pnt.y);
        };
    calculate(q_lhs, t_lhs, camera_lhs, img_pnt_lhs, residuals);
    calculate(q_rhs, t_rhs, camera_rhs, img_pnt_rhs, residuals + 2);
    return true;
  }

  static ceres::CostFunction* Create(Point<float> img_pnt_lhs, Point<float> img_pnt_rhs) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 4, 4, 4, 3, 3, 4, 4, 3>(
        new ReprojectionError(img_pnt_lhs, img_pnt_rhs)));
  }

  Point<float> img_pnt_lhs, img_pnt_rhs;
};

struct BAResults {
  cv::Mat                  R_lhs, R_rhs;
  cv::Mat                  t_lhs, t_rhs;
  cv::Mat                  K_lhs, K_rhs;
  std::vector<cv::Point3f> points3d;
};

BAResults
ba(const Points<float>&            lhs_pnts,
   const Points<float>&            rhs_pnts,
   ImgData&                        img_lhs,
   ImgData&                        img_rhs,
   const std::vector<cv::Point3f>& pnts3d) {
  ceres::Problem problem;
  // Camera parameters: [rotation(3), translation(3), fx(1), fy(1), cx(1), cy(1)]
  auto quaternion = [](const cv::Mat& R) {
    Eigen::Matrix3d m;
    cv::cv2eigen(R, m);
    Eigen::Quaterniond q(m);
    return std::array<double, 4>{q.w(), q.x(), q.y(), q.z()};
  };
  std::array<double, 4> q_lhs = quaternion(img_lhs.R()), q_rhs = quaternion(img_rhs.R());
  problem.AddParameterBlock(q_lhs.data(), 4, new ceres::QuaternionManifold());
  problem.AddParameterBlock(q_rhs.data(), 4, new ceres::QuaternionManifold());
  auto get_camera_params = [](ImgData& img) {
    return std::array<double, 4>{
        img.K().at<float>(0, 0), img.K().at<float>(1, 1), img.K().at<float>(0, 2), img.K().at<float>(1, 2)};
  };
  auto get_transpose_params = [](ImgData& img) {
    return std::array<double, 3>{img.t().at<float>(0), img.t().at<float>(1), img.t().at<float>(2)};
  };

  auto add_parameter_block    = [&problem](auto& param) { problem.AddParameterBlock(param.data(), param.size()); };
  std::array<double, 3> t_lhs = get_transpose_params(img_lhs), t_rhs = get_transpose_params(img_rhs);
  add_parameter_block(t_lhs);
  add_parameter_block(t_rhs);
  std::array<double, 4> camera_lhs = get_camera_params(img_lhs), camera_rhs = get_camera_params(img_rhs);
  add_parameter_block(camera_lhs);
  add_parameter_block(camera_rhs);

  problem.SetParameterLowerBound(camera_lhs.data(), 0, 0.0);
  problem.SetParameterLowerBound(camera_lhs.data(), 1, 0.0);
  problem.SetParameterLowerBound(camera_lhs.data(), 2, 0.0);
  problem.SetParameterLowerBound(camera_lhs.data(), 3, 0.0);

  problem.SetParameterLowerBound(camera_rhs.data(), 0, 0.0);
  problem.SetParameterLowerBound(camera_rhs.data(), 1, 0.0);
  problem.SetParameterLowerBound(camera_rhs.data(), 2, 0.0);
  problem.SetParameterLowerBound(camera_rhs.data(), 3, 0.0);

  std::vector<std::array<double, 3>> points3d(lhs_pnts.size());
  for(int i = 0; i < pnts3d.size(); ++i) {
    points3d[i] = {pnts3d[i].x, pnts3d[i].y, pnts3d[i].z};
    problem.AddParameterBlock(points3d[i].data(), 3);
    ceres::CostFunction* cost_function = ReprojectionError::Create(lhs_pnts[i], rhs_pnts[i]);
    problem.AddResidualBlock(
        cost_function,
        // new ceres::HuberLoss(1.0),
        nullptr,
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
    return cv::Point3f{static_cast<float>(point[0]), static_cast<float>(point[1]), static_cast<float>(point[2])};
  });
  auto q2mat = [](const std::array<double, 4>& q) {
    Eigen::Quaterniond quaternion(q[1], q[2], q[3], q[0]);
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
      .points3d = std::vector<cv::Point3f>{v.begin(), v.end()}};
}
} // namespace Ortho
#endif