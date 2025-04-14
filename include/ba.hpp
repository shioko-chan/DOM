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
      const T* const camera_lhs,
      const T* const camera_rhs,
      const T* const point3d,
      T*             residuals) const {
    auto calculate = [this, point3d](const T* const q, const T* const camera, Point<float> img_pnt, T* residuals) {
      const T* translation = camera;
      const T& fx          = camera[3];
      const T& fy          = camera[4];
      const T& cx          = camera[5];
      const T& cy          = camera[6];
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
    calculate(q_lhs, camera_lhs, img_pnt_lhs, residuals);
    calculate(q_rhs, camera_rhs, img_pnt_rhs, residuals + 2);
    return true;
  }

  static ceres::CostFunction* Create(Point<float> img_pnt_lhs, Point<float> img_pnt_rhs) {
    return (new ceres::AutoDiffCostFunction<ReprojectionError, 4, 4, 4, 7, 7, 3>(
        new ReprojectionError(img_pnt_lhs, img_pnt_rhs)));
  }

  Point<float> img_pnt_lhs, img_pnt_rhs;
};

void ba(
    const Points<float>&            lhs_pnts,
    const Points<float>&            rhs_pnts,
    ImgData&                        img_lhs,
    ImgData&                        img_rhs,
    const std::vector<cv::Point3f>& pnts3d) {
  ceres::Problem problem;
  // Camera parameters: [rotation(3), translation(3), fx(1), fy(1), cx(1), cy(1)]
  std::vector<std::array<double, 3>> points3d(lhs_pnts.size());
  auto                               quaternion = [](const cv::Mat& R) {
    Eigen::Matrix3d m;
    cv::cv2eigen(R, m);
    Eigen::Quaterniond q(m);
    return std::array<double, 4>{q.x(), q.y(), q.z(), q.w()};
  };
  std::array<double, 4> q_lhs = quaternion(img_lhs.R()), q_rhs = quaternion(img_rhs.R());
  problem.AddParameterBlock(q_lhs.data(), 4, new ceres::QuaternionManifold());
  problem.AddParameterBlock(q_rhs.data(), 4, new ceres::QuaternionManifold());
  auto get_camera_params = [](ImgData& img) {
    return std::array<double, 7>{
        img.t().at<float>(0),
        img.t().at<float>(1),
        img.t().at<float>(2),
        img.K().at<float>(0, 0),
        img.K().at<float>(1, 1),
        img.K().at<float>(0, 2),
        img.K().at<float>(1, 2)};
  };
  std::array<double, 7> camera_lhs = get_camera_params(img_lhs), camera_rhs = get_camera_params(img_rhs);
  for(int i = 0; i < pnts3d.size(); ++i) {
    points3d[i] = {pnts3d[i].x, pnts3d[i].y, pnts3d[i].z};

    ceres::CostFunction* cost_function = ReprojectionError::Create(lhs_pnts[i], rhs_pnts[i]);
    problem.AddResidualBlock(
        cost_function,
        new ceres::HuberLoss(1.0),
        q_lhs.data(),
        q_rhs.data(),
        camera_lhs.data(),
        camera_rhs.data(),
        points3d[i].data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type           = ceres::DENSE_NORMAL_CHOLESKY;
  options.check_gradients              = false;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations           = 500;
  options.num_threads                  = std::thread::hardware_concurrency();

  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}
} // namespace Ortho
#endif