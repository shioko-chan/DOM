#ifndef ORTHO_REPROJECTION_ERROR_HPP
#define ORTHO_REPROJECTION_ERROR_HPP

#include <span>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/numeric_diff_cost_function.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>
#include <ceres/types.h>

#include "tools/report_error.hpp"
#include "types/common_types.hpp"

namespace Ortho {

struct alignas(128) SimpReprojectionError {
public:

  SimpReprojectionError(
      double                 observe_x,
      double                 observe_y,
      const RotateAxisAngle& axisangle,
      const CameraArray&     camera,
      const TranslateArray&  transpose) noexcept :
      observe_x(observe_x), observe_y(observe_y), q(axisangle), c(camera), t(transpose) {}

  template <typename T>
  auto operator()(const T* const point_3d, T* residuals) const noexcept -> bool {
    Eigen::Matrix<T, 3, 1> axisangle{T(q[0]), T(q[1]), T(q[2])};
    Eigen::Matrix<T, 3, 1> point;
    std::span<T>           resid{residuals, 2};
    ceres::AngleAxisRotatePoint(axisangle.data(), point_3d, point.data());
    point(0) += t[0];
    point(1) += t[1];
    point(2) += t[2];

    const T point_x = point(1);
    const T point_y = -point(0);
    const T point_z = point(2);

    const T f_x       = T(c[0]);
    const T f_y       = T(c[1]);
    const T c_x       = T(c[2]);
    const T c_y       = T(c[3]);
    const T predict_x = (point_x * f_x / point_z) + c_x;
    const T predict_y = (point_y * f_y / point_z) + c_y;
    resid[0]          = predict_x - observe_x;
    resid[1]          = predict_y - observe_y;
    return true;
  }

  static auto
  create(double observe_x, double observe_y, RotateAxisAngle axisangle, CameraArray camera, TranslateArray transpose) noexcept
      -> ceres::CostFunction* {
    SimpReprojectionError* error_ptr{};
    try {
      error_ptr = new SimpReprojectionError(observe_x, observe_y, axisangle, camera, transpose);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    try {
      return new ceres::AutoDiffCostFunction<SimpReprojectionError, 2, 3>(error_ptr);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
  }

private:

  double          observe_x, observe_y;
  RotateAxisAngle q;
  CameraArray     c;
  TranslateArray  t;
};

struct alignas(16) ReprojectionError {
public:

  explicit ReprojectionError(double observe_x, double observe_y) noexcept :
      observe_x(observe_x), observe_y(observe_y) {}

  template <typename T>
  auto operator()(
      const T* const axisangle,
      const T* const transpose,
      const T* const camera,
      const T* const distort,
      const T* const point_3d,
      T*             residuals) const noexcept -> bool {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> transpose_eigen(transpose);
    Eigen::Map<const Eigen::Matrix<T, 4, 1>> camera_eigen(camera);
    Eigen::Map<const Eigen::Matrix<T, 5, 1>> distort_eigen(distort);
    Eigen::Matrix<T, 3, 1>                   point;
    std::span<T>                             resid{residuals, 2};

    ceres::AngleAxisRotatePoint(axisangle, point_3d, point.data());
    point += transpose_eigen;

    const T point_x = point(1);
    const T point_y = -point(0);
    const T point_z = point(2);

    const T f_x = camera_eigen(0);
    const T f_y = camera_eigen(1);
    const T c_x = camera_eigen(2);
    const T c_y = camera_eigen(3);

    const T k_1 = distort_eigen(0);
    const T k_2 = distort_eigen(1);
    const T k_3 = distort_eigen(2);
    const T p_1 = distort_eigen(3);
    const T p_2 = distort_eigen(4);

    const T norm_x = point_x / point_z;
    const T norm_y = point_y / point_z;

    const T r_2 = (norm_x * norm_x) + (norm_y * norm_y);
    const T r_4 = r_2 * r_2;
    const T r_6 = r_4 * r_2;

    const T radial_distortion = 1.0 + (k_1 * r_2) + (k_2 * r_4) + (k_3 * r_6);
    const T distorted_x =
        (norm_x * radial_distortion) + (2.0 * p_1 * norm_x * norm_y) + (p_2 * (r_2 + 2.0 * norm_x * norm_x));
    const T distorted_y =
        (norm_y * radial_distortion) + (2.0 * p_2 * norm_x * norm_y) + (p_1 * (r_2 + 2.0 * norm_y * norm_y));

    const T predict_x = (distorted_x * f_x) + c_x;
    const T predict_y = (distorted_y * f_y) + c_y;
    resid[0]          = predict_x - observe_x;
    resid[1]          = predict_y - observe_y;
    return true;
  }

  static auto create(double observe_x, double observe_y) noexcept -> ceres::CostFunction* {
    ReprojectionError* error_ptr{};
    try {
      error_ptr = new ReprojectionError(observe_x, observe_y);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    try {
      return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 4, 5, 3>(error_ptr);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
  }

private:

  double observe_x, observe_y;
};

void add_parameter_block(ceres::Problem& problem, auto& param, ceres::Manifold* manifold = nullptr) noexcept {
  if(manifold) {
    problem.AddParameterBlock(param.data(), static_cast<int>(param.size()), manifold);
  } else {
    problem.AddParameterBlock(param.data(), static_cast<int>(param.size()));
  }
}

void set_parameter_block_constant(ceres::Problem& problem, const auto& param) noexcept {
  problem.SetParameterBlockConstant(param.data());
}

void set_parameter_block_variable(ceres::Problem& problem, auto& param) noexcept {
  problem.SetParameterBlockVariable(param.data());
}

void set_bound(ceres::Problem& problem, auto& param, size_t idx, double lower_bound = 0.0, double upper_bound = 0.0) noexcept {
  problem.SetParameterLowerBound(param.data(), idx, lower_bound);
  problem.SetParameterUpperBound(param.data(), idx, upper_bound);
}

void set_bound_delta(ceres::Problem& problem, auto& param, size_t idx, double delta = 0.0) noexcept {
  double value = param[idx];
  set_bound(problem, param, idx, value - delta, value + delta);
}

void set_bound_percentage(ceres::Problem& problem, auto& param, size_t idx, double percentage = 0.0) noexcept {
  double value = param[idx];
  percentage /= 100.0;
  set_bound(problem, param, idx, (1.0 - percentage) * value, (1.0 + percentage) * value);
}

} // namespace Ortho
#endif