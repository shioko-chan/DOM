#ifndef ORTHO_REPROJECTION_ERROR_HPP
#define ORTHO_REPROJECTION_ERROR_HPP

#include <array>
#include <span>

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
    std::array       axisangle{T(q[0]), T(q[1]), T(q[2])};
    std::array<T, 3> point{};
    std::span<T>     residuals_span{residuals, 2};
    ceres::AngleAxisRotatePoint(axisangle.data(), point_3d, point.data());
    point[0] += t[0];
    point[1] += t[1];
    point[2] += t[2];
    const T point_x   = point[1];
    const T point_y   = -point[0];
    const T point_z   = point[2];
    const T f_x       = T(c[0]);
    const T f_y       = T(c[1]);
    const T c_x       = T(c[2]);
    const T c_y       = T(c[3]);
    const T predict_x = (point_x * f_x / point_z) + c_x;
    const T predict_y = (point_y * f_y / point_z) + c_y;
    residuals_span[0] = predict_x - observe_x;
    residuals_span[1] = predict_y - observe_y;
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
    std::array<T, 3>   point{};
    std::span<const T> transpose_span{transpose, 3};
    std::span<T>       residuals_span{residuals, 2};
    std::span<const T> camera_span{camera, 4};
    std::span<const T> distort_span{distort, 5};
    ceres::AngleAxisRotatePoint(axisangle, point_3d, point.data());
    point[0] += transpose_span[0];
    point[1] += transpose_span[1];
    point[2] += transpose_span[2];
    const T point_x   = point[1];
    const T point_y   = -point[0];
    const T point_z   = point[2];
    const T f_x       = camera_span[0];
    const T f_y       = camera_span[1];
    const T c_x       = camera_span[2];
    const T c_y       = camera_span[3];
    const T k_1       = distort_span[0];
    const T k_2       = distort_span[1];
    const T k_3       = distort_span[2];
    const T p_1       = distort_span[3];
    const T p_2       = distort_span[4];
    const T predict_x = (point_x * f_x / point_z) + c_x;
    const T predict_y = (point_y * f_y / point_z) + c_y;
    residuals_span[0] = predict_x - observe_x;
    residuals_span[1] = predict_y - observe_y;
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