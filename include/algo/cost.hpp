#ifndef ORTHO_REPROJECTION_ERROR_HPP
#define ORTHO_REPROJECTION_ERROR_HPP

#include <array>
#include <span>

#include <ceres/ceres.h>
#include <ceres/problem.h>
#include <ceres/rotation.h>

#include "tools/report_error.hpp"
#include "types/common_types.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

// struct alignas(64) SoftRotationConstraint {
//   explicit SoftRotationConstraint(Eigen::Quaterniond prior_q, double weight) :
//       prior_q_(std::move(prior_q)), weight_(weight) {}

//   template <typename T>
//   auto operator()(const T* const q_raw, T* residuals) const noexcept -> bool {
//     std::span<T>         q_span{q_raw, 4};
//     Eigen::Quaternion<T> q_curr(q_span[0], q_span[1], q_span[2], q_span[3]);
//     q_curr.normalize();
//     Eigen::Quaternion<T>   q_prior(T(prior_q_.w()), T(prior_q_.x()), T(prior_q_.y()), T(prior_q_.z()));
//     Eigen::Quaternion<T>   delta_q = q_prior * q_curr.conjugate();
//     Eigen::Matrix<T, 3, 1> angle_axis;
//     ceres::QuaternionToAngleAxis(delta_q.coeffs().data(), angle_axis.data());
//     residuals[0] = weight_ * angle_axis[0];
//     residuals[1] = weight_ * angle_axis[1];
//     residuals[2] = weight_ * angle_axis[2];
//     return true;
//   }

// private:

//   Eigen::Quaterniond prior_q_;
//   double             weight_;
// };

// struct alignas(32) SoftConstraint {
//   SoftConstraint(const double* prior, double weight) : prior_{prior[0], prior[1], prior[2]}, weight_{weight} {}

//   template <typename T>
//   auto operator()(const T* const param, T* residuals) const noexcept -> bool {
//     for(int i = 0; i < 3; ++i) {
//       residuals[i] = T(weight_) * (param[i] - T(prior_[i]));
//     }
//     return true;
//   }

//   double prior_[3];
//   double weight_;
// };

struct alignas(128) SimpReprojectionError {
public:

  SimpReprojectionError(
      Point<double>         img_pnt,
      const RotateQArray&   quaternion,
      const CameraArray&    camera,
      const TranslateArray& transpose) noexcept : point_2d(img_pnt), q(quaternion), c(camera), t(transpose) {}

  template <typename T>
  auto operator()(const T* const point_3d, T* residuals) const -> bool {
    std::array       quaternion{T(q[0]), T(q[1]), T(q[2]), T(q[3])};
    std::array<T, 3> point{};
    std::span<T>     residuals_span{residuals, 2};
    ceres::QuaternionRotatePoint(quaternion.data(), point_3d, point.data());
    point[0] += T(t[0]);
    point[1] += T(t[1]);
    point[2] += T(t[2]);
    // T point_z         = ceres::fmax(point[2], T(1e-6));
    T point_z         = point[2];
    residuals_span[0] = T(c[0]) * point[1] / point_z + T(c[2]) - T(point_2d.x);
    residuals_span[1] = -T(c[1]) * point[0] / point_z + T(c[3]) - T(point_2d.y);
    return true;
  }

  static auto create(Point<double> img_pnt, RotateQArray quaternion, CameraArray camera, TranslateArray transpose) noexcept
      -> ceres::CostFunction* {
    SimpReprojectionError* error_ptr{};
    try {
      error_ptr = new SimpReprojectionError(Point<double>(img_pnt), quaternion, camera, transpose);
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

  Point<double> point_2d;

  RotateQArray   q;
  CameraArray    c;
  TranslateArray t;
};

struct alignas(16) ReprojectionError {
public:

  explicit ReprojectionError(Point<double> img_pnt) noexcept : point_2d(img_pnt) {}

  template <typename T>
  auto operator()(
      const T* const quaternion,
      const T* const transpose,
      const T* const camera,
      const T* const point_3d,
      T*             residuals) const noexcept -> bool {
    std::array<T, 3>   point{};
    std::span<const T> transpose_span{transpose, 3};
    std::span<T>       residuals_span{residuals, 2};
    std::span<const T> camera_span{camera, 4};
    ceres::QuaternionRotatePoint(quaternion, point_3d, point.data());
    point[0] += transpose_span[0];
    point[1] += transpose_span[1];
    point[2] += transpose_span[2];
    // T point_z         = ceres::fmax(point[2], T(1e-6));
    T point_z         = point[2];
    residuals_span[0] = camera_span[0] * point[1] / point_z + camera_span[2] - T(point_2d.x);
    residuals_span[1] = -camera_span[1] * point[0] / point_z + camera_span[3] - T(point_2d.y);
    return true;
  }

  static auto create(Point<double> img_pnt) noexcept -> ceres::CostFunction* {
    ReprojectionError* error_ptr{};
    try {
      error_ptr = new ReprojectionError(img_pnt);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    try {
      return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 4, 3, 4, 3>(error_ptr);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
  }

private:

  Point<double> point_2d;
};

void add_parameter_block(ceres::Problem& problem, auto& param, ceres::Manifold* manifold = nullptr) noexcept {
  if(manifold) {
    problem.AddParameterBlock(param.data(), static_cast<int>(param.size()), manifold);
  } else {
    problem.AddParameterBlock(param.data(), static_cast<int>(param.size()));
  }
};

void set_parameter_block_constant(ceres::Problem& problem, const auto& param) {
  problem.SetParameterBlockConstant(param.data());
}

void set_parameter_block_variable(ceres::Problem& problem, auto& param) {
  problem.SetParameterBlockVariable(param.data());
}

void set_bound(ceres::Problem& problem, auto& param, size_t idx, double lower_bound = 0.0, double upper_bound = 0.0) {
  problem.SetParameterLowerBound(param.data(), idx, lower_bound);
  problem.SetParameterUpperBound(param.data(), idx, upper_bound);
}

void set_bound_delta(ceres::Problem& problem, auto& param, size_t idx, double delta = 0.0) {
  double value = param[idx];
  set_bound(problem, param, idx, value - delta, value + delta);
}

void set_bound_percentage(ceres::Problem& problem, auto& param, size_t idx, double percentage = 0.0) {
  double value = param[idx];
  percentage /= 100.0;
  set_bound(problem, param, idx, (1.0 - percentage) * value, (1.0 + percentage) * value);
}

} // namespace Ortho
#endif