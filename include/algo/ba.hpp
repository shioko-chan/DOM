#ifndef ORTHO_BA_HPP
#define ORTHO_BA_HPP

#include <array>
#include <cassert>
#include <exception>
#include <thread>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "ds/imgdata.hpp"
#include "tools/report_error.hpp"
#include "types/common_types.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {
struct alignas(16) BaReprojectionError {
public:

  explicit BaReprojectionError(Point<double> img_pnt) noexcept : pnt2d(img_pnt) {}

  template <typename T>
  auto operator()(
      const T* const quaternion,
      const T* const transpose,
      const T* const camera,
      const T* const pnt3d,
      T*             residuals) const noexcept -> bool {
    std::array<T, 3>   pnt0;
    std::array<T, 3>   pnt1;
    std::span<const T> pnt3d_span{pnt3d, 3};
    std::span<const T> transpose_span{transpose, 3};
    std::span<const T> camera_span{camera, 4};
    std::span<T>       residuals_span{residuals, 2};
    for(size_t i = 0; i < 3; ++i) {
      pnt0[i] = pnt3d_span[i] + transpose_span[i];
    }
    ceres::QuaternionRotatePoint(quaternion, pnt0.data(), pnt1.data());
    T p1_z = pnt1[2];
    if(ceres::abs(p1_z) < 1e-6) {
      return false;
    }
    residuals_span[0] = camera_span[0] * pnt1[0] / p1_z + camera_span[2] - T(pnt2d.x);
    residuals_span[1] = camera_span[1] * pnt1[1] / p1_z + camera_span[3] - T(pnt2d.y);
    return true;
  }

  static auto create(Point<double> img_pnt) noexcept -> ceres::CostFunction* {
    BaReprojectionError* error_ptr{};
    try {
      error_ptr = new BaReprojectionError(img_pnt);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    try {
      return new ceres::AutoDiffCostFunction<BaReprojectionError, 2, 4, 3, 4, 3>(error_ptr);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
  }

private:

  Point<double> pnt2d;
};

void ba(ImgsData& imgs_data, auto& res) noexcept {
  ceres::Problem problem;
  auto           add_parameter_block = [&problem](auto& param, ceres::Manifold* manifold = nullptr) {
    if(manifold) {
      problem.AddParameterBlock(param.data(), param.size(), manifold);
    } else {
      problem.AddParameterBlock(param.data(), param.size());
    }
  };
  auto set_bound = [&problem](auto& param, size_t idx, double lower_bound = 0.0, double upper_bound = 0.0) {
    problem.SetParameterLowerBound(param.data(), idx, lower_bound);
    problem.SetParameterUpperBound(param.data(), idx, upper_bound);
  };
  auto set_bound_delta = [&set_bound](auto& param, size_t idx, double delta = 0.0) {
    double value = param[idx];
    set_bound(param, idx, value - delta, value + delta);
  };
  auto set_bound_percentage = [&set_bound](auto& param, size_t idx, double percentage = 0.0) {
    double value = param[idx];
    percentage /= 100.0;
    set_bound(param, idx, (1.0 - percentage) * value, (1.0 + percentage) * value);
  };
  for(auto& img_data : imgs_data) {
    try {
      add_parameter_block(img_data.Q_proj_array_raw(), new ceres::QuaternionManifold);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    add_parameter_block(img_data.t_proj_array_raw());
    add_parameter_block(img_data.camera_array_raw());

    // set_bound_delta(img_data.t_proj_array_raw(), 0, 5);
    // set_bound_delta(img_data.t_proj_array_raw(), 1, 5);
    // set_bound_delta(img_data.t_proj_array_raw(), 2, 25);
    // set_bound_percentage(img_data.camera_array_raw(), 0, 5);
    // set_bound_percentage(img_data.camera_array_raw(), 1, 5);
    // set_bound_delta(img_data.camera_array_raw(), 2, 10);
    // set_bound_delta(img_data.camera_array_raw(), 3, 10);

    problem.SetParameterBlockConstant(img_data.Q_proj_array_raw().data());
    problem.SetParameterBlockConstant(img_data.t_proj_array_raw().data());
    problem.SetParameterBlockConstant(img_data.camera_array_raw().data());
  }
  for(auto& [pnt3d, pnt2d_idx_vec] : res) {
    if(pnt2d_idx_vec.empty()) {
      continue;
    }
    add_parameter_block(pnt3d);
    for(const auto& pnt2d_idx : pnt2d_idx_vec) {
      auto& img_data = imgs_data[pnt2d_idx.img_idx];
      try {
        problem.AddResidualBlock(
            BaReprojectionError::create(img_data.get_kpnts().get(pnt2d_idx.pnt_idx)),
            new ceres::HuberLoss(1.0),
            img_data.Q_proj_array_raw().data(),
            img_data.t_proj_array_raw().data(),
            img_data.camera_array_raw().data(),
            pnt3d.data());
      } catch(const std::exception& e) {
        report_error(e, "Bad allocation");
      }
    }
  }
  ceres::Solver::Options options;
  options.num_threads                  = static_cast<int>(std::thread::hardware_concurrency());
  options.linear_solver_type           = ceres::SPARSE_SCHUR;
  options.check_gradients              = false;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations           = 2000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << '\n';

  // for(auto& img_data : imgs_data) {
  //   problem.SetParameterBlockVariable(img_data.Q_proj_array_raw().data());
  //   problem.SetParameterBlockVariable(img_data.t_proj_array_raw().data());
  //   problem.SetParameterBlockVariable(img_data.camera_array_raw().data());
  //   set_percentage_bounds(img_data.t_proj_array_raw(), 10);
  //   set_percentage_bounds(img_data.camera_array_raw(), 10);
  // }

  // for(auto& [pnt3d, pnt2d_idx_vec] : res) {
  //   if(pnt2d_idx_vec.empty()) {
  //     continue;
  //   }
  //   problem.SetParameterBlockConstant(pnt3d.data());
  // }

  // ceres::Solve(options, &problem, &summary);
  // std::cout << summary.BriefReport() << std::endl;
}
} // namespace Ortho
#endif
