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
struct BaReprojectionError {
public:

  BaReprojectionError(Point<double> img_pnt) : pnt2d(std::move(img_pnt)) {}

  template <typename T>
  bool operator()(const T* const q, const T* const t, const T* const c, const T* const pnt3d, T* residuals) const {
    T p0[3];
    for(size_t i = 0; i < 3; ++i) {
      p0[i] = pnt3d[i] + t[i];
    }
    T p1[3];
    ceres::QuaternionRotatePoint(q, p0, p1);
    T p1_z = p1[2];
    if(ceres::abs(p1_z) < 1e-6) {
      return false;
    }
    residuals[0] = c[0] * p1[0] / p1_z + c[2] - T(pnt2d.x);
    residuals[1] = c[1] * p1[1] / p1_z + c[3] - T(pnt2d.y);
    return true;
  }

  static ceres::CostFunction* create(const Point<float>& img_pnt) {
    return new ceres::AutoDiffCostFunction<BaReprojectionError, 2, 4, 3, 4, 3>(
        new BaReprojectionError(Point<double>(img_pnt)));
  }

private:

  Point<double> pnt2d;
};

void ba(ImgsData& imgs_data, auto& res) {
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
    add_parameter_block(img_data.Q_proj_array_raw(), new ceres::QuaternionManifold);
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
      auto&                img_data = imgs_data[pnt2d_idx.img_idx];
      ceres::CostFunction* cost     = BaReprojectionError::create(img_data.get_kpnts().get(pnt2d_idx.pnt_idx));
      problem.AddResidualBlock(
          cost,
          new ceres::HuberLoss(1.0),
          img_data.Q_proj_array_raw().data(),
          img_data.t_proj_array_raw().data(),
          img_data.camera_array_raw().data(),
          pnt3d.data());
    }
  }
  ceres::Solver::Options options;
  options.num_threads                  = std::thread::hardware_concurrency();
  options.linear_solver_type           = ceres::SPARSE_SCHUR;
  options.check_gradients              = false;
  options.minimizer_progress_to_stdout = false;
  options.max_num_iterations           = 2000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

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
