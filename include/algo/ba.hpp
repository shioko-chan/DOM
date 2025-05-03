#ifndef ORTHO_BA_HPP
#define ORTHO_BA_HPP

#include <cassert>
#include <exception>
#include <thread>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/types.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "algo/cost.hpp"
#include "ds/imgdata.hpp"
#include "tools/report_error.hpp"

namespace Ortho {

void ba(ImgsData& imgs_data, auto& res) noexcept {
  ceres::Problem problem;
  auto           set_bound = [&problem](auto& param, size_t idx, double lower_bound = 0.0, double upper_bound = 0.0) {
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
      add_parameter_block(problem, img_data.Q_w2c_array_raw(), new ceres::QuaternionManifold);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    add_parameter_block(problem, img_data.t_w2c_array_raw());
    add_parameter_block(problem, img_data.camera_array_raw());

    // set_parameter_block_constant(problem, img_data.Q_w2c_array_raw());
    // set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
    // set_parameter_block_constant(problem, img_data.camera_array_raw());
  }
  for(auto& [pnt3d, pnt2d_idx_vec] : res) {
    if(pnt2d_idx_vec.empty()) {
      continue;
    }
    add_parameter_block(problem, pnt3d);
    for(const auto& pnt2d_idx : pnt2d_idx_vec) {
      auto& img_data = imgs_data[pnt2d_idx.img_idx];
      try {
        problem.AddResidualBlock(
            ReprojectionError::create(img_data.get_kpnts().get(pnt2d_idx.pnt_idx)),
            new ceres::HuberLoss(1.0),
            img_data.Q_w2c_array_raw().data(),
            img_data.t_w2c_array_raw().data(),
            img_data.camera_array_raw().data(),
            pnt3d.data());
      } catch(const std::exception& e) {
        report_error(e, "Bad allocation");
      }
    }
  }
  ceres::Solver::Options options;
  options.num_threads                       = static_cast<int>(std::thread::hardware_concurrency());
  options.linear_solver_type                = ceres::SPARSE_SCHUR;
  options.check_gradients                   = true;
  options.gradient_check_relative_precision = 1e-4;
  options.minimizer_progress_to_stdout      = true;
  options.max_num_iterations                = 2000;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << '\n';
}
} // namespace Ortho
#endif
