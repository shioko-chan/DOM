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
#include "algo/tri.hpp"
#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/report_error.hpp"

namespace Ortho {

inline void ba(ImgsData& imgs_data, std::vector<TriRes>& res) noexcept { // NOLINT
  ceres::Problem         problem;
  ceres::Solver::Options options;
  options.num_threads                       = static_cast<int>(std::thread::hardware_concurrency());
  options.minimizer_progress_to_stdout      = true;
  options.max_num_iterations                = 2000;
  options.linear_solver_type                = ceres::SPARSE_SCHUR;
  options.check_gradients                   = false;
  options.gradient_check_relative_precision = 1e-2;
  options.trust_region_strategy_type        = ceres::LEVENBERG_MARQUARDT;
  ceres::Solver::Summary summary;

  for(auto& img_data : imgs_data) {
    try {
      add_parameter_block(problem, img_data.Q_w2c_array_raw(), new ceres::QuaternionManifold);
    } catch(const std::exception& e) {
      report_error(e, "Bad allocation");
    }
    add_parameter_block(problem, img_data.t_w2c_array_raw());
    add_parameter_block(problem, img_data.camera_array_raw());
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

  for(const auto& img_data : imgs_data) {
    set_parameter_block_constant(problem, img_data.camera_array_raw());
    // set_parameter_block_constant(problem, img_data.Q_w2c_array_raw());
  }
  ceres::Solve(options, &problem, &summary);
  THIS_MESSAGE("Step 1: {}", summary.BriefReport());

  //
  // Firstly, optimize the camera extrinsic
  // Make [K, pnt3d] constant
  //      [R, t] variable
  // {
  //   for(const auto& img_data : imgs_data) {
  //     set_parameter_block_constant(problem, img_data.camera_array_raw());
  //   }
  //   for(const auto& [pnt3d, pnt2d_idx_vec] : res) {
  //     if(pnt2d_idx_vec.empty()) {
  //       continue;
  //     }
  //     set_parameter_block_constant(problem, pnt3d);
  //   }
  //   ceres::Solve(options, &problem, &summary);
  //   THIS_MESSAGE("Step 1: {}", summary.BriefReport());
  // }

  //
  // Secondly, optimize the 3d points
  // Make [R, t, K] constant
  //      [pnt3d] variable
  // {
  //   for(const auto& img_data : imgs_data) {
  //     set_parameter_block_constant(problem, img_data.Q_w2c_array_raw());
  //     set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
  //   }
  //   for(auto& [pnt3d, pnt2d_idx_vec] : res) {
  //     if(pnt2d_idx_vec.empty()) {
  //       continue;
  //     }
  //     set_parameter_block_variable(problem, pnt3d);
  //   }
  //   ceres::Solve(options, &problem, &summary);
  //   THIS_MESSAGE("Step 2: {}", summary.BriefReport());
  // }

  //
  // Thirdly, optimize the 3d points and extrinsic
  // Make [K] constant
  //      [pnt3d, R, t] variable
  // {
  //   for(auto& img_data : imgs_data) {
  //     set_parameter_block_variable(problem, img_data.Q_w2c_array_raw());
  //     set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
  //   }
  //   ceres::Solve(options, &problem, &summary);
  //   THIS_MESSAGE("Step 3: {}", summary.BriefReport());
  // }

  //
  // Fourthly, optimize the intrinsic
  // Make [R, t, pnt3d] constant
  //      [K] variable
  // {
  //   for(auto& img_data : imgs_data) {
  //     set_parameter_block_constant(problem, img_data.Q_w2c_array_raw());
  //     set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
  //     set_parameter_block_variable(problem, img_data.camera_array_raw());
  //   }
  //   for(const auto& [pnt3d, pnt2d_idx_vec] : res) {
  //     if(pnt2d_idx_vec.empty()) {
  //       continue;
  //     }
  //     set_parameter_block_constant(problem, pnt3d);
  //   }
  //   ceres::Solve(options, &problem, &summary);
  //   THIS_MESSAGE("Step 4: {}", summary.BriefReport());
  // }

  //
  // Finally, optimize all together
  // {
  //   for(auto& img_data : imgs_data) {
  //     set_parameter_block_variable(problem, img_data.Q_w2c_array_raw());
  //     set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
  //   }
  //   for(auto& [pnt3d, pnt2d_idx_vec] : res) {
  //     if(pnt2d_idx_vec.empty()) {
  //       continue;
  //     }
  //     set_parameter_block_variable(problem, pnt3d);
  //   }
  //   ceres::Solve(options, &problem, &summary);
  //   THIS_MESSAGE("Step 5: {}", summary.BriefReport());
  // }
}
} // namespace Ortho
#endif
