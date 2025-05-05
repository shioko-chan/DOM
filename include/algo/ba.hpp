#ifndef ORTHO_BA_HPP
#define ORTHO_BA_HPP

#include <cassert>
#include <cmath>
#include <exception>
#include <ranges>
#include <thread>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/loss_function.h>
#include <ceres/rotation.h>
#include <ceres/types.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "algo/cost.hpp"
#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/report_error.hpp"

namespace Ortho {

inline void ba(ImgsData& imgs_data, TriResVec* res) noexcept {
  auto imgs_data_filtered =
      imgs_data | std::views::filter([](const auto& img_data) noexcept { return img_data.is_valid(); });
  std::erase_if(*res, [](const TriRes& tri_res) noexcept { return tri_res.pnt2d_idx_vec.size() < 2; });

  ceres::Problem         problem;
  ceres::Solver::Options options;
  ceres::Solver::Summary summary;

  options.num_threads        = static_cast<int>(std::thread::hardware_concurrency());
  options.max_num_iterations = 2000;

  options.minimizer_progress_to_stdout      = true;
  options.check_gradients                   = false;
  options.gradient_check_relative_precision = 1e-3;

  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type         = ceres::SPARSE_SCHUR;
  options.use_inner_iterations       = true;

  for(auto& img_data : imgs_data_filtered) {
    add_parameter_block(problem, img_data.A_w2c_array_raw());
    add_parameter_block(problem, img_data.t_w2c_array_raw());
    add_parameter_block(problem, img_data.camera_array_raw());
    add_parameter_block(problem, img_data.distort_array_raw());
  }

  for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
    add_parameter_block(problem, pnt3d);
    for(const auto& pnt2d_idx : pnt2d_idx_vec) {
      auto&             img_data = imgs_data[pnt2d_idx.img_idx];
      ceres::HuberLoss* loss{};
      try {
        loss = new ceres::HuberLoss(1.0);
      } catch(const std::exception& e) {
        report_error(e, "Bad allocation");
      }
      const auto& [ob_x, ob_y] = img_data.get_kpnts().get(pnt2d_idx.pnt_idx);
      problem.AddResidualBlock(
          ReprojectionError::create(ob_x, ob_y),
          loss,
          img_data.A_w2c_array_raw().data(),
          img_data.t_w2c_array_raw().data(),
          img_data.camera_array_raw().data(),
          img_data.distort_array_raw().data(),
          pnt3d.data());
    }
  }
  // Firstly, optimize the camera extrinsic
  // Make [K, d, pnt3d] constant
  //      [R, t] variable
  {
    for(const auto& img_data : imgs_data_filtered) {
      set_parameter_block_constant(problem, img_data.camera_array_raw());
      set_parameter_block_constant(problem, img_data.distort_array_raw());
    }
    for(const auto& [pnt3d, pnt2d_idx_vec] : *res) {
      set_parameter_block_constant(problem, pnt3d);
    }
    ceres::Solve(options, &problem, &summary);
    THIS_MESSAGE("Step 1: {} {}", summary.FullReport(), summary.BriefReport());
  }
  // Secondly, optimize the 3d points
  // Make [R, t, K, d] constant
  //      [pnt3d] variable
  {
    for(const auto& img_data : imgs_data_filtered) {
      set_parameter_block_constant(problem, img_data.A_w2c_array_raw());
      set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
    }
    for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
      set_parameter_block_variable(problem, pnt3d);
    }
    ceres::Solve(options, &problem, &summary);
    THIS_MESSAGE("Step 2: {}", summary.BriefReport());
  }
  // Thirdly, optimize the 3d points and extrinsic
  // Make [K, d] constant
  //      [pnt3d, R, t] variable
  {
    for(auto& img_data : imgs_data_filtered) {
      set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
      set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
    }
    ceres::Solve(options, &problem, &summary);
    THIS_MESSAGE("Step 3: {}", summary.BriefReport());
  }
  // Fourthly, optimize the intrinsic
  // Make [R, t, pnt3d] constant
  //      [K, d] variable
  {
    for(auto& img_data : imgs_data_filtered) {
      set_parameter_block_constant(problem, img_data.A_w2c_array_raw());
      set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
      set_parameter_block_variable(problem, img_data.camera_array_raw());
      set_parameter_block_variable(problem, img_data.distort_array_raw());
    }
    for(const auto& [pnt3d, pnt2d_idx_vec] : *res) {
      set_parameter_block_constant(problem, pnt3d);
    }
    ceres::Solve(options, &problem, &summary);
    THIS_MESSAGE("Step 4: {}", summary.BriefReport());
  }
  // Finally, optimize all together
  {
    for(auto& img_data : imgs_data_filtered) {
      set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
      set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
    }
    for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
      set_parameter_block_variable(problem, pnt3d);
    }
    ceres::Solve(options, &problem, &summary);
    THIS_MESSAGE("Step 5: {}", summary.BriefReport());
  }
}
} // namespace Ortho
#endif
