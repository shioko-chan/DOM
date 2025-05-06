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
#include <ceres/solver.h>
#include <ceres/types.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/report_error.hpp"
#include "tools/utility.hpp"

namespace Ortho {

class BA {
public:

  static void ba(ImgsData& imgs_data, TriResVec* res) noexcept {
    if(res->empty() || imgs_data.empty()) {
      THIS_LOG_WARN("No input!");
      return;
    }
    THIS_MESSAGE("Start Bundle Adjustment");
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

    add_parameter_block(problem, imgs_data.camera_array_raw());
    add_parameter_block(problem, imgs_data.distort_array_raw());
    for(auto& img_data : imgs_data_filtered) {
      add_parameter_block(problem, img_data.A_w2c_array_raw());
      add_parameter_block(problem, img_data.t_w2c_array_raw());
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
            imgs_data.camera_array_raw().data(),
            imgs_data.distort_array_raw().data(),
            pnt3d.data());
      }
    }
    // Firstly, optimize the camera extrinsic
    // Make [K, d, pnt3d] constant
    //      [R, t] variable
    {
      THIS_MESSAGE("Step 1 Info: Optimize camera extrinsic, keep intrinsic and 3D points fixed.");
      set_parameter_block_constant(problem, imgs_data.camera_array_raw());
      set_parameter_block_constant(problem, imgs_data.distort_array_raw());
      for(const auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_constant(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 1);
    }
    // Secondly, optimize the 3d points
    // Make [R, t, K, d] constant
    //      [pnt3d] variable
    {
      THIS_MESSAGE("Step 2 Info: Optimize 3D points, keep extrinsic and intrinsic fixed.");
      for(const auto& img_data : imgs_data_filtered) {
        set_parameter_block_constant(problem, img_data.A_w2c_array_raw());
        set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
      }
      for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_variable(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 2);
    }
    // Thirdly, optimize the 3d points and extrinsic
    // Make [K, d] constant
    //      [pnt3d, R, t] variable
    {
      THIS_MESSAGE("Step 3 Info: Optimize 3D points and extrinsic, keep intrinsic fixed.");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
        set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 3);
    }
    // Fourthly, optimize the intrinsic
    // Make [R, t, pnt3d] constant
    //      [K, d] variable
    {
      THIS_MESSAGE("Step 4 Info: Optimize intrinsic, keep extrinsic and 3D points fixed.");
      set_parameter_block_variable(problem, imgs_data.camera_array_raw());
      set_parameter_block_variable(problem, imgs_data.distort_array_raw());
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_constant(problem, img_data.A_w2c_array_raw());
        set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
      }
      for(const auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_constant(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 4);
    }
    // Finally, optimize all together
    {
      THIS_MESSAGE("Step 5 Info: Optimize all parameters together.");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
        set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
      }
      for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_variable(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 5);
    }
    THIS_MESSAGE("Bundle Adjustment Finished");
  }

private:

  struct alignas(16) ReprojectionError {
  public:

    explicit ReprojectionError(double observe_x, double observe_y) noexcept :
        observe_x(observe_x), observe_y(observe_y) {}

    template <typename T>
    auto operator()(
        const T* const axisangle,
        const T* const translation,
        const T* const camera,
        const T* const distort,
        const T* const point_3d,
        T*             residuals) const noexcept -> bool {
      auto         point = world2camera(axisangle, translation, point_3d);
      auto         pixel = camera2pixel(camera, distort, point.data());
      std::span<T> resid{residuals, 2};
      T            predict_x = pixel(0);
      T            predict_y = pixel(1);
      resid[0]               = predict_x - observe_x;
      resid[1]               = predict_y - observe_y;
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

  static void add_parameter_block(ceres::Problem& problem, auto& param, ceres::Manifold* manifold = nullptr) noexcept {
    if(manifold) {
      problem.AddParameterBlock(param.data(), static_cast<int>(param.size()), manifold);
    } else {
      problem.AddParameterBlock(param.data(), static_cast<int>(param.size()));
    }
  }

  static void set_parameter_block_constant(ceres::Problem& problem, const auto& param) noexcept {
    problem.SetParameterBlockConstant(param.data());
  }

  static void set_parameter_block_variable(ceres::Problem& problem, auto& param) noexcept {
    problem.SetParameterBlockVariable(param.data());
  }

  static void
  set_bound(ceres::Problem& problem, auto& param, size_t idx, double lower_bound = 0.0, double upper_bound = 0.0) noexcept {
    problem.SetParameterLowerBound(param.data(), idx, lower_bound);
    problem.SetParameterUpperBound(param.data(), idx, upper_bound);
  }

  static void set_bound_delta(ceres::Problem& problem, auto& param, size_t idx, double delta = 0.0) noexcept {
    double value = param[idx];
    set_bound(problem, param, idx, value - delta, value + delta);
  }

  static void set_bound_percentage(ceres::Problem& problem, auto& param, size_t idx, double percentage = 0.0) noexcept {
    double value = param[idx];
    percentage /= 100.0;
    set_bound(problem, param, idx, (1.0 - percentage) * value, (1.0 + percentage) * value);
  }

  static void check_summary(const ceres::Solver::Summary& summary, int step) {
    if(summary.IsSolutionUsable()) {
      THIS_MESSAGE("Step {}: {}", step, summary.BriefReport());
    } else {
      THIS_LOG_ERROR("Step {} failed: {}", step, summary.FullReport());
    }
  }
};
} // namespace Ortho
#endif
