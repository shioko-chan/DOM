#ifndef SKYMERGE_BA_HPP
#define SKYMERGE_BA_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>
#include <ranges>
#include <thread>

#include <Eigen/Dense>

#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/loss_function.h>
#include <ceres/ordered_groups.h>
#include <ceres/rotation.h>
#include <ceres/solver.h>
#include <ceres/types.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

namespace SkyMerge {

class BA {
public:

  static void ba(ImgsData& imgs_data, TrackPointVec* const res, double huber_threshold = 3.0) noexcept {
    if(res->empty() || imgs_data.empty()) {
      THIS_LOG_WARN("[BA] Empty input data");
      return;
    }
    THIS_LOG_INFO("[BA] Starting Bundle Adjustment");

    auto imgs_data_filtered =
        imgs_data | std::views::filter([](const auto& img_data) noexcept { return img_data.is_valid(); });

    ceres::Problem         problem;
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    options.num_threads        = static_cast<int>(std::thread::hardware_concurrency());
    options.max_num_iterations = 1000;

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
        auto& img_data           = imgs_data[pnt2d_idx.img_idx];
        const auto& [ob_x, ob_y] = img_data.get_kpnts()[pnt2d_idx.pnt_idx];
        problem.AddResidualBlock(
            ReprojectionError::create(ob_x, ob_y).release(),
            (huber_threshold > 0 ? std::make_unique<ceres::HuberLoss>(huber_threshold).release() : nullptr),
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
      THIS_LOG_INFO("[BA] Step 1: Optimizing camera extrinsic parameters");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
        set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
      }
      set_parameter_block_constant(problem, imgs_data.camera_array_raw());
      set_parameter_block_constant(problem, imgs_data.distort_array_raw());
      for(const auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_constant(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 1);
    }

    // Secondly, optimize the intrinsic
    // Make [R, t, pnt3d] constant
    //      [K, d] variable
    {
      THIS_LOG_INFO("[BA] Step 2: Optimizing camera intrinsic parameters");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_constant(problem, img_data.A_w2c_array_raw());
        set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
      }
      set_parameter_block_variable(problem, imgs_data.camera_array_raw());
      set_parameter_block_variable(problem, imgs_data.distort_array_raw());
      for(const auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_constant(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 2);
    }

    // Thirdly, optimize the 3d points
    // Make [R, t, K, d] constant
    //      [pnt3d] variable
    {
      THIS_LOG_INFO("[BA] Step 3: Optimizing 3D points");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_constant(problem, img_data.A_w2c_array_raw());
        set_parameter_block_constant(problem, img_data.t_w2c_array_raw());
      }
      set_parameter_block_constant(problem, imgs_data.camera_array_raw());
      set_parameter_block_constant(problem, imgs_data.distort_array_raw());
      for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_variable(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 3);
    }

    // Fourthly, optimize the 3d points and extrinsic
    // Make [K, d] constant
    //      [pnt3d, R, t] variable
    {
      THIS_LOG_INFO("[BA] Step 4: Optimizing 3D points and camera extrinsic parameters");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
        set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
      }
      set_parameter_block_constant(problem, imgs_data.camera_array_raw());
      set_parameter_block_constant(problem, imgs_data.distort_array_raw());
      for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_variable(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 4);
    }

    // Finally, optimize all together
    {
      THIS_LOG_INFO("[BA] Step 5: Optimizing all parameters together");
      for(auto& img_data : imgs_data_filtered) {
        set_parameter_block_variable(problem, img_data.A_w2c_array_raw());
        set_parameter_block_variable(problem, img_data.t_w2c_array_raw());
      }
      set_parameter_block_variable(problem, imgs_data.camera_array_raw());
      set_parameter_block_variable(problem, imgs_data.distort_array_raw());
      for(auto& [pnt3d, pnt2d_idx_vec] : *res) {
        set_parameter_block_variable(problem, pnt3d);
      }
      ceres::Solve(options, &problem, &summary);
      check_summary(summary, 5);
    }
    THIS_LOG_INFO("[BA] Bundle Adjustment completed");

#ifdef LOGLEVEL_INFO
    std::vector<double> residuals;
    problem.Evaluate(ceres::Problem::EvaluateOptions{}, nullptr, &residuals, nullptr, nullptr);
    auto len  = residuals.size() / 2;
    auto view = std::views::iota(0UL, len) | std::views::transform([&residuals](const auto& idx) noexcept {
                  return std::hypot(residuals[idx * 2], residuals[(idx * 2) + 1]);
                });
    std::vector<double> pixel_loss(view.begin(), view.end());
    auto                minmax = std::ranges::minmax_element(pixel_loss);
    THIS_LOG_INFO("=============");
    THIS_LOG_INFO(
        "[BA] Average residual block norm: {}",
        std::accumulate(pixel_loss.begin(), pixel_loss.end(), 0.0) / static_cast<double>(len));
    THIS_LOG_INFO("[BA] Min residual block norm: {}", *minmax.min);
    THIS_LOG_INFO("[BA] Max residual block norm: {}", *minmax.max);
    std::ranges::sort(pixel_loss);
    THIS_LOG_INFO("[BA] Middle residual block norm: {}", pixel_loss[len / 2]);
    THIS_LOG_INFO("=============");
#endif
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
      auto         pixel = world2pixel(axisangle, translation, camera, distort, point_3d);
      std::span<T> resid{residuals, 2};
      T            predict_x = pixel(0);
      T            predict_y = pixel(1);
      resid[0]               = predict_x - observe_x;
      resid[1]               = predict_y - observe_y;
      return true;
    }

    static auto create(double observe_x, double observe_y) noexcept -> std::unique_ptr<ceres::CostFunction> {
      auto error_ptr = std::make_unique<ReprojectionError>(observe_x, observe_y);
      return std::make_unique<ceres::AutoDiffCostFunction<
          ReprojectionError,
          ResidualBlockSize,
          RotateAxisAngleSize,
          TranslateArraySize,
          CameraArraySize,
          DistortArraySize,
          Point3DSize>>(error_ptr.release());
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
      THIS_LOG_INFO("[BA] Step {} completed: {}", step, summary.FullReport());
    } else {
      THIS_LOG_ERROR("[BA] Step {} failed: {}", step, summary.FullReport());
    }
  }
};
} // namespace SkyMerge
#endif
