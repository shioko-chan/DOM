#ifndef ORTHO_ONNXRUNTIME_HPP
#define ORTHO_ONNXRUNTIME_HPP

#include <algorithm>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "tools/debug.hpp"
#include "tools/report_error.hpp"

namespace Ortho {

namespace fs = std::filesystem;

using OrtValues = std::vector<Ort::Value>;

static auto ort_env() -> Ort::Env& {
  static Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "ONNXRUNTIME");
  return env;
}

class InferEnv {
private:

  std::unique_ptr<Ort::Session> session;
  OrtValues                     inputs;
  std::vector<std::string>      input_names, output_names;
  std::vector<const char*>      input_names_cstr, output_names_cstr;

public:

  InferEnv() = delete;

  InferEnv(
      const std::string&    name,
      const std::string&    model_path,
      const OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_ERROR) {
    fs::path model_path_(model_path);
    if(!fs::exists(model_path_)) {
      throw std::runtime_error("Error: " + model_path_.string() + " does not exist");
    }
    Ort::SessionOptions    session_options;
    OrtCUDAProviderOptions provider_options;
    provider_options.device_id                 = 0;
    provider_options.arena_extend_strategy     = 0; // kNextPowerOfTwo
    provider_options.do_copy_in_default_stream = 0;
    provider_options.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchHeuristic;
    session_options.AppendExecutionProvider_CUDA(provider_options);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetLogSeverityLevel(log_level);
    session_options.SetLogId(name.c_str());
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(ort_env(), model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    for(int i = 0; i < session->GetInputCount(); ++i) {
      inputs.emplace_back(nullptr);
      input_names.emplace_back(session->GetInputNameAllocated(i, allocator).get());
    }
    for(int i = 0; i < session->GetOutputCount(); ++i) {
      output_names.emplace_back(session->GetOutputNameAllocated(i, allocator).get());
    }
    std::ranges::transform(input_names, std::back_inserter(input_names_cstr), [](const std::string& name) noexcept {
      return name.c_str();
    });
    std::ranges::transform(output_names, std::back_inserter(output_names_cstr), [](const std::string& name) noexcept {
      return name.c_str();
    });
  }

  template <typename T>
    requires std::is_arithmetic_v<T>
  void set_input(const std::string& name, std::vector<T>& input, const std::vector<int64_t>& shape) {
    size_t idx  = std::ranges::find(input_names, name) - input_names.begin();
    inputs[idx] = std::move(Ort::Value::CreateTensor<T>(
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeCPUInput),
        input.data(),
        input.size(),
        shape.data(),
        shape.size()));
  }

  auto infer() noexcept -> OrtValues {
    try {
      return session->Run(
          Ort::RunOptions{nullptr},
          input_names_cstr.data(),
          inputs.data(),
          input_names.size(),
          output_names_cstr.data(),
          output_names.size());
    } catch(const Ort::Exception& exception) {
      report_error(exception, "An error occurred during inference");
    }
  }

  [[nodiscard]] auto get_input_names() const noexcept -> const std::vector<std::string>& { return input_names; }

  [[nodiscard]] auto get_output_names() const noexcept -> const std::vector<std::string>& { return output_names; }

  [[nodiscard]] auto get_output_index(const std::string& name) const noexcept -> size_t {
    THIS_ASSERTION_SHOULD_NEQ(
        static_cast<size_t>(std::ranges::find(output_names, name) - output_names.begin()),
        output_names.size(),
        "Input key of tensor name did not found.");
    return std::ranges::find(output_names, name) - output_names.begin();
  }
};
} // namespace Ortho
#endif