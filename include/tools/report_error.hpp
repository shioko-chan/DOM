#ifndef ORTHO_REPORT_ERROR_HPP
#define ORTHO_REPORT_ERROR_HPP

#include <exception>
#include <filesystem>
#include <format>
#include <string>
#include <string_view>
#include <typeinfo>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "tools/log.hpp"

namespace std {
template <>
struct formatter<OrtErrorCode> : formatter<std::string_view> { // NOLINT

  auto format(OrtErrorCode code, format_context& ctx) const {
    string_view name;
    switch(code) {
      case ORT_OK:
        name = "ORT_OK";
        break;
      case ORT_FAIL:
        name = "ORT_FAIL";
        break;
      case ORT_INVALID_ARGUMENT:
        name = "ORT_INVALID_ARGUMENT";
        break;
      case ORT_NO_SUCHFILE:
        name = "ORT_NO_SUCHFILE";
        break;
      case ORT_NO_MODEL:
        name = "ORT_NO_MODEL";
        break;
      case ORT_ENGINE_ERROR:
        name = "ORT_ENGINE_ERROR";
        break;
      case ORT_RUNTIME_EXCEPTION:
        name = "ORT_RUNTIME_EXCEPTION";
        break;
      case ORT_INVALID_PROTOBUF:
        name = "ORT_INVALID_PROTOBUF";
        break;
      case ORT_MODEL_LOADED:
        name = "ORT_MODEL_LOADED";
        break;
      case ORT_NOT_IMPLEMENTED:
        name = "ORT_NOT_IMPLEMENTED";
        break;
      case ORT_INVALID_GRAPH:
        name = "ORT_INVALID_GRAPH";
        break;
      case ORT_EP_FAIL:
        name = "ORT_EP_FAIL";
        break;
      default:
        name = "Unknown error code";
        break;
    }
    return formatter<std::string_view>::format(name, ctx);
  }
};
} // namespace std

namespace Ortho {

namespace fs = std::filesystem;

inline auto format_exception(const std::exception& exception) noexcept -> std::string {
  return std::format(
      "\nStd Exception occurred:\n"
      "  Type         : {}\n"
      "  Message      : {}\n",
      typeid(exception).name(),
      exception.what());
}

inline auto format_exception(const cv::Exception& exception) noexcept -> std::string {
  return std::format(
      "\nOpenCV Exception occurred:\n"
      "  Error Code   : {}\n"
      "  Description  : {}\n"
      "  Function     : {}\n"
      "  File         : {}\n"
      "  Line         : {}\n"
      "  Full Message : {}\n",
      exception.code,
      exception.err,
      exception.func,
      exception.file,
      exception.line,
      exception.what());
}

inline auto format_exception(const fs::filesystem_error& exception) noexcept -> std::string {
  return std::format(
      "\nFilesystem Exception occurred:\n"
      "  What         : {}\n"
      "  Path1        : {}\n"
      "  Path2        : {}\n"
      "  Error Code   : {}\n",
      exception.what(),
      exception.path1().string(),
      exception.path2().string(),
      exception.code().message());
}

inline auto format_exception(const Ort::Exception& exception) noexcept -> std::string {
  return std::format(
      "\nONNX Runtime Exception occurred:\n"
      "  Error Code   : {}\n"
      "  Message      : {}\n",
      exception.GetOrtErrorCode(),
      exception.what());
}

template <typename Exception, typename... Args>
  requires std::derived_from<Exception, std::exception>
[[noreturn]] void report_error(const Exception& exception, std::string_view error_msg, const Args&... args) noexcept {
  THIS_LOG_ERROR("{}\nDetail: {}", std::vformat(error_msg, std::make_format_args(args...)), format_exception(exception));
  std::terminate();
}

template <typename... Args>
[[noreturn]] void report_error(std::string_view error_msg, const Args&... args) noexcept {
  THIS_LOG_ERROR(error_msg, args...);
  std::terminate();
}

} // namespace Ortho

#endif