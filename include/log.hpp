#ifndef ORTHO_LOG_HPP
#define ORTHO_LOG_HPP

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define BOLD "\033[1m"

#include <concepts>
#include <format>
#include <iostream>
#include <mutex>
#include <string_view>

namespace Ortho {

std::mutex cerr_mtx;

template <typename... Args>
void log(const char* prefix, std::string_view format, Args&&... args) {
  std::lock_guard<std::mutex> lock(cerr_mtx);
  std::cerr << prefix << std::vformat(format, std::make_format_args(args...)) << "\n";
}

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO) || defined(LOGLEVEL_WARN) || defined(LOGLEVEL_ERROR)
  #define ERROR(...) log(RESET RED BOLD "[ERROR] ", __VA_ARGS__)
#else
  #define ERROR(...)
#endif

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO) || defined(LOGLEVEL_WARN)
  #define WARN(...) log(YELLOW BOLD "[WARN] " RESET, __VA_ARGS__)
#else
  #define WARN(...)
#endif

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO)
  #define INFO(...) log(GREEN "[INFO] " RESET, __VA_ARGS__)
#else
  #define INFO(...)
#endif

#if defined(LOGLEVEL_DEBUG)
  #define DEBUG(...) log(BLUE "[DEBUG] " RESET, __VA_ARGS__)
#else
  #define DEBUG(...)
#endif

#define MESSAGE(...) log(GREEN BOLD "[MESSAGE] " RESET, __VA_ARGS__)

} // namespace Ortho
#endif