#ifndef ORTHO_LOG_HPP
#define ORTHO_LOG_HPP

#include <concepts>
#include <format>
#include <iostream>
#include <mutex>
#include <string_view>

namespace Ortho {

#define RESET "\033[0m"
#define RED "\033[31m"
#define GREEN "\033[32m"
#define YELLOW "\033[33m"
#define BLUE "\033[34m"
#define BOLD "\033[1m"

static inline std::mutex stream_mtx;

template <typename... Args>
void log(std::ostream& ostream, const char* prefix, std::string_view format, Args&&... args) {
  std::lock_guard<std::mutex> lock(stream_mtx);
  ostream << prefix << std::vformat(format, std::make_format_args(args...)) << "\n";
}

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO) || defined(LOGLEVEL_WARN) || defined(LOGLEVEL_ERROR)
  #define LOG_ERROR(...) log(std::cerr, RESET RED BOLD "[LOG_ERROR] ", __VA_ARGS__)
#else
  #define LOG_ERROR(...)
#endif

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO) || defined(LOGLEVEL_WARN)
  #define LOG_WARN(...) log(std::cerr, YELLOW BOLD "[LOG_WARN] " RESET, __VA_ARGS__)
#else
  #define LOG_WARN(...)
#endif

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO)
  #define LOG_INFO(...) log(std::cout, GREEN "[LOG_INFO] " RESET, __VA_ARGS__)
#else
  #define LOG_INFO(...)
#endif

#if defined(LOGLEVEL_DEBUG)
  #define LOG_DEBUG(...) log(std::cout, BLUE "[LOG_DEBUG] " RESET, __VA_ARGS__)
#else
  #define LOG_DEBUG(...)
#endif

template <typename... Args>
void MESSAGE(std::string_view format, Args&&... args) {
  std::lock_guard<std::mutex> lock(stream_mtx);
  std::cout << GREEN          BOLD << std::vformat(format, std::make_format_args(args...)) << RESET "\n";
}

} // namespace Ortho
#endif