#ifndef ORTHO_LOG_HPP
#define ORTHO_LOG_HPP

#include <format>
#include <iostream>
#include <mutex>
#include <string_view>

#include "tools/ansi.hpp"

namespace Ortho {

inline auto stream_mtx() -> std::mutex& {
  static std::mutex stream_mtx;
  return stream_mtx;
}

template <typename... Args>
void log(std::ostream& ostream, const std::string& prefix, std::string_view format, const Args&... args) {
  std::lock_guard<std::mutex> lock(stream_mtx());
  ostream << prefix << std::vformat(format, std::make_format_args(args...)) << ansi::RESET << "\n";
}

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO) || defined(LOGLEVEL_WARN) || defined(LOGLEVEL_ERROR)
template <typename... Args>
void log_error(std::string_view format, const Args&... args) {
  auto prefix = std::format("{}{}[ERROR] ", ansi::RED, ansi::BOLD);
  log(std::cerr, prefix, format, args...);
}

  #define THIS_LOG_ERROR(fmt, ...) Ortho::log_error(fmt __VA_OPT__(, ) __VA_ARGS__)
#else
  #define THIS_LOG_ERROR(...)
#endif

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO) || defined(LOGLEVEL_WARN)
template <typename... Args>
void log_warn(std::string_view format, const Args&... args) {
  auto prefix = std::format("{}{}[WARN] ", ansi::YELLOW, ansi::BOLD);
  log(std::cerr, prefix, format, args...);
}

  #define THIS_LOG_WARN(fmt, ...) Ortho::log_warn(fmt __VA_OPT__(, ) __VA_ARGS__)
#else
  #define THIS_LOG_WARN(...)
#endif

#if defined(LOGLEVEL_DEBUG) || defined(LOGLEVEL_INFO)
template <typename... Args>
void log_info(std::string_view format, const Args&... args) {
  auto prefix = std::format("{}[INFO] ", ansi::GREEN);
  log(std::cout, prefix, format, args...);
}

  #define THIS_LOG_INFO(fmt, ...) Ortho::log_info(fmt __VA_OPT__(, ) __VA_ARGS__)
#else
  #define THIS_LOG_INFO(...)
#endif

#if defined(LOGLEVEL_DEBUG)
template <typename... Args>
void log_debug(std::string_view format, const Args&... args) {
  auto prefix = std::format("{}[DEBUG] ", ansi::BLUE);
  log(std::cout, prefix, format, args...);
}

  #define THIS_LOG_DEBUG(fmt, ...) Ortho::log_debug(fmt __VA_OPT__(, ) __VA_ARGS__)
#else
  #define THIS_LOG_DEBUG(...)
#endif

template <typename... Args>
void message(std::string_view format, const Args&... args) {
  auto prefix = std::format("{}{}", ansi::GREEN, ansi::BOLD);
  log(std::cout, prefix, format, args...);
}

#define THIS_MESSAGE(fmt, ...) Ortho::message(fmt __VA_OPT__(, ) __VA_ARGS__)

} // namespace Ortho
#endif