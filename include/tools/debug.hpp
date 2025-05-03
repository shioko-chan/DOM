#ifndef ORTHO_DEBUG_HPP
#define ORTHO_DEBUG_HPP

#ifdef ENABLE_ASSERTION

  #include <concepts>
  #include <exception>
  #include <format>
  #include <source_location>
  #include <sstream>
  #include <string_view>

  #include "tools/log.hpp"

namespace Ortho {
template <typename T>
concept streamable = requires(T&& type, std::ostream& ostream) {
  { ostream << std::forward<T>(type) } -> std::convertible_to<std::ostream&>;
};

template <typename T>
concept equal_comparable = requires(const T& lhs, const T& rhs) {
  { lhs == rhs } -> std::convertible_to<bool>;
};

template <typename T>
concept nequal_comparable = requires(const T& lhs, const T& rhs) {
  { lhs != rhs } -> std::convertible_to<bool>;
};

template <typename T>
concept less_comparable = requires(const T& lhs, const T& rhs) {
  { lhs < rhs } -> std::convertible_to<bool>;
};

template <typename T>
concept leq_comparable = requires(const T& lhs, const T& rhs) {
  { lhs <= rhs } -> std::convertible_to<bool>;
};

constexpr std::array antonyms_str{
    std::array{"true", "false"},
    std::array{"less", "greater equal"},
    std::array{"less equal", "greater"},
    std::array{"equal", "not equal"}};

constexpr auto antonym(std::string_view word) -> std::string_view {
  for(const auto& pair : antonyms_str) {
    if(pair[0] == word) {
      return pair[1];
    }
    if(pair[1] == word) {
      return pair[0];
    }
  }
  return "";
}

[[noreturn]] inline void report_assertion_failure(
    std::string_view            custom_msg,
    std::string_view            expected,
    const std::source_location& loc,
    std::string_view            lhs_rhs_msg = "") {
  THIS_LOG_ERROR(
      "\nAssertion failed!\n"
      "  Expected: {}\n"
      "  Actual:   {}\n"
      "  {}"
      "  Location: {}:{}\n"
      "  Function: {}\n"
      "  Message:  {}",
      expected,
      antonym(expected),
      lhs_rhs_msg,
      loc.file_name(),
      loc.line(),
      loc.function_name(),
      custom_msg.empty() ? "(No additional message)" : custom_msg);
  std::terminate();
}

template <typename T>
  requires std::formattable<T, char> || streamable<T>
inline auto format_lhs_rhs(const T& lhs, const T& rhs) -> std::string {
  if constexpr(std::formattable<T, char>) {
    return std::format("Lhs: {}, rhs: {}\n", lhs, rhs);
  } else if constexpr(streamable<T>) {
    std::stringstream stream;
    stream << "Lhs: " << lhs << ", rhs " << rhs << "\n";
    return stream.str();
  }
}

template <typename T>
  requires equal_comparable<T>
void eq_assertion(
    const T&                    lhs,
    const T&                    rhs,
    std::string_view            custom_msg = "",
    const std::source_location& loc        = std::source_location::current()) {
  if(!(lhs == rhs)) {
    report_assertion_failure(custom_msg, "equal", loc, format_lhs_rhs(lhs, rhs));
  }
}

template <typename T>
  requires nequal_comparable<T>
void neq_assertion(
    const T&                    lhs,
    const T&                    rhs,
    std::string_view            custom_msg = "",
    const std::source_location& loc        = std::source_location::current()) {
  if(!(lhs != rhs)) {
    report_assertion_failure(custom_msg, "not equal", loc, format_lhs_rhs(lhs, rhs));
  }
}

template <typename T>
  requires less_comparable<T>
void les_assertion(
    const T&                    lhs,
    const T&                    rhs,
    std::string_view            custom_msg = "",
    const std::source_location& loc        = std::source_location::current()) {
  if(!(lhs < rhs)) {
    report_assertion_failure(custom_msg, "less", loc, format_lhs_rhs(lhs, rhs));
  }
}

template <typename T>
  requires leq_comparable<T>
void leq_assertion(
    const T&                    lhs,
    const T&                    rhs,
    std::string_view            custom_msg = "",
    const std::source_location& loc        = std::source_location::current()) {
  if(!(lhs <= rhs)) {
    report_assertion_failure(custom_msg, "less equal", loc, format_lhs_rhs(lhs, rhs));
  }
}

inline void true_assertion(
    bool                        expression,
    std::string_view            custom_msg = "",
    const std::source_location& loc        = std::source_location::current()) {
  if(!expression) {
    report_assertion_failure(custom_msg, "true", loc);
  }
}

inline void false_assertion(
    bool                        expression,
    std::string_view            custom_msg = "",
    const std::source_location& loc        = std::source_location::current()) {
  if(expression) {
    report_assertion_failure(custom_msg, "false", loc);
  }
}
} // namespace Ortho

  #define THIS_ASSERTION_SHOULD_EQ(lhs, rhs, ...) Ortho::eq_assertion(lhs, rhs __VA_OPT__(, ) __VA_ARGS__)
  #define THIS_ASSERTION_SHOULD_NEQ(lhs, rhs, ...) Ortho::neq_assertion(lhs, rhs __VA_OPT__(, ) __VA_ARGS__)
  #define THIS_ASSERTION_SHOULD_LES(lhs, rhs, ...) Ortho::les_assertion(lhs, rhs __VA_OPT__(, ) __VA_ARGS__)
  #define THIS_ASSERTION_SHOULD_LEQ(lhs, rhs, ...) Ortho::leq_assertion(lhs, rhs __VA_OPT__(, ) __VA_ARGS__)
  #define THIS_ASSERTION_SHOULD_TRUE(exp, ...) Ortho::true_assertion(exp __VA_OPT__(, ) __VA_ARGS__)
  #define THIS_ASSERTION_SHOULD_FALSE(exp, ...) Ortho::false_assertion(exp __VA_OPT__(, ) __VA_ARGS__)

#else

  #define THIS_ASSERTION_SHOULD_EQ(...)
  #define THIS_ASSERTION_SHOULD_NEQ(...)
  #define THIS_ASSERTION_SHOULD_LES(...)
  #define THIS_ASSERTION_SHOULD_LEQ(...)
  #define THIS_ASSERTION_SHOULD_TRUE(...)
  #define THIS_ASSERTION_SHOULD_FALSE(...)
#endif

#endif