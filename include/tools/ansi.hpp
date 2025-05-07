#ifndef SKYMERGE_ANSI_COLOR_HPP
#define SKYMERGE_ANSI_COLOR_HPP

#include <string_view>

namespace SkyMerge::ansi {
constexpr std::string_view RESET      = "\033[0m";
constexpr std::string_view BOLD       = "\033[1m";
constexpr std::string_view DIM        = "\033[2m";
constexpr std::string_view ITALIC     = "\033[3m";
constexpr std::string_view UNDERLINE  = "\033[4m";
constexpr std::string_view BLINK      = "\033[5m";
constexpr std::string_view REVERSE    = "\033[7m";
constexpr std::string_view HIDDEN     = "\033[8m";
constexpr std::string_view STRIKE     = "\033[9m";
constexpr std::string_view BLACK      = "\033[30m";
constexpr std::string_view RED        = "\033[31m";
constexpr std::string_view GREEN      = "\033[32m";
constexpr std::string_view YELLOW     = "\033[33m";
constexpr std::string_view BLUE       = "\033[34m";
constexpr std::string_view MAGENTA    = "\033[35m";
constexpr std::string_view CYAN       = "\033[36m";
constexpr std::string_view WHITE      = "\033[37m";
constexpr std::string_view GRAY       = "\033[90m";
constexpr std::string_view BRIGHT_RED = "\033[91m";
} // namespace SkyMerge::ansi
#endif