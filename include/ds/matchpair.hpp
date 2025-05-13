#ifndef MATCHPAIR_HPP
#define MATCHPAIR_HPP

#include <iostream>
#include <tuple>

#include <opencv2/opencv.hpp>

#include "types.hpp"

namespace SkyMerge {

class MatchPair {
public:

  int     first{}, second{};
  bool    valid = false;
  cv::Mat M;
  Matches matches;

  MatchPair() = default;

  MatchPair(int first, int second) noexcept : first(first), second(second) {}

  auto operator<=>(const MatchPair& other) const noexcept {
    return std::tie(first, second) <=> std::tie(other.first, other.second);
  }

  auto operator==(const MatchPair& other) const noexcept {
    return std::tie(first, second) == std::tie(other.first, other.second);
  }

  friend auto operator<<(std::ostream& ostream, const MatchPair& pair) noexcept -> std::ostream& {
    ostream << pair.first << " " << pair.second << "\n";
    return ostream;
  }
};

using MatchPairs = std::vector<MatchPair>;

} // namespace SkyMerge

#endif