#ifndef MATCHPAIR_HPP
#define MATCHPAIR_HPP

#include <iostream>
#include <tuple>

#include "types/common_types.hpp"
#include "types/cv_alias.hpp"

namespace Ortho {

class MatchPair {
public:

  int            first{}, second{};
  bool           valid = false;
  cv::Mat        M;
  Point<double>  lhs_pnts, rhs_pnts;
  Point3<double> pnts3d;
  Matches        matches;

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

} // namespace Ortho

#endif