#ifndef MATCHPAIR_HPP
#define MATCHPAIR_HPP

#include <iostream>
#include <utility>

#include "types.hpp"

namespace Ortho {

class MatchPair {
public:

  int           first, second;
  bool          valid = false;
  cv::Mat       M;
  Point<float>  lhs_pnts, rhs_pnts;
  Point3<float> pnts3d;
  Matches       matches;

  MatchPair() = default;

  MatchPair(int first, int second) : first(first), second(second) {}

  bool operator==(const MatchPair& rhs) const { return first == rhs.first && second == rhs.second; }

  bool operator!=(const MatchPair& rhs) const { return !(*this == rhs); }

  bool operator<(const MatchPair& rhs) const {
    return first < rhs.first || (first == rhs.first && second < rhs.second);
  }

  friend std::ostream& operator<<(std::ostream& os, const MatchPair& pair) {
    os << pair.first << " " << pair.second << "\n";
    return os;
  }
};

using MatchPairs = std::vector<MatchPair>;

} // namespace Ortho

#endif