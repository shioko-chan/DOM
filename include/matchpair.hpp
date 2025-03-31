#ifndef MATCHPAIR_HPP
#define MATCHPAIR_HPP

#include <iostream>
#include <utility>

namespace Ortho {

class MatchPair {
public:

  int first, second;

  bool valid = false;

  cv::Mat M;

  MatchPair() = default;

  MatchPair(int first, int second) : first(first), second(second), valid(true) {}

  friend bool operator==(const MatchPair& lhs, const MatchPair& rhs) {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }

  friend bool operator!=(const MatchPair& lhs, const MatchPair& rhs) { return !(lhs == rhs); }

  friend bool operator<(const MatchPair& lhs, const MatchPair& rhs) {
    return lhs.first < rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
  }

  friend std::ostream& operator<<(std::ostream& os, const MatchPair& pair) {
    os << pair.first << " " << pair.second << "\n";
    return os;
  }
};

using MatchPairs = std::vector<MatchPair>;

} // namespace Ortho

#endif