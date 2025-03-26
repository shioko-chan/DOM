#ifndef MATCHPAIR_HPP
#define MATCHPAIR_HPP

#include <iostream>
#include <utility>

namespace Ortho {

class MatchPair {
public:

  int first, second;

  MatchPair() = default;

  MatchPair(int first, int second) : first(first), second(second) {}

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

} // namespace Ortho

#endif