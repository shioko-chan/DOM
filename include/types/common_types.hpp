#ifndef ORTHO_COMMON_TYPES_HPP
#define ORTHO_COMMON_TYPES_HPP

#include <array>
#include <mutex>
#include <ostream>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "types/hash.hpp"

namespace Ortho {

using RotateQArray   = std::array<double, 4>;
using CameraArray    = std::array<double, 4>;
using DistortArray   = std::array<double, 6>;
using TranslateArray = std::array<double, 3>;
template <typename T>
using USets = std::vector<std::unordered_set<T>>;

template <typename T>
using Sets = std::vector<std::set<T>>;

struct alignas(32) Match {
  size_t lhs, rhs;
  double score;
};

using Matches = std::vector<Match>;

struct alignas(16) PointIdx {
  int    img_idx;
  size_t pnt_idx;

  auto operator<=>(const PointIdx&) const        = default;
  auto operator==(const PointIdx&) const -> bool = default;

  friend auto operator<<(std::ostream& ostream, const PointIdx& idx) -> std::ostream& {
    return ostream << "(" << idx.img_idx << ", " << idx.pnt_idx << ")";
  }
};

struct PointIdxHasher {
  auto operator()(const PointIdx& point_idx) const noexcept -> uint64_t {
    return hash(point_idx.img_idx, point_idx.pnt_idx);
  }
};

template <typename T>
using PointIdxUMap = std::unordered_map<PointIdx, T, PointIdxHasher>;

template <typename T>
using PointIdxUMapRev = std::unordered_map<T, PointIdx>;

using PointIdxUSet = std::unordered_set<PointIdx, PointIdxHasher>;

using PointIdxs = std::vector<PointIdx>;

using Lock = std::unique_lock<std::mutex>;

} // namespace Ortho
#endif
