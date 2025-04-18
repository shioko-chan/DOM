#ifndef ORTHO_DSU_HPP
#define ORTHO_DSU_HPP

#include <algorithm>
#include <numeric>
#include <ranges>
#include <unordered_set>
#include <vector>

#include "types.hpp"

namespace Ortho {

struct DisjointSetUnion {
public:

  void append_match(PointIdx idx1, PointIdx idx2) {
    size_t internal_idx1 = append_pntidx_if_not_exist(idx1), internal_idx2 = append_pntidx_if_not_exist(idx2);
    unite_(internal_idx1, internal_idx2);
  }

  std::vector<PointIdxs> get_pntidxs() {
    std::unordered_map<size_t, PointIdxs> result;
    for(size_t i = 0; i < parent.size(); ++i) {
      size_t root = find_(i);
      result[root].push_back(inner_outer_map[i]);
    }
    auto v = result | std::views::transform([](auto&& pair) { return std::move(pair.second); });
    return {v.begin(), v.end()};
  }

private:

  std::vector<size_t>     parent, size;
  PointIdxUMap<size_t>    outer_inner_map;
  PointIdxUMapRev<size_t> inner_outer_map;

  size_t append_pntidx_if_not_exist(const PointIdx& idx) {
    auto it = outer_inner_map.find(idx);
    if(it != outer_inner_map.end()) {
      return it->second;
    }
    size_t internal_idx = parent.size();
    parent.push_back(internal_idx);
    size.push_back(1);
    outer_inner_map.emplace(idx, internal_idx);
    inner_outer_map.emplace(internal_idx, idx);
    return internal_idx;
  }

  size_t find_(size_t x) { return parent[x] == x ? x : parent[x] = find_(parent[x]); }

  void unite_(size_t x, size_t y) {
    x = find_(x), y = find_(y);
    if(x == y) {
      return;
    }
    if(size[x] < size[y]) {
      std::swap(x, y);
    }
    parent[y] = x;
    size[x] += size[y];
  }
};

} // namespace Ortho
#endif