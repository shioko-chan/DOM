#ifndef SKYMERGE_TRACKS_HPP
#define SKYMERGE_TRACKS_HPP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <limits>
#include <memory>
#include <queue>
#include <unordered_map>
#include <vector>

#include "tools/debug.hpp"
#include "types.hpp"

namespace SkyMerge {

using WeightMap     = PointIdxUMap<PointIdxUMap<double>>;
using PointIdxPair  = std::pair<PointIdx, PointIdx>;
using PointIdxPairs = std::vector<PointIdxPair>;

struct alignas(64) EdgeWithWeight {
  PointIdx u, v;
  double   weight;

  friend auto operator<<(std::ostream& ostream, const EdgeWithWeight& edge) -> std::ostream& {
    ostream << "[" << edge.u << ", " << edge.v << ", " << edge.weight << "]" << '\n';
    return ostream;
  }
};

inline auto operator<<(std::ostream& ostream, const PointIdxPairs& min_cut) -> std::ostream& {
  ostream << "Minimum Cut Edges (" << min_cut.size() << " edges): " << '\n';
  for(const auto& [u, v] : min_cut) {
    ostream << u << "--" << v << '\n';
  }
  return ostream;
}

inline auto operator<<(std::ostream& ostream, const WeightMap& map) -> std::ostream& {
  ostream << "WeightMap: " << '\n';
  for(const auto& [u, next] : map) {
    for(const auto& [v, w] : next) {
      ostream << u << "--" << v << '\n';
    }
  }
  return ostream;
}

inline auto operator<<(WeightMap& map, const EdgeWithWeight& edge) -> WeightMap& {
  map[edge.u][edge.v] = edge.weight;
  return map;
}

template <typename Func>
  requires std::invocable<Func, const PointIdx&> || std::invocable<Func, const EdgeWithWeight&>
void bfs(const WeightMap& pnt_map, const PointIdx& start, Func visit, PointIdxUSet* arranged = nullptr) {
  std::unique_ptr<PointIdxUSet> ptr{};
  if(!arranged) {
    ptr      = std::make_unique<PointIdxUSet>();
    arranged = ptr.get();
  }
  std::queue<PointIdx> queue;
  if(arranged->contains(start)) {
    return;
  }
  queue.push(start);
  arranged->insert(start);
  while(!queue.empty()) {
    PointIdx current = queue.front();
    queue.pop();
    if constexpr(std::invocable<Func, const PointIdx&>) {
      visit(current);
    }
    const auto& neighbors = pnt_map.find(current);
    if(neighbors == pnt_map.end()) {
      continue;
    }
    for(const auto& [next, weight] : neighbors->second) {
      if(!arranged->contains(next)) {
        queue.push(next);
        arranged->insert(next);
      }
      if constexpr(std::invocable<Func, const EdgeWithWeight&>) {
        visit({current, next, weight});
      }
    }
  }
}

class Dinic {
public:

  auto operator()(const WeightMap& pnt_map, const PointIdx& start, const PointIdx& end) -> PointIdxPairs {
    s = start, t = end;
    bfs(pnt_map, s, [this](const EdgeWithWeight& pair) noexcept { graph << pair; });
    double flow_sum = 0;
    while(true) {
      update_level();
      if(level[t] == 0) {
        break;
      }
      iter.clear();
      double flow{};
      while((flow = augment(s, t, std::numeric_limits<double>::max())) > 0) {
        flow_sum += flow;
      }
    }
    PointIdxPairs ans;
    for(const auto& [u, next] : graph) {
      if(level[u] == 0) {
        continue;
      }
      for(const auto& [v, _] : next) {
        if(level[v] == 0) {
          ans.emplace_back(u, v);
        }
      }
    }
    return ans;
  }

private:

  WeightMap graph;
  using Iter = typename PointIdxUMap<double>::iterator;
  PointIdxUMap<Iter> iter;
  PointIdxUMap<int>  level;
  PointIdx           s{}, t{};

  void update_level() {
    level.clear();
    std::queue<PointIdx> queue;
    level[s] = 1;
    queue.push(s);
    while(!queue.empty()) {
      PointIdx current = queue.front();
      queue.pop();
      for(const auto& [next, cap] : graph[current]) {
        if(cap > 0 && level[next] == 0) {
          level[next] = level[current] + 1;
          queue.push(next);
        }
      }
    }
  }

  auto augment(PointIdx start, PointIdx end, double flow) -> double {
    if(start == end) {
      return flow;
    }
    if(!iter.contains(start)) {
      iter.emplace(start, graph[start].begin());
    }
    for(Iter& i = iter[start]; i != graph[start].end(); ++i) {
      PointIdx next = i->first;
      double&  cap  = i->second;
      if(cap > 0 && level[start] < level[next]) {
        double d_flow = augment(next, end, std::min(flow, cap));
        if(d_flow > 0) {
          cap -= d_flow;
          graph[next][start] += d_flow;
          return d_flow;
        }
      }
    }
    return 0;
  }
};

struct alignas(64) TracksMaintainer {
public:

  void append_match(PointIdx idx0, PointIdx idx1, double score) {
    THIS_ASSERTION_SHOULD_NEQ(idx0.img_idx, idx1.img_idx);
    double weight     = score2weight(score);
    double max_weight = std::max(pnt_map[idx0][idx1], weight);
    pnt_map << EdgeWithWeight{.u = idx0, .v = idx1, .weight = max_weight};
    pnt_map << EdgeWithWeight{.u = idx1, .v = idx0, .weight = max_weight};
    check_and_remove_weak_match(idx0);
  }

  auto get_tracks() const -> Tracks {
    PointIdxUSet visited;
    Tracks       result;
    for(const auto& [pnt, _] : pnt_map) {
      if(visited.contains(pnt)) {
        continue;
      }
      PointIdxs res;
      bfs(pnt, [&res](const PointIdx& idx) noexcept { res.push_back(idx); }, &visited);
      result.push_back(res);
    }
    return result;
  }

private:

  WeightMap pnt_map;

  static auto score2weight(double score) -> double { return -std::log(std::max(1e-6, 1.0 - score)); }

  template <typename Func>
  void bfs(const PointIdx& start, Func visit, PointIdxUSet* arranged = nullptr) const {
    SkyMerge::bfs(pnt_map, start, visit, arranged);
  }

  auto find_conflicts(const PointIdx& start) const -> PointIdxPairs {
    std::unordered_map<int, PointIdx> img_pnt_match;
    PointIdxPairs                     conflicts;
    bfs(start, [&img_pnt_match, &conflicts](const PointIdx& idx) noexcept {
      if(img_pnt_match.contains(idx.img_idx)) {
        conflicts.emplace_back(img_pnt_match[idx.img_idx], idx);
      } else {
        img_pnt_match.emplace(idx.img_idx, idx);
      }
    });
    return conflicts;
  }

  static auto find_min_cut(const WeightMap& pnt_map, const PointIdx& start, const PointIdx& end) -> PointIdxPairs {
    return Dinic{}(pnt_map, start, end);
  }

  void check_and_remove_weak_match(const PointIdx& node) {
    auto conflicts = find_conflicts(node);
    for(const auto& [s, t] : conflicts) {
      auto pointidx_pairs = find_min_cut(pnt_map, s, t);
      for(const auto& pair : pointidx_pairs) {
        pnt_map[pair.first].erase(pair.second);
        pnt_map[pair.second].erase(pair.first);
      }
    }
    auto conf = find_conflicts(node);
    THIS_ASSERTION_SHOULD_TRUE(conf.empty());
  }
};

} // namespace SkyMerge
#endif
