#ifndef ORTHO_TRACKS_HPP
#define ORTHO_TRACKS_HPP

#include <algorithm>
#include <cassert>
#include <numeric>
#include <optional>
#include <queue>
#include <ranges>
#include <stack>
#include <unordered_set>
#include <vector>

#include "types.hpp"

namespace Ortho {

struct TracksMaintainer {
public:

  void append_match(PointIdx idx0, PointIdx idx1, float score) {
    assert(idx0.img_idx != idx1.img_idx);
    if(pnt_map.contains(idx0) && pnt_map[idx0].contains(idx1) && pnt_map[idx0][idx1] < score) {
      pnt_map[idx0][idx1] = score;
      pnt_map[idx1][idx0] = score;
      return;
    }
    pnt_map[idx0].emplace(idx1, score);
    pnt_map[idx1].emplace(idx0, score);
    check_and_remove_weak_match(idx0);
  }

  std::vector<PointIdxs> get_tracks() {
    PointIdxUSet           visited;
    std::vector<PointIdxs> result;
    for(const auto& [pnt, _] : pnt_map) {
      if(visited.contains(pnt)) {
        continue;
      }
      PointIdxs res;
      bfs(pnt, [&res](const PointIdx& idx) { res.push_back(idx); }, &visited);
      result.push_back(res);
    }
    return result;
  }

private:

  template <typename T>
    requires std::is_same_v<T, std::queue<PointIdx>> || std::is_same_v<T, std::stack<PointIdx>>
  struct Container {
    T container;

    void push(const PointIdx& idx) { container.push(idx); }

    PointIdx pop() {
      PointIdx first;
      if constexpr(std::is_same_v<T, std::queue<PointIdx>>) {
        first = container.front();
      } else {
        first = container.top();
      }
      container.pop();
      return first;
    }

    bool empty() const { return container.empty(); }
  };

  using Queue = Container<std::queue<PointIdx>>;
  using Stack = Container<std::stack<PointIdx>>;

  template <typename C>
    requires std::is_same_v<C, Queue> || std::is_same_v<C, Stack>
  void dfs_bfs(const PointIdx& start, std::function<void(const PointIdx&)> visit, PointIdxUSet* arranged = nullptr) {
    bool arranged_owned = false;
    if(!arranged) {
      arranged       = new PointIdxUSet;
      arranged_owned = true;
    }
    C    container;
    auto add = [&arranged, &container](const PointIdx& idx) {
      if(arranged->contains(idx)) {
        return;
      }
      container.push(idx);
      arranged->insert(idx);
    };
    add(start);
    while(!container.empty()) {
      PointIdx current = container.pop();
      visit(current);
      for(const auto& [next, _] : pnt_map[current]) {
        add(next);
      }
    }
    if(arranged_owned) {
      delete arranged;
    }
  }

  void dfs(const PointIdx& start, std::function<void(const PointIdx&)> visit, PointIdxUSet* arranged = nullptr) {
    dfs_bfs<Stack>(start, visit, arranged);
  }

  void bfs(const PointIdx& start, std::function<void(const PointIdx&)> visit, PointIdxUSet* arranged = nullptr) {
    dfs_bfs<Queue>(start, visit, arranged);
  }

  using WeightMap = PointIdxUMap<PointIdxUMap<float>>;
  WeightMap pnt_map;

  using Edge  = std::pair<PointIdx, PointIdx>;
  using Edges = std::vector<Edge>;

  struct PathInfo {
    float               min_score;
    std::optional<Edge> min_edge;
    PointIdx            current;
    PointIdxUSet        visited;
  };

  std::optional<Edge> find_weak_match(const PointIdx& start, const PointIdx& end) {
    auto cmp = [](const PathInfo& a, const PathInfo& b) { return a.min_score > b.min_score; };
    std::priority_queue<PathInfo, std::vector<PathInfo>, decltype(cmp)> pq{cmp};
    pq.emplace(std::numeric_limits<float>::max(), std::nullopt, start, PointIdxUSet{start});
    std::optional<Edge> global_min_edge;
    float               global_min_score = std::numeric_limits<float>::max();
    while(!pq.empty()) {
      auto current_state = pq.top();
      pq.pop();
      if(current_state.min_score >= global_min_score) {
        continue;
      }
      if(current_state.current == end) {
        global_min_edge  = current_state.min_edge;
        global_min_score = current_state.min_score;
        continue;
      }
      for(const auto& [neighbor, score] : pnt_map[current_state.current]) {
        if(current_state.visited.count(neighbor)) {
          continue;
        }
        float new_min_score = std::min(current_state.min_score, score);
        if(new_min_score >= global_min_score) {
          continue;
        }
        auto new_min_edge = (new_min_score == score ? Edge{current_state.current, neighbor} : current_state.min_edge);
        auto new_visited  = current_state.visited;
        new_visited.insert(neighbor);
        pq.emplace(new_min_score, new_min_edge, neighbor, std::move(new_visited));
      }
    }
    return global_min_edge;
  }

  using PointIdxPair  = std::pair<PointIdx, PointIdx>;
  using PointIdxPairs = std::vector<PointIdxPair>;

  PointIdxPairs find_conflicts(const PointIdx& start) {
    std::unordered_map<int, PointIdx> img_pnt_match;
    PointIdxPairs                     conflicts;
    bfs(start, [&img_pnt_match, &conflicts](const PointIdx& idx) {
      if(img_pnt_match.contains(idx.img_idx)) {
        conflicts.emplace_back(img_pnt_match[idx.img_idx], idx);
      } else {
        img_pnt_match.emplace(idx.img_idx, idx);
      }
    });
    return conflicts;
  }

  void check_and_remove_weak_match(const PointIdx& node) {
    auto conflicts = find_conflicts(node);
    if(conflicts.empty()) {
      return;
    }
    for(const auto& [start, end] : conflicts) {
      while(true) {
        auto weak_edge = find_weak_match(start, end);
        if(!weak_edge) {
          break;
        }
        const auto& [from, to] = *weak_edge;
        pnt_map[from].erase(to);
        pnt_map[to].erase(from);
      }
    }
  }
};

} // namespace Ortho
#endif
