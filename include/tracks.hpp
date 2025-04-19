#ifndef ORTHO_TRACKS_HPP
#define ORTHO_TRACKS_HPP

#include <algorithm>
#include <numeric>
#include <ranges>
#include <unordered_set>
#include <vector>
#include <queue>
#include <cassert>
#include <optional>

#include "types.hpp"

namespace Ortho {

struct TracksMaintainer {
public:

  void append_match(PointIdx idx0, PointIdx idx1, float score) {
    if (pnt_map.contains(idx0) && pnt_map[idx0].contains(idx1) && pnt_map[idx0][idx1] < score) {
      pnt_map[idx0][idx1] = score;
      pnt_map[idx1][idx0] = score;
      return;
    }
    pnt_map[idx0].emplace(idx1, score);
    pnt_map[idx1].emplace(idx0, score);
    check_and_remove_weak_match(idx0);
  }

  std::vector<PointIdxs> get_tracks() {
    PointIdxUSet visited;
    auto bfs = [&visited](const PointIdx& start) {
      if (visited.contains(start)) { return std::nullopt; }
      PointIdxs res { start };
      std::queue<PointIdx> bfs_queue { start };
      visited.insert(start);
      while (!bfs_queue.empty()) {
        PointIdx current = bfs_queue.front();
        for (const auto& [next, _] : pnt_map[current]) {
          if (visited.contains(next)) { continue; }
          res.push_back(next);
          bfs_queue.push(next);
          visited.insert(next);
        }
        bfs_queue.pop();
      }
      return res;
      };
    std::vector<PointIdxs> result;
    for (const auto& pnt : pnts) {
      auto res = bfs(pnt);
      if (res) {
        result.push_back(std::move(*res));
      }
    }
    return result;
  }

private:
  PointIdxs pnts;
  PointIdxUMap<PointIdxUMap<float>> pnt_map;

  using EdgeWithWeight = std::pair<std::pair<PointIdx, PointIdx>, float>;
  using EdgesWithWeight = std::vector<EdgeWithWeight>;

  struct DFS {
  public:
    DFS(const auto& pnt_map, const PointIdx& start, const PointIdx& end) :pnt_map(pnt_map), start(start), end(end) {}
    EdgesWithWeight operator()() {
      EdgesWithWeight weak_edges;
      PointIdxUSet visited { start };
      find_weak_match_per_path(&weak_edges, &visited, start);
      return std::move(weak_edges);
    }
  private:
    const PointIdxUMap<PointIdxUMap<float>>& pnt_map;
    PointIdx start, end;
    void find_weak_match_per_path(EdgesWithWeight* const weak_edges, PointIdxUSet* const visited, const PointIdx& cur, std::optional<EdgeWithWeight> weak_edge = std::nullopt) {
      if (cur == end) {
        if (weak_edge) {
          weak_edges->push_back(*weak_edge);
        }
        return;
      }
      for (const auto& [next, score] : pnt_map[cur]) {
        if (visited->contains(next)) { continue; }
        visited->emplace(next);
        if (!weak_edge || score < *weak_edge.second) {
          find_weak_match_per_path(weak_edges, next, { {cur,next},score });
        } else {
          find_weak_match_per_path(weak_edges, next, weak_edge);
        }
        visited->erase(next);
      }
    }
  };

  void check_and_remove_weak_match(const PointIdx& start) {
    std::queue<PointIdx> bfs_queue { start };
    std::unordered_map<int, PointIdx> img_pnt_match { {start.img_idx, start} };
    PointIdxUSet visited { start };
    std::vector<PointIdx> conflict;
    while (!bfs_queue.empty()) {
      PointIdx current = bfs_queue.front();
      for (const auto& [next, _] : pnt_map[current]) {
        if (visited.contains(next)) { continue; }
        if (img_pnt_match.contains(next.img_idx)) {
          conflict.push_back(img_pnt_match[next.img_idx]);
          conflict.push_back(next);
        }
        img_pnt_match.emplace(next.img_idx, next);
        bfs_queue.push(next);
        visited.insert(next);
      }
      bfs_queue.pop();
    }
    assert(conflict.size() == 2);
    DFS dfs { pnt_map, conflict[0], conflict[1] };
    while (true) {
      auto edges = dfs();
      if (edges.empty()) {
        break;
      }
      auto [idx0, idx1] = std::min_element(edges.start(), edges.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second < rhs.second;
      })->first;
      pnt_map[idx0].erase(idx1);
      pnt_map[idx1].erase(idx0);
    }
  }
};

} // namespace Ortho
#endif
