#ifndef ORTHO_TRACKS_HPP
#define ORTHO_TRACKS_HPP

#include <algorithm>
#include <cassert>
#include <execution>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <ranges>
#include <stack>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "types.hpp"

namespace Ortho {

struct TracksMaintainer {
public:

  void append_match(PointIdx idx0, PointIdx idx1, float score) {
    assert(idx0.img_idx != idx1.img_idx);
    float weight        = score2weight(score);
    float max_weight    = std::max(pnt_map[idx0][idx1], weight);
    pnt_map[idx0][idx1] = max_weight;
    pnt_map[idx1][idx0] = max_weight;
    check_and_remove_weak_match(idx0);
  }

  std::vector<PointIdxs> get_tracks() const {
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

  float score2weight(float score) { return -std::log(std::max(1e-6f, 1.0f - score)); }

  using WeightMap     = PointIdxUMap<PointIdxUMap<float>>;
  using PointIdxPair  = std::pair<PointIdx, PointIdx>;
  using PointIdxPairs = std::vector<PointIdxPair>;
  WeightMap pnt_map;

  struct EdgeWithWeight {
    PointIdx u, v;
    float    weight;
  };

  template <typename Func>
    requires std::is_invocable_v<Func, const PointIdx&> || std::is_invocable_v<Func, const EdgeWithWeight&>
  static void bfs(const WeightMap& pnt_map, const PointIdx& start, Func visit, PointIdxUSet* arranged = nullptr) {
    bool arranged_owned = !arranged;
    if(arranged_owned) {
      arranged = new PointIdxUSet;
    }
    std::queue<PointIdx> queue;
    auto                 add_new_node = [&arranged, &queue](const PointIdx& idx) {
      if(arranged->contains(idx)) {
        return false;
      }
      queue.push(idx);
      arranged->insert(idx);
      return true;
    };
    if(!add_new_node(start)) {
      return;
    }
    while(!queue.empty()) {
      PointIdx current = queue.front();
      queue.pop();
      if constexpr(std::is_invocable_v<Func, const PointIdx&>) {
        visit(current);
      }
      const auto& neighbors = pnt_map.find(current);
      if(neighbors == pnt_map.end()) {
        continue;
      }
      for(const auto& [next, weight] : neighbors->second) {
        if(!add_new_node(next)) {
          continue;
        }
        if constexpr(std::is_invocable_v<Func, const EdgeWithWeight&>) {
          visit({current, next, weight});
        }
      }
    }
    if(arranged_owned) {
      delete arranged;
    }
  }

  template <typename Func>
  void bfs(const PointIdx& start, Func visit, PointIdxUSet* arranged = nullptr) const {
    bfs(pnt_map, start, visit, arranged);
  }

  PointIdxPairs find_conflicts(const PointIdx& start) const {
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

  class Dinic {
  public:

    Dinic(const WeightMap& pnt_map, const PointIdx& start, const PointIdx& end) : s(start), t(end) {
      bfs(pnt_map, start, [this](const EdgeWithWeight& pair) {
        graph[pair.u][pair.v] = pair.weight;
        graph[pair.v][pair.u] = pair.weight;
      });
    }

    PointIdxPairs operator()() {
      float flow = 0;
      while(true) {
        update_level(s);
        if(level[t] == 0) {
          break;
        }
        float f;
        while((f = augment(s, t, std::numeric_limits<float>::max())) > 0) {
          flow += f;
        }
      }
      PointIdxPairs ans;
      for(const auto& [u, next] : graph) {
        if(level[u] == 0) {
          continue;
        }
        for(const auto& [v, weight] : next) {
          if(level[v] == 0) {
            ans.emplace_back(u, v);
          }
        }
      }
      return ans;
    }

  private:

    WeightMap graph;
    using Iter = typename PointIdxUMap<float>::iterator;
    PointIdxUMap<Iter> iter;
    PointIdxUMap<int>  level;
    PointIdx           s, t;

    void update_level(PointIdx s) {
      level.clear();
      iter.clear();
      std::queue<PointIdx> q;
      level[s] = 1;
      q.push(s);
      while(!q.empty()) {
        PointIdx u = q.front();
        q.pop();
        for(const auto& [v, cap] : graph[u]) {
          if(cap > 0 && level[v] == 0) {
            level[v] = level[u] + 1;
            q.push(v);
          }
        }
      }
    }

    float augment(PointIdx u, PointIdx t, float f) {
      if(u == t) {
        return f;
      }
      if(!iter.count(u)) {
        iter.emplace(u, graph[u].begin());
      }
      for(Iter& i = iter[u]; i != graph[u].end(); ++i) {
        PointIdx v   = i->first;
        float&   cap = i->second;
        if(cap > 0 && level[u] < level[v]) {
          float d = augment(v, t, std::min(f, cap));
          if(d > 0) {
            cap -= d;
            graph[v][u] += d;
            return d;
          }
        }
      }
      return 0;
    }
  };

  void check_and_remove_weak_match(const PointIdx& node) {
    auto conflicts = find_conflicts(node);
    for(const auto& [s, t] : conflicts) {
      Dinic dinic{pnt_map, s, t};
      auto  pointidx_pairs = dinic();
      for(const auto& pair : pointidx_pairs) {
        pnt_map[pair.first].erase(pair.second);
        pnt_map[pair.second].erase(pair.first);
      }
    }
  }
};

} // namespace Ortho
#endif
