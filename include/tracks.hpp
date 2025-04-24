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

  float score2weight(float score) { return -std::log(std::max(1e-6f, 1.0f - score)); }

  using WeightMap     = PointIdxUMap<PointIdxUMap<float>>;
  using Edge          = std::pair<PointIdx, PointIdx>;
  using Edges         = std::vector<Edge>;
  using PointIdxPair  = std::pair<PointIdx, PointIdx>;
  using PointIdxPairs = std::vector<PointIdxPair>;
  WeightMap pnt_map;

  void bfs(const PointIdx& start, std::function<void(const PointIdx&)> visit, PointIdxUSet* arranged = nullptr) {
    bool arranged_owned = !arranged;
    if(arranged_owned) {
      arranged = new PointIdxUSet;
    }
    std::queue<PointIdx> queue;
    auto                 add = [&arranged, &queue](const PointIdx& idx) {
      if(arranged->contains(idx)) {
        return;
      }
      queue.push(idx);
      arranged->insert(idx);
    };
    add(start);
    while(!queue.empty()) {
      PointIdx current = queue.front();
      queue.pop();
      visit(current);
      for(const auto& [next, _] : pnt_map[current]) {
        add(next);
      }
    }
    if(arranged_owned) {
      delete arranged;
    }
  }

  static constexpr int N = 1e4 + 5, M = 2e5 + 5;
  int                  n, m, s, t, tot = 1, lnk[N], ter[M], nxt[M], val[M], dep[N], cur[N];

  void add(int u, int v, int w) { ter[++tot] = v, nxt[tot] = lnk[u], lnk[u] = tot, val[tot] = w; }

  void addedge(int u, int v, int w) { add(u, v, w), add(v, u, 0); }

  int bfs(int s, int t) {
    memset(dep, 0, sizeof(dep));
    memcpy(cur, lnk, sizeof(lnk));
    std::queue<int> q;
    q.push(s), dep[s] = 1;
    while(!q.empty()) {
      int u = q.front();
      q.pop();
      for(int i = lnk[u]; i; i = nxt[i]) {
        int v = ter[i];
        if(val[i] && !dep[v])
          q.push(v), dep[v] = dep[u] + 1;
      }
    }
    return dep[t];
  }

  int dfs(int u, int t, int flow) {
    if(u == t)
      return flow;
    int ans = 0;
    for(int& i = cur[u]; i && ans < flow; i = nxt[i]) {
      int v = ter[i];
      if(val[i] && dep[v] == dep[u] + 1) {
        int x = dfs(v, t, std::min(val[i], flow - ans));
        if(x)
          val[i] -= x, val[i ^ 1] += x, ans += x;
      }
    }
    if(ans < flow)
      dep[u] = -1;
    return ans;
  }

  int dinic(int s, int t) {
    int ans = 0;
    while(bfs(s, t)) {
      int x;
      while((x = dfs(s, t, 1 << 30)))
        ans += x;
    }
    return ans;
  }

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
    for(const auto& [s, t] : conflicts) {
    }
  }
};

} // namespace Ortho
#endif
