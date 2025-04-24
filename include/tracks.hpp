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

  static void
  bfs(const WeightMap&                     pnt_map,
      const PointIdx&                      start,
      std::function<void(const PointIdx&)> visit,
      PointIdxUSet*                        arranged = nullptr) {
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
      const auto& neighbors = pnt_map.find(current);
      if(neighbors == pnt_map.end()) {
        continue;
      }
      for(const auto& [next, _] : neighbors->second) {
        add(next);
      }
    }
    if(arranged_owned) {
      delete arranged;
    }
  }

  void bfs(const PointIdx& start, std::function<void(const PointIdx&)> visit, PointIdxUSet* arranged = nullptr) const {
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

  struct Edge {
    int   u;
    int   v;
    float weight;
  };

  PointIdxPairs find_min_cut(const PointIdxPairs& pairs) const {
    if(pairs.empty())
      return {};

    // 1. 构建图结构
    PointIdxUMap<int> node_to_index;
    PointIdxs         index_to_node;
    std::vector<Edge> edges;

    // 收集所有节点并建立索引映射
    for(const auto& [u, v] : pairs) {
      if(!node_to_index.count(u)) {
        node_to_index[u] = index_to_node.size();
        index_to_node.push_back(u);
      }
      if(!node_to_index.count(v)) {
        node_to_index[v] = index_to_node.size();
        index_to_node.push_back(v);
      }
    }

    // 收集所有边及其权重
    for(const auto& [u, v] : pairs) {
      int   u_idx  = node_to_index.at(u);
      int   v_idx  = node_to_index.at(v);
      float weight = pnt_map.at(u).at(v);
      edges.push_back({u_idx, v_idx, weight});
    }

    // 2. 构建ILP/LP问题
    int num_nodes = node_to_index.size();
    int num_edges = edges.size();
    int num_pairs = pairs.size();

    // 变量: 每条边一个变量x_e (0 <= x_e <= 1)
    // 目标: 最小化 sum(weight_e * x_e)

    // 构建目标函数
    Eigen::VectorXd c(num_edges);
    for(int e = 0; e < num_edges; ++e) {
      c[e] = edges[e].weight;
    }

    // 约束条件:
    // 对于每个源汇对(s,t), 需要至少有一条边在割中
    // 我们可以表示为 sum_{e in path} x_e >= 1 对于所有s-t路径

    // 由于路径可能很多，我们采用更高效的方法:
    // 对每个源汇对(s,t), 添加流变量f_pu, 并设置流约束

    // 变量顺序: [x_0, ..., x_{m-1}, f_0u0, f_0u1, ..., f_{k-1}u_{n-1}]
    // 其中m是边数，k是源汇对数，n是节点数

    // 这里简化处理: 我们只考虑给定的冲突对作为源汇对
    // 每个冲突对(u,v)要求u和v不在同一连通分量

    // 约束矩阵
    std::vector<Eigen::Triplet<double>> triplets;
    Eigen::VectorXd                     b;
    int                                 constraint_idx = 0;

    // 为每个冲突对添加约束
    for(const auto& [s, t] : pairs) {
      int s_idx = node_to_index.at(s);
      int t_idx = node_to_index.at(t);

      // 添加流守恒约束
      for(int u = 0; u < num_nodes; ++u) {
        if(u == s_idx || u == t_idx)
          continue;

        // 入流 = 出流
        for(const auto& edge : edges) {
          if(edge.u == u) {
            triplets.emplace_back(constraint_idx, num_edges + u, 1);
          }
          if(edge.v == u) {
            triplets.emplace_back(constraint_idx, num_edges + u, -1);
          }
        }
        b.conservativeResize(constraint_idx + 1);
        b(constraint_idx) = 0;
        constraint_idx++;
      }

      // s的出流=1
      for(const auto& edge : edges) {
        if(edge.u == s_idx) {
          triplets.emplace_back(constraint_idx, num_edges + edge.v, 1);
        }
      }
      b.conservativeResize(constraint_idx + 1);
      b(constraint_idx) = 1;
      constraint_idx++;

      // t的入流=1
      for(const auto& edge : edges) {
        if(edge.v == t_idx) {
          triplets.emplace_back(constraint_idx, num_edges + edge.u, 1);
        }
      }
      b.conservativeResize(constraint_idx + 1);
      b(constraint_idx) = 1;
      constraint_idx++;

      // 边容量约束: f_u + f_v <= x_e
      for(int e = 0; e < num_edges; ++e) {
        int u = edges[e].u;
        int v = edges[e].v;

        triplets.emplace_back(constraint_idx, e, -1);
        triplets.emplace_back(constraint_idx, num_edges + u, 1);
        triplets.emplace_back(constraint_idx, num_edges + v, 1);
        b.conservativeResize(constraint_idx + 1);
        b(constraint_idx) = 0;
        constraint_idx++;
      }
    }

    // 边变量约束: 0 <= x_e <= 1
    for(int e = 0; e < num_edges; ++e) {
      triplets.emplace_back(constraint_idx, e, 1);
      b.conservativeResize(constraint_idx + 1);
      b(constraint_idx) = 1;
      constraint_idx++;

      triplets.emplace_back(constraint_idx, e, -1);
      b.conservativeResize(constraint_idx + 1);
      b(constraint_idx) = 0;
      constraint_idx++;
    }

    // 构建稀疏矩阵
    Eigen::SparseMatrix<double> A(constraint_idx, num_edges + num_nodes * num_pairs);
    A.setFromTriplets(triplets.begin(), triplets.end());

    // 3. 求解LP
    Eigen::VectorXd x;
    try {
      // 使用最小二乘法求解
      Eigen::SparseMatrix<double>                        AtA = A.transpose() * A;
      Eigen::VectorXd                                    Atb = A.transpose() * b;
      Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      solver.compute(AtA);
      if(solver.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed");
      }
      x = solver.solve(Atb);
      if(solver.info() != Eigen::Success) {
        throw std::runtime_error("Solving failed");
      }
    } catch(...) {
      // 如果求解失败，回退到简单的启发式方法
      std::cerr << "LP solver failed, falling back to greedy" << std::endl;
      return greedy_min_cut(pairs);
    }

    // 4. 确定性舍入
    PointIdxPairs cut_edges;
    double        threshold = 1.0 / num_pairs;
    for(int e = 0; e < num_edges; ++e) {
      if(x(e) >= threshold) {
        cut_edges.emplace_back(index_to_node[edges[e].u], index_to_node[edges[e].v]);
      }
    }

    return cut_edges;
  }

  // 简单的启发式方法作为回退
  PointIdxPairs greedy_min_cut(const PointIdxPairs& pairs) const {
    PointIdxPairs result;
    PointIdxUSet  processed;

    for(const auto& [u, v] : pairs) {
      if(processed.count(u) || processed.count(v))
        continue;

      // 选择权重较小的边
      float        min_weight = std::numeric_limits<float>::max();
      PointIdxPair min_edge;

      for(const auto& [neighbor, weight] : pnt_map.at(u)) {
        if(weight < min_weight) {
          min_weight = weight;
          min_edge   = {u, neighbor};
        }
      }

      for(const auto& [neighbor, weight] : pnt_map.at(v)) {
        if(weight < min_weight) {
          min_weight = weight;
          min_edge   = {v, neighbor};
        }
      }

      if(min_weight != std::numeric_limits<float>::max()) {
        result.push_back(min_edge);
        processed.insert(min_edge.first);
        processed.insert(min_edge.second);
      }
    }

    return result;
  }

  void check_and_remove_weak_match(const PointIdx& node) {
    // auto conflicts = find_conflicts(node);
    // auto edges     = find_min_cut(conflicts);
    // for(const auto& [idx0, idx1] : edges) {
    //   pnt_map[idx0].erase(idx1);
    //   pnt_map[idx1].erase(idx0);
    // }
  }
};

} // namespace Ortho
#endif
