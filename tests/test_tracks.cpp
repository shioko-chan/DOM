#include <iostream>

#include <gtest/gtest.h>

#include "algo/tracks.hpp"

using namespace SkyMerge;

PointIdx test_make_idx(int i) { return {i, 0}; }

bool contains_pair(const PointIdxPairs& pairs, const PointIdx& u, const PointIdx& v) {
  return std::find(pairs.begin(), pairs.end(), std::make_pair(u, v)) != pairs.end()
         || std::find(pairs.begin(), pairs.end(), std::make_pair(v, u)) != pairs.end();
}

bool contains_pair(const PointIdxPairs& pairs, int u_, int v_) {
  auto u = test_make_idx(u_), v = test_make_idx(v_);
  return std::find(pairs.begin(), pairs.end(), std::make_pair(u, v)) != pairs.end()
         || std::find(pairs.begin(), pairs.end(), std::make_pair(v, u)) != pairs.end();
}

struct EdgeTest {
  int    u, v;
  double w;

  explicit operator EdgeWithWeight() const { return EdgeWithWeight{test_make_idx(u), test_make_idx(v), w}; }
};

WeightMap& operator<<(WeightMap& map, EdgeTest edge) {
  map << static_cast<EdgeWithWeight>(edge);
  std::swap(edge.u, edge.v);
  map << static_cast<EdgeWithWeight>(edge);
  return map;
}

TEST(DinicTest, testSimpleGraph) {
  WeightMap graph;
  graph << EdgeTest{0, 1, 3.f} << EdgeTest{0, 2, 2.f} << EdgeTest{1, 2, 1.f} << EdgeTest{1, 3, 1.f}
        << EdgeTest{2, 3, 3.f};
  PointIdxPairs min_cut      = Dinic{}(graph, test_make_idx(0), test_make_idx(3));
  bool          correct_cut1 = contains_pair(min_cut, test_make_idx(1), test_make_idx(3))
                      && contains_pair(min_cut, test_make_idx(2), test_make_idx(3));
  bool correct_cut2 = contains_pair(min_cut, test_make_idx(1), test_make_idx(3))
                      && contains_pair(min_cut, test_make_idx(1), test_make_idx(2))
                      && contains_pair(min_cut, test_make_idx(0), test_make_idx(2));
  EXPECT_TRUE(correct_cut1 || correct_cut2) << min_cut;
}

TEST(DinicTest, testLargerGraph) {
  WeightMap graph;
  graph << EdgeTest{0, 1, 16.f} << EdgeTest{0, 2, 13.f} << EdgeTest{1, 2, 10.f} << EdgeTest{1, 3, 12.f}
        << EdgeTest{2, 4, 14.f} << EdgeTest{2, 3, 9.f} << EdgeTest{3, 5, 20.f} << EdgeTest{3, 4, 7.f}
        << EdgeTest{4, 5, 4.f};
  PointIdxPairs min_cut  = Dinic{}(graph, test_make_idx(0), test_make_idx(5));
  double        capacity = 0.f;
  for(const auto& [u, v] : min_cut) {
    if(graph.find(u) != graph.end() && graph[u].find(v) != graph[u].end()) {
      capacity += graph[u][v];
    }
  }
  EXPECT_LE(capacity, 24.f) << "Min cut capacity: " << capacity << "larger than 24.0.";
  EXPECT_TRUE(
      contains_pair(min_cut, test_make_idx(3), test_make_idx(5))
      && contains_pair(min_cut, test_make_idx(4), test_make_idx(5)))
      << min_cut;
}

TEST(DinicTest, testDisconnectedGraph) {
  WeightMap graph;
  graph << EdgeTest{0, 1, 10.f} << EdgeTest{1, 2, 10.f} << EdgeTest{3, 4, 10.f} << EdgeTest{4, 5, 10.f};
  PointIdxPairs min_cut = Dinic{}(graph, test_make_idx(0), test_make_idx(5));
  EXPECT_TRUE(min_cut.empty()) << min_cut;
}

TEST(DinicTest, testSingleEdgeGraph) {
  WeightMap graph;
  graph << EdgeTest{0, 1, 5.f};
  PointIdxPairs min_cut = Dinic{}(graph, test_make_idx(0), test_make_idx(1));
  EXPECT_TRUE(min_cut.size() == 1);
  EXPECT_TRUE(contains_pair(min_cut, test_make_idx(0), test_make_idx(1))) << min_cut;
}