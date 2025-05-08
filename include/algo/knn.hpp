#ifndef SKYMERGE_KNN_ON_EUCLIDEAN_DISTANCE_HPP
#define SKYMERGE_KNN_ON_EUCLIDEAN_DISTANCE_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "types/cv_alias.hpp"

namespace SkyMerge {

template <typename T>
  requires std::is_arithmetic_v<T>
class KNN {
public:

  template <typename U>
    requires std::same_as<std::decay_t<U>, Points<T>>
  KNN(int k_num, U&& data) : k_num(k_num), dataset(std::forward<U>(data)) {}

  template <std::ranges::range R>
  KNN(int k_num, R view) : k_num(k_num), dataset(view.begin(), view.end()) {}

  [[nodiscard]] auto find_nearest_neighbour(const int index) const -> std::vector<int> {
    const auto& point0 = dataset[index];
    auto        view0  = std::views::zip_transform(
                     [&point0](const int index, const Point<T>& point1) noexcept {
                       return std::make_pair(euclidean_distance(point0, point1), index);
                     },
                     std::views::iota(0),
                     dataset)
                 | std::views::filter([index](auto&& pair) noexcept { return pair.second != index; })
                 | std::views::common;
    std::vector<std::pair<double, int>> distances(view0.begin(), view0.end());
    std::nth_element(distances.begin(), distances.begin() + k_num - 1, distances.end());
    auto view1 = distances | std::views::take(k_num)
                 | std::views::transform([](const auto& pair) noexcept { return pair.second; }) | std::views::common;
    return std::vector<int>{view1.begin(), view1.end()};
  }

  [[nodiscard]] auto find_nearest_neighbour(const Point<double> point) const -> std::vector<int> {
    auto view0 = std::views::zip_transform(
                     [&point](const int index, const Point<T>& point1) noexcept {
                       return std::make_pair(euclidean_distance(point, point1), index);
                     },
                     std::views::iota(0),
                     dataset)
                 | std::views::common;
    std::vector<std::pair<double, int>> distances(view0.begin(), view0.end());
    std::nth_element(distances.begin(), distances.begin() + k_num - 1, distances.end());
    auto view1 = distances | std::views::take(k_num)
                 | std::views::transform([](const auto& pair) noexcept { return pair.second; }) | std::views::common;
    return std::vector<int>{view1.begin(), view1.end()};
  }

private:

  int       k_num;
  Points<T> dataset;

  static auto euclidean_distance(const Point<T>& point0, const Point<T>& point1) -> double {
    return std::sqrt(std::pow(point0.x - point1.x, 2) + std::pow(point0.y - point1.y, 2));
  }
};

inline auto find_neighbors(const ImgsData& imgs_data, const int k_neighbors = 8) -> MatchPairs {
  auto knn = KNN<double>(k_neighbors, imgs_data.get() | std::views::transform([](const auto& data) noexcept {
                                        return data.get_coord();
                                      }) | std::views::common);
  std::vector<std::vector<MatchPair>> matches(imgs_data.size());
  run(imgs_data.size(), [&knn, &matches](int idx) noexcept {
    auto neighbors = knn.find_nearest_neighbour(idx);
    for(auto&& neighbour : neighbors) {
      if(idx < neighbour) {
        matches[idx].emplace_back(idx, neighbour);
      } else {
        matches[idx].emplace_back(neighbour, idx);
      }
    }
  });
  auto                view = matches | std::views::join | std::views::common;
  std::set<MatchPair> match_set(view.begin(), view.end());
  return {match_set.begin(), match_set.end()};
}

} // namespace SkyMerge

#endif