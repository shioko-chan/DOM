#ifndef ORTHO_KNN_ON_EUCLIDEAN_DISTANCE_HPP
#define ORTHO_KNN_ON_EUCLIDEAN_DISTANCE_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <ranges>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

namespace Ortho {

class KNN {
public:

  using Point  = cv::Point2f;
  using Points = std::vector<Point>;

  template <typename U>
    requires std::same_as<std::decay_t<U>, Points>
  KNN(int k, U&& data) : k_(k), dataset(std::forward<U>(data)) {}

  template <typename V>
    requires std::ranges::view<V>
  KNN(int k, V v) : k_(k), dataset(v.begin(), v.end()) {}

  std::vector<int> find_nearest_neighbour(const int index) const {
    const auto& point = dataset[index];

    auto v =
        std::views::zip_transform(
            [&point](const int index, const Point& p) { return std::make_pair(euclidean_distance(point, p), index); },
            std::views::iota(0),
            dataset)
        | std::views::filter([index](auto&& pair) { return pair.second != index; }) | std::views::common;

    std::vector<std::pair<float, int>> distances(v.begin(), v.end());
    std::sort(distances.begin(), distances.end());

    auto w = distances | std::views::take(k_) | std::views::transform([](const auto& pair) { return pair.second; })
             | std::views::common;

    return std::vector<int>(w.begin(), w.end());
  }

private:

  const int    k_;
  const Points dataset;

  static float euclidean_distance(const Point& a, const Point& b) {
    return std::sqrt(std::pow(a.x - b.x, 2) + std::pow(a.y - b.y, 2));
  }
};

} // namespace Ortho

#endif