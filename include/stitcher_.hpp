#ifndef ORTHO_STITCHER_HPP
#define ORTHO_STITCHER_HPP

#include <algorithm>
#include <filesystem>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/blenders.hpp>

#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "tools/utility.hpp"

namespace Ortho {

namespace fs = std::filesystem;

class Stitcher {
public:

  Stitcher() = delete;

  Stitcher(MatchPairs& match_pairs, ImgsData& imgs_data, fs::path temporary_save_path) :
      temporary_save_path(std::move(temporary_save_path)) {
    auto v = imgs_data | std::views::transform([](auto&& data) noexcept { return Node{data.img(), data.mask()}; });
    nodes.assign(v.begin(), v.end());
    for(const auto& match_pair : match_pairs) {
      // adjacent[match_pair.second][match_pair.first].M = match_pair.M.inv();
      cv::invertAffineTransform(match_pair.M, adjacent[match_pair.second][match_pair.first].M);
      adjacent[match_pair.first].emplace(match_pair.second, match_pair.M);
    }
  }

  auto stitch() -> cv::Mat {
    if(nodes.empty()) {
      throw std::runtime_error("No image!");
    }
    std::erase_if(adjacent, [](const auto& pair) noexcept { return pair.second.size() == 0; });
    int  priority = 0;
    auto i_       = find_max_out_degree(priority);
    while(adjacent.size() > 1 && i_) {
      int i                 = *i_;
      auto [stitched, rect] = stitch_adjacent_images(i, priority);
      cv::imwrite(stitched.img.get_img_path().string(), stitched.img.get().get());
      cv::Mat M_p_i          = cv::Mat::eye(3, 3, CV_64FC1);
      M_p_i.at<double>(0, 2) = rect.x;
      M_p_i.at<double>(1, 2) = rect.y;
      int p                  = nodes.size();
      nodes.push_back(std::move(stitched));
      std::vector<int> to_erase;
      for(const auto& [j, edge_i_j] : adjacent[i]) {
        if(j == i) {
          continue;
        }
        for(auto& [l, edge_j_l] : adjacent[j]) {
          if(l == i || l == j || adjacent[i].contains(l)) {
            continue;
          }
          cv::Mat M_i_j = edge_i_j.M;
          cv::vconcat(M_i_j, cv::Mat{(cv::Mat_<double>(1, 3) << 0, 0, 1)}, M_i_j);
          cv::Mat M_p_l = edge_j_l.M * M_i_j * M_p_i;
          cv::invertAffineTransform(M_p_l, adjacent[l][p].M);
          // adjacent[l][p].M        = M_p_l.inv();
          adjacent[l][p].priority = edge_j_l.priority + 1;
          adjacent[p][l].M        = M_p_l;
          adjacent[p][l].priority = edge_j_l.priority + 1;
          adjacent[l].erase(j);
        }
        to_erase.push_back(j);
      }
      adjacent.erase(i);
      for(const auto& j : to_erase) {
        adjacent.erase(j);
      }
      i_ = find_max_out_degree(priority);
      if(!i_) {
        i_ = find_max_out_degree(++priority);
      }
    }
    cv::Mat stitched_img;
    nodes.back().img.get().get().copyTo(stitched_img);
    return stitched_img;
  }

private:

  struct alignas(128) Node {
    Image img, mask;
  };

  using NodeList = std::vector<Node>;

  struct alignas(128) Edge {
    cv::Mat M;
    int     priority{0};
  };

  using Adjacent = std::unordered_map<int, std::unordered_map<int, Edge>>;

  NodeList nodes;
  Adjacent adjacent;
  fs::path temporary_save_path;

  auto find_max_out_degree(int priority) -> std::optional<int> {
    if(adjacent.empty()) {
      return std::nullopt;
    }
    auto [idx, len] = std::ranges::max(
        adjacent | std::views::transform([&priority](const auto& pair) noexcept -> std::pair<int, size_t> {
          return {pair.first, std::ranges::count_if(pair.second, [&priority](const auto& pair) noexcept {
                    return pair.second.priority == priority;
                  })};
        }),
        {},
        &std::pair<int, size_t>::second);
    if(len == 0) {
      return std::nullopt;
    }
    return idx;
  }

  auto stitch_adjacent_images(int idx0, int priority) -> std::pair<Node, cv::Rect> {
    auto [w, h] = nodes[idx0].mask.get().get().size();
    Points<int> all_corners{Point<int>(0, 0), Point<int>(w - 1, 0), Point<int>(w - 1, h - 1), Point<int>(0, h - 1)};
    for(const auto& [idx1, edge] : adjacent[idx0]) {
      auto [w, h] = nodes[idx1].mask.get().get().size();
      Points<double>
          corners{Point<double>(0, 0), Point<double>(w - 1, 0), Point<double>(w - 1, h - 1), Point<double>(0, h - 1)};
      cv::Mat M;
      cv::invertAffineTransform(edge.M, M);
      // M = edge.M.inv();
      cv::transform(corners, corners, M);
      // cv::perspectiveTransform(corners, corners, M);
      std::ranges::move(
          corners | std::views::transform([](const Point<double>& p) noexcept {
            return Point<int>(Ortho::abs_ceil(p.x), Ortho::abs_ceil(p.y));
          }),
          std::back_inserter(all_corners));
    }
    cv::Rect rect            = cv::boundingRect(all_corners);
    cv::Mat  center_img      = nodes[idx0].img.get().get();
    cv::Mat  center_img_mask = nodes[idx0].mask.get().get();
    cv::Mat  result(rect.height, rect.width, center_img.type(), cv::Scalar(0, 0, 0));
    cv::Mat  mask_result(rect.height, rect.width, center_img_mask.type(), cv::Scalar(0));
    center_img.copyTo(result(cv::Rect(-rect.x, -rect.y, w, h)));
    center_img_mask.copyTo(mask_result(cv::Rect(-rect.x, -rect.y, w, h)));
    // cv::detail::MultiBandBlender blender{true, 3};
    // cv::Rect                     rect1{0, 0, rect.width, rect.height};
    // blender.prepare(rect1);
    // blender.feed(result, mask_result, cv::Point(0, 0));
    for(const auto& [idx1, edge] : adjacent[idx0]) {
      cv::Mat append_img      = nodes[idx1].img.get().get();
      cv::Mat append_img_mask = nodes[idx1].mask.get().get();
      cv::Mat M;
      cv::invertAffineTransform(edge.M, M);
      cv::vconcat(M, cv::Mat{(cv::Mat_<double>(1, 3) << 0, 0, 1)}, M);
      // M                  = edge.M.inv();
      cv::Mat T          = cv::Mat::eye(3, 3, CV_64FC1);
      T.at<double>(0, 2) = -rect.x;
      T.at<double>(1, 2) = -rect.y;
      M                  = T * M;
      M                  = M.rowRange(0, 2);
      cv::Mat warped;
      cv::Mat mask_warped;
      // cv::warpPerspective(append_img, warped, M, result.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::warpAffine(append_img, warped, M, result.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      // cv::warpPerspective(append_img_mask, mask_warped, M, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      cv::warpAffine(append_img_mask, mask_warped, M, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      cv::Mat mixed;
      cv::addWeighted(result, 0.5, warped, 0.5, 0, mixed);
      cv::Mat mask_mixed = mask_result & mask_warped;
      mixed.copyTo(result, mask_mixed);
      cv::Mat mask_warped_unique = ~mask_result & mask_warped;
      warped.copyTo(result, mask_warped_unique);
      mask_result = mask_result | mask_warped;
      // blender.feed(warped, mask_warped, cv::Point(0, 0));
    }
    // blender.blend(result, mask_result);
    std::string extension = nodes[idx0].img.get_img_extension().string();
    return std::make_pair(
        Node{
            .img  = Image(temporary_save_path / std::format("p{}-i{}{}", priority, idx0, extension), std::move(result)),
            .mask = Image(
                temporary_save_path / std::format("p{}-i{}_mask{}", priority, idx0, extension), std::move(mask_result))},
        rect);
  }
};

} // namespace Ortho
#endif
