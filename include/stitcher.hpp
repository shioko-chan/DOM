#ifndef ORTHO_STITCHER_HPP
#define ORTHO_STITCHER_HPP

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <optional>
#include <random>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "utility.hpp"

namespace fs = std::filesystem;

namespace Ortho {
class Stitcher {
private:

  struct Node {
    Image img, mask;
  };

  using NodeList = std::vector<Node>;

  struct Edge {
    cv::Mat M;
    int     priority{0};
  };

  using Adjacent = std::unordered_map<int, std::unordered_map<int, Edge>>;

  NodeList nodes;
  Adjacent adjacent;
  fs::path temporary_save_path;

  static std::string generate_uuid_v4() {
    std::random_device                      rd;
    std::mt19937                            gen(rd());
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
    auto                                    rand_hex = [&gen, &dist](int width) {
      std::stringstream ss;
      ss << std::hex << std::setfill('0') << std::setw(width) << dist(gen);
      return ss.str();
    };
    std::stringstream uuid;
    uuid << rand_hex(8) << "-";
    uuid << rand_hex(4) << "-";
    uuid << std::hex << std::setfill('0') << std::setw(4) << ((dist(gen) & 0x0FFF) | 0x4000) << "-";
    uuid << std::hex << std::setfill('0') << std::setw(4) << ((dist(gen) & 0x3FFF) | 0x8000) << "-";
    uuid << rand_hex(12);
    return uuid.str();
  }

  std::optional<int> find_max_out_degree(int priority) {
    if(adjacent.empty()) {
      return std::nullopt;
    }
    auto [idx, len] = std::ranges::max(
        adjacent | std::views::transform([&priority](const auto& pair) -> std::pair<int, size_t> {
          return {pair.first, std::ranges::count_if(pair.second, [&priority](const auto& pair) {
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

  std::pair<Node, cv::Rect> stitch_adjacent_images(int idx0) {
    auto [w, h] = nodes[idx0].mask.get().get().size();
    Points<int> all_corners{Point<int>(0, 0), Point<int>(w - 1, 0), Point<int>(w - 1, h - 1), Point<int>(0, h - 1)};
    for(const auto& [idx1, edge] : adjacent[idx0]) {
      auto [w, h] = nodes[idx1].mask.get().get().size();
      Points<float> corners{Point<float>(0, 0), Point<float>(w - 1, 0), Point<float>(w - 1, h - 1), Point<float>(0, h - 1)};
      cv::Mat M;
      cv::invertAffineTransform(edge.M, M);
      cv::transform(corners, corners, M);
      std::ranges::move(
          corners | std::views::transform([](const Point<float>& p) {
            return Point<int>(Ortho::abs_ceil(p.x), Ortho::abs_ceil(p.y));
          }),
          std::back_inserter(all_corners));
    }
    cv::Rect rect       = cv::boundingRect(all_corners);
    cv::Mat  center_img = nodes[idx0].img.get().get(), center_img_mask = nodes[idx0].mask.get().get();
    cv::Mat  result(rect.height, rect.width, center_img.type(), cv::Scalar(0, 0, 0)),
        mask_result(rect.height, rect.width, center_img_mask.type(), cv::Scalar(0));
    center_img.copyTo(result(cv::Rect(-rect.x, -rect.y, w, h)));
    center_img_mask.copyTo(mask_result(cv::Rect(-rect.x, -rect.y, w, h)));
    for(const auto& [idx1, edge] : adjacent[idx0]) {
      cv::Mat append_img = nodes[idx1].img.get().get(), append_img_mask = nodes[idx1].mask.get().get();
      cv::Mat M;
      cv::invertAffineTransform(edge.M, M);
      M.at<float>(0, 2) -= rect.x;
      M.at<float>(1, 2) -= rect.y;
      cv::Mat warped, mask_warped;
      cv::warpAffine(append_img, warped, M, result.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::warpAffine(append_img_mask, mask_warped, M, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      cv::Mat mixed;
      cv::addWeighted(result, 0.5, warped, 0.5, 0, mixed);
      cv::Mat mask_mixed = mask_result & mask_warped;
      mixed.copyTo(result, mask_mixed);
      cv::Mat mask_warped_unique = ~mask_result & mask_warped;
      warped.copyTo(result, mask_warped_unique);
      mask_result = mask_result | mask_warped;
    }
    std::string uuid = generate_uuid_v4(), extension = nodes[idx0].img.get_img_extension().string();
    return std::make_pair(
        Node{
            Image(temporary_save_path / (uuid + extension), std::move(result)),
            Image(temporary_save_path / (uuid + "_mask" + extension), std::move(mask_result))},
        rect);
  }

public:

  Stitcher() = delete;

  Stitcher(MatchPairs& match_pairs, ImgsData& imgs_data, fs::path temporary_save_path) :
      temporary_save_path(temporary_save_path) {
    auto v = imgs_data | std::views::transform([](auto&& data) {
               return Node{data.rotate_rectified(), data.rotate_rectified_mask()};
             });
    nodes.assign(v.begin(), v.end());
    for(const auto& match_pair : match_pairs) {
      cv::invertAffineTransform(match_pair.M, adjacent[match_pair.second][match_pair.first].M);
      adjacent[match_pair.first].emplace(match_pair.second, std::move(match_pair.M));
    }
  }

  cv::Mat stitch() {
    if(nodes.empty()) {
      throw std::runtime_error("No image!");
    }
    std::erase_if(adjacent, [](const auto& pair) { return pair.second.size() == 0; });
    int  priority = 0;
    auto i_       = find_max_out_degree(priority);
    while(adjacent.size() > 1 && i_) {
      int i                 = i_.value();
      auto [stitched, rect] = stitch_adjacent_images(i);
      fs::path path         = temporary_save_path / std::format("p{}-i{}.jpg", priority, i);
      cv::imwrite(path.string(), stitched.img.get().get());
      cv::Mat M_p_i         = cv::Mat::eye(3, 3, CV_32FC1);
      M_p_i.at<float>(0, 2) = rect.x;
      M_p_i.at<float>(1, 2) = rect.y;
      nodes.push_back(std::move(stitched));
      for(auto& [j, edge_i_j] : adjacent[i]) {
        if(j == i) {
          continue;
        }
        for(auto& [l, edge_j_l] : adjacent[j]) {
          if(l == i || l == j || adjacent[i].contains(l)) {
            continue;
          }
          cv::Mat M_i_j = edge_i_j.M;
          cv::vconcat(M_i_j, cv::Mat{(cv::Mat_<float>(1, 3) << 0, 0, 1)}, M_i_j);
          cv::Mat M_p_l    = edge_j_l.M * M_i_j * M_p_i;
          auto&   edge_l_p = adjacent[l][nodes.size() - 1];
          cv::invertAffineTransform(M_p_l, edge_l_p.M);
          edge_l_p.priority = edge_j_l.priority + 1;
          auto& edge_p_l    = adjacent[nodes.size() - 1][l];
          edge_p_l.M        = M_p_l;
          edge_p_l.priority = edge_j_l.priority + 1;
          adjacent[l].erase(j);
        }
        adjacent.erase(j);
      }
      adjacent.erase(i);
      i_ = find_max_out_degree(priority);
      if(!i_) {
        i_ = find_max_out_degree(++priority);
      }
    }
    cv::Mat stitched_img;
    nodes.back().img.get().get().copyTo(stitched_img);
    return stitched_img;
  }
};

} // namespace Ortho
#endif