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

// {
//   Points<float> corners =
//       {Point<float>(0, 0),
//        Point<float>(img2.cols - 1, 0),
//        Point<float>(img2.cols - 1, img2.rows - 1),
//        Point<float>(0, img2.rows - 1)};
//   Points<float> corners1 =
//       {Point<float>(0, 0),
//        Point<float>(img1.cols - 1, 0),
//        Point<float>(img1.cols - 1, img1.rows - 1),
//        Point<float>(0, img1.rows - 1)};
//   cv::transform(corners, corners, M);
//   corners.insert(corners.end(), corners1.begin(), corners1.end());
//   auto v =
//       corners | std::views::transform([](const Point<float>& p) { return Point<int>(abs_ceil(p.x), abs_ceil(p.y)); });
//   Points<int> corners_int(v.begin(), v.end());
//   cv::Rect    rect = cv::boundingRect(corners_int);
//   std::for_each(corners.begin(), corners.end(), [&rect](Point<float>& p) {
//     p.x -= rect.x;
//     p.y -= rect.y;
//   });
//   cv::Mat result1(rect.height, rect.width, img1.type(), cv::Scalar(0, 0, 0));
//   img1.copyTo(result1(cv::Rect(corners[4].x, corners[4].y, img1.cols, img1.rows)));
//   cv::Mat       result2(rect.height, rect.width, img1.type(), cv::Scalar(0, 0, 0));
//   Points<float> from =
//       {Point<float>(0, 0),
//        Point<float>(img2.cols - 1, 0),
//        Point<float>(img2.cols - 1, img2.rows - 1),
//        Point<float>(0, img2.rows - 1)};
//   Points<float> to(corners.begin(), corners.begin() + 4);
//   cv::Mat       M = cv::estimateAffinePartial2D(from, to);
//   cv::warpAffine(img2, result2, M, result1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
//   cv::Mat avg;
//   cv::addWeighted(result1, 0.5, result2, 0.5, 0, avg);
//   cv::Mat mask1(rect.height, rect.width, img1_mask.type(), cv::Scalar(0));
//   img1_mask.copyTo(mask1(cv::Rect(corners[4].x, corners[4].y, img1_mask.cols, img1_mask.rows)));
//   cv::Mat mask2(rect.height, rect.width, img1_mask.type(), cv::Scalar(0));
//   cv::warpAffine(img2_mask, mask2, M, result1.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
//   cv::Mat res;
//   result1.copyTo(res, mask1);
//   result2.copyTo(res, mask2);
//   avg.copyTo(res, mask1 & mask2);
//   if(!fs::exists(temporary_save_path / "foo")) {
//     fs::create_directories(temporary_save_path / "foo");
//   }
//   cv::imwrite(
//       temporary_save_path / "foo"
//           / (lhs_img.get_img_stem().string() + "_" + rhs_img.get_img_stem().string() + "_avg.jpg"),
//       res);
// }
// auto v0 = points0 | std::views::transform([](const Point<float>& p) { return cv::KeyPoint(p.x, p.y, 1); });
//         auto v1 = points1 | std::views::transform([](const Point<float>& p) { return cv::KeyPoint(p.x, p.y, 1); });
//         std::vector<cv::KeyPoint> kpts0(v0.begin(), v0.end()), kpts1(v1.begin(), v1.end());
//         cv::Mat                   resultImg;
//         cv::drawMatches(img1, kpts0, img2, kpts1, inlier_matches, resultImg);

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

  std::pair<Node, std::pair<float, float>> stitch_adjacent_images(int idx0) {
    std::string extension   = nodes[idx0].img.get_img_extension().string();
    cv::Mat     center_img  = nodes[idx0].img.get().get();
    cv::Mat     center_mask = nodes[idx0].mask.get().get();
    float       delta_x{0.0f}, delta_y{0.0f};
    for(const auto& [idx1, edge] : adjacent[idx0]) {
      cv::Mat                  append_img      = nodes[idx1].img.get().get();
      cv::Mat                  append_img_mask = nodes[idx1].mask.get().get();
      cv::Mat                  M               = edge.M;
      std::vector<cv::Point2f> corners =
          {cv::Point2f(0, 0),
           cv::Point2f(append_img.cols - 1, 0),
           cv::Point2f(append_img.cols - 1, append_img.rows - 1),
           cv::Point2f(0, append_img.rows - 1)};
      std::vector<cv::Point> center_corners =
          {cv::Point(0, 0),
           cv::Point(center_img.cols - 1, 0),
           cv::Point(center_img.cols - 1, center_img.rows - 1),
           cv::Point(0, center_img.rows - 1)};
      cv::transform(corners, corners, M);
      std::vector<cv::Point> corners_int;
      for(const auto& p : corners) {
        corners_int.push_back(cv::Point(Ortho::abs_ceil(p.x), Ortho::abs_ceil(p.y)));
      }
      corners_int.insert(corners_int.end(), center_corners.begin(), center_corners.end());
      cv::Rect rect = cv::boundingRect(corners_int);
      M.at<float>(0, 2) -= rect.x;
      M.at<float>(1, 2) -= rect.y;
      delta_x -= rect.x;
      delta_y -= rect.y;
      cv::Mat result(rect.height, rect.width, center_img.type(), cv::Scalar(0, 0, 0));
      cv::Mat mask_result(rect.height, rect.width, center_mask.type(), cv::Scalar(0));
      center_img.copyTo(result(cv::Rect(-rect.x, -rect.y, center_img.cols, center_img.rows)));
      center_mask.copyTo(mask_result(cv::Rect(-rect.x, -rect.y, center_mask.cols, center_mask.rows)));
      cv::Mat warped, mask_warped;
      cv::warpAffine(append_img, warped, M, result.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
      cv::warpAffine(append_img_mask, mask_warped, M, result.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      cv::Mat mixed;
      cv::addWeighted(result, 0.5, warped, 0.5, 0, mixed);
      cv::Mat mask_mixed = mask_result & mask_warped;
      mixed.copyTo(result, mask_mixed);
      cv::Mat mask_warped_unique = ~mask_result & mask_warped;
      warped.copyTo(result, mask_warped_unique);
      center_img  = result;
      center_mask = mask_result | mask_warped;
    }
    std::string uuid = generate_uuid_v4();
    return std::make_pair(
        Node{
            Image(temporary_save_path / (uuid + extension), std::move(center_img)),
            Image(temporary_save_path / (uuid + "_mask" + extension), std::move(center_mask))},
        std::make_pair(delta_x, delta_y));
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
    if(adjacent.empty()) {
      throw std::runtime_error("No image!");
    }
    std::erase_if(adjacent, [](const auto& pair) { return pair.second.size() == 0; });
    int  priority = 0;
    auto idx_     = find_max_out_degree(priority);
    for(; adjacent.size() > 1 && idx_; idx_ = find_max_out_degree(++priority)) {
      int idx                   = idx_.value();
      auto [edge, pair]         = stitch_adjacent_images(idx);
      cv::Mat M_new_idx         = cv::Mat::eye(3, 3, CV_32FC1);
      M_new_idx.at<float>(0, 2) = -pair.first;
      M_new_idx.at<float>(1, 2) = -pair.second;
      nodes.push_back(std::move(edge));
      for(auto& [j, edge_idx_j] : adjacent[idx]) {
        if(j == idx) {
          continue;
        }
        for(auto& [l, edge_j_l] : adjacent[j]) {
          if(l == idx || l == j || adjacent[idx].contains(l)) {
            continue;
          }
          cv::Mat M_idx_j = edge_idx_j.M;
          cv::vconcat(M_idx_j, cv::Mat{(cv::Mat_<float>(1, 3) << 0, 0, 1)}, M_idx_j);
          cv::Mat M_new_l    = cv::Mat{edge_j_l.M * M_idx_j * M_new_idx}.rowRange(0, 2);
          auto&   edge_l_idx = adjacent[l][nodes.size() - 1];
          cv::invertAffineTransform(M_new_l, edge_l_idx.M);
          edge_l_idx.priority = edge_j_l.priority + 1;
          auto& edge_idx_l    = adjacent[nodes.size() - 1][l];
          edge_idx_l.M        = M_new_l;
          edge_idx_l.priority = edge_j_l.priority + 1;
          adjacent[l].erase(j);
        }
        adjacent.erase(j);
      }
      adjacent.erase(idx);
    }
    cv::Mat stitched_img;
    nodes.back().img.get().get().copyTo(stitched_img);
    return stitched_img;
  }
};

} // namespace Ortho
#endif