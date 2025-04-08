#ifndef ORTHO_PIPELINE_HPP
#define ORTHO_PIPELINE_HPP

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "knn.hpp"
#include "log.hpp"
#include "matcher.hpp"
#include "matchpair.hpp"
#include "progress.hpp"

namespace fs = std::filesystem;

namespace Ortho {

struct Exiv2XmpParserInitializer {
  Exiv2XmpParserInitializer() { Exiv2::XmpParser::initialize(); }

  ~Exiv2XmpParserInitializer() { Exiv2::XmpParser::terminate(); }
};

struct MultiThreadProcess {
private:

  Progress                  progress;
  std::vector<fs::path>     img_paths;
  fs::path                  output_dir, temporary_save_dir;
  ImgsData                  imgs_data;
  Exiv2XmpParserInitializer exiv2_xmp_parser_initializer;
  MatchPairs                match_pairs;

  static auto generate_start_end(unsigned int total, unsigned int divisor) {
    int  base      = total / divisor;
    int  remainder = total % divisor;
    auto sequence =
        std::views::iota(0u, divisor) | std::views::transform([=](int i) { return i < remainder ? base + 1 : base; });
    std::vector<int> cumulative{0};
    std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
    return std::views::iota(0u, divisor)
           | std::views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
  }

  auto run(std::function<void(int)>&& process) {
    std::vector<std::thread> threads;
    progress.rerun();
    auto v = generate_start_end(img_paths.size(), std::thread::hardware_concurrency())
             | std::views::transform([this, &process](auto&& start_end) {
                 auto&& [start, end] = start_end;
                 return std::thread([this, start, end, &process]() {
                   for(int i = start; i < end; i++, progress.update()) {
                     process(i);
                   }
                 });
               });
    threads.assign(v.begin(), v.end());
    for(auto&& thread : threads) {
      if(thread.joinable()) {
        thread.join();
      }
    }
  }

  MatchPairs find_neighbors(const int k = 20) {
    auto knn =
        KNN(k,
            imgs_data.get() | std::views::transform([](auto&& data) { return data.get_coord(); }) | std::views::common);
    std::vector<std::vector<MatchPair>> matches(imgs_data.size());
    run([this, &knn, &matches](int i) {
      auto neighbors = knn.find_nearest_neighbour(i);
      for(auto&& neighbour : neighbors) {
        if(i < neighbour) {
          matches[i].emplace_back(i, neighbour);
        } else {
          matches[i].emplace_back(neighbour, i);
        }
      }
    });
    auto                v = matches | std::views::join | std::views::common;
    std::set<MatchPair> match_set(v.begin(), v.end());
    return std::vector<MatchPair>(match_set.begin(), match_set.end());
  }

public:

  MultiThreadProcess(fs::path input_dir, fs::path output_dir, fs::path temporary_save_dir) :
      output_dir(output_dir), temporary_save_dir(temporary_save_dir) {
    std::transform(
        fs::directory_iterator(input_dir),
        fs::directory_iterator(),
        std::back_inserter(img_paths),
        [](const auto& entry) { return entry.path(); });
    progress.reset(img_paths.size());
  }

  void get_image_info() {
    run([this](int i) {
      auto&& img_path = img_paths[i];
      auto   res      = ImgDataFactory::build(img_path, temporary_save_dir);
      if(!res) {
        ERROR("Error: {} could not be processed", img_path.string());
        return;
      }
      imgs_data.push_back(std::move(res.value()));
    });
    imgs_data.find_and_set_reference_coord();
  }

  void match(int neighbor_proposal = 20, float iou_threshold = 0.5) {
    MESSAGE("Finding image pairs with neighbor proposal {}", neighbor_proposal);
    auto match_pairs_ = find_neighbors(neighbor_proposal);
    MESSAGE("Found {} image pairs", match_pairs_.size());
    MESSAGE("Filtering image pairs with IoU threshold {}", iou_threshold);
    auto       v = match_pairs_ | std::views::filter([this, iou_threshold](auto&& pair) {
               auto &img0 = imgs_data[pair.first], &img1 = imgs_data[pair.second];
               if(cv::isContourConvex(img0.get_spans()) && cv::isContourConvex(img1.get_spans())) {
                 return Ortho::iou(img0.get_spans(), img1.get_spans()) >= iou_threshold;
               }
               WARN("Image {} and {} has non-convex span", img0.get_img_name().string(), img1.get_img_name().string());
               WARN(
                   "Image1 yaw: {}, pitch: {}, roll: {}",
                   img0.get_yaw().degrees(),
                   img0.get_pitch().degrees(),
                   img0.get_roll().degrees());
               WARN(
                   "Image2 yaw: {}, pitch: {}, roll: {}",
                   img1.get_yaw().degrees(),
                   img1.get_pitch().degrees(),
                   img1.get_roll().degrees());
               return false;
             });
    MatchPairs filtered_match_pairs(v.begin(), v.end());
    MESSAGE("Found {} image pairs after filtering", filtered_match_pairs.size());
    MESSAGE("Matching image pairs");
    Matcher matcher = matcher_factory<SuperPointExtractor>(temporary_save_dir);
    matcher.match(filtered_match_pairs, imgs_data, progress);
    std::ranges::move(
        filtered_match_pairs | std::views::filter([](auto&& pair) { return pair.valid; }),
        std::back_inserter(match_pairs));
  }

  void stitch() {}
};

} // namespace Ortho

#endif