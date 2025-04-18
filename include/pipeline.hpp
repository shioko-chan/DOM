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

#include "config.hpp"
#include "imgdata.hpp"
#include "knn.hpp"
#include "log.hpp"
#include "matcher.hpp"
#include "matchpair.hpp"
#include "progress.hpp"
#include "stitcher.hpp"
#include "tri.hpp"
#include "types.hpp"

namespace Ortho {

class Pipeline {
private:

  struct Exiv2XmpParserInitializer {
    Exiv2XmpParserInitializer() { Exiv2::XmpParser::initialize(); }

    ~Exiv2XmpParserInitializer() { Exiv2::XmpParser::terminate(); }
  };

  Progress                  progress;
  std::vector<fs::path>     img_paths;
  fs::path                  output_dir, temporary_save_path;
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

  auto run(size_t tasks, std::function<void(int)>&& process) {
    std::vector<std::thread> threads;
    progress.reset(tasks);
    auto v = generate_start_end(tasks, std::thread::hardware_concurrency())
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

  MatchPairs find_neighbors(const int k = 8) {
    auto knn =
        KNN(k,
            imgs_data.get() | std::views::transform([](auto&& data) { return data.get_coord(); }) | std::views::common);
    std::vector<std::vector<MatchPair>> matches(imgs_data.size());
    run(imgs_data.size(), [this, &knn, &matches](int i) {
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

  Pipeline(fs::path input_dir, fs::path output_dir, fs::path temporary_save_path) :
      output_dir(output_dir), temporary_save_path(temporary_save_path) {
    std::transform(
        fs::directory_iterator(input_dir),
        fs::directory_iterator(),
        std::back_inserter(img_paths),
        [](const auto& entry) { return entry.path(); });
    progress.reset(img_paths.size());
  }

  void get_image_info() {
    run(img_paths.size(), [this](int i) {
      auto&& img_path = img_paths[i];
      if(!ImgDataFactory::validate(img_path)) {
        return;
      }
      imgs_data.push_back(ImgDataFactory::build(img_path, temporary_save_path));
    });
    imgs_data.find_and_set_reference_coord();
  }

  void rotate_rectify() {
    run(imgs_data.size(), [this](int i) {
      imgs_data[i].rotate_rectify();
      // cv::imwrite(temporary_save_path / imgs_data[i].get_img_name().string(), imgs_data[i].img().get().get());
    });
  }

  void match(int neighbor_proposal = 8) {
    MESSAGE("Finding image pairs with neighbor proposal {}", neighbor_proposal);
    auto match_pairs_ = find_neighbors(neighbor_proposal);
    MESSAGE("Found {} image pairs", match_pairs_.size());
    if(FEATURE_EXTRACTION_METHOD == method_t::SUPERPOINT) {
      MESSAGE("Using SuperPoint feature extraction");
      Matcher matcher = matcher_factory<SuperPointExtractor>(temporary_save_path);
      matcher.match(match_pairs_, imgs_data, progress);
    } else if(FEATURE_EXTRACTION_METHOD == method_t::DISK) {
      MESSAGE("Using DISK feature extraction");
      Matcher matcher = matcher_factory<DiskExtractor>(temporary_save_path);
      matcher.match(match_pairs_, imgs_data, progress);
    } else {
      LOG_ERROR("Unknown feature extraction method");
      return;
    }
    std::ranges::move(
        match_pairs_ | std::views::filter([](auto&& pair) { return pair.valid; }), std::back_inserter(match_pairs));
  }

  void triangulate() { triangulation(match_pairs, imgs_data); }

  void stitch() {
    MESSAGE("Stitching images");
    Stitcher stitcher(match_pairs, imgs_data, temporary_save_path);
    auto     stitched_img = stitcher.stitch();
    if(stitched_img.empty()) {
      LOG_ERROR("Stitching failed");
      return;
    }
    fs::path stitched_img_path = output_dir / "stitched_image.jpg";
    cv::imwrite(stitched_img_path.string(), stitched_img);
    MESSAGE("Stitched image saved to {}", stitched_img_path.string());
  }
};

} // namespace Ortho

#endif