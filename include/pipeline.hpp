#ifndef ORTHO_PIPELINE_HPP
#define ORTHO_PIPELINE_HPP

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <ranges>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "algo/ba.hpp"
#include "algo/knn.hpp"
#include "algo/tri.hpp"
#include "config.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "nn/matcher.hpp"
#include "stitcher.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "algo/filter.hpp"

namespace Ortho {

namespace fs = std::filesystem;

class Pipeline {
private:

  struct Exiv2XmpParserInitializer {
    Exiv2XmpParserInitializer() { Exiv2::XmpParser::initialize(); }

    Exiv2XmpParserInitializer(const Exiv2XmpParserInitializer&)                    = delete;
    Exiv2XmpParserInitializer(Exiv2XmpParserInitializer&&)                         = delete;
    auto operator=(const Exiv2XmpParserInitializer&) -> Exiv2XmpParserInitializer& = delete;
    auto operator=(Exiv2XmpParserInitializer&&) -> Exiv2XmpParserInitializer&      = delete;

    ~Exiv2XmpParserInitializer() { Exiv2::XmpParser::terminate(); }
  };

  Progress                  progress;
  std::vector<fs::path>     img_paths;
  fs::path                  output_dir, temporary_save_path;
  ImgsData                  imgs_data;
  Exiv2XmpParserInitializer exiv2_xmp_parser_initializer;
  MatchPairs                match_pairs;

  auto find_neighbors(const int k_neighbors = 8) -> MatchPairs {
    auto knn = KNN<double>(k_neighbors, imgs_data.get() | std::views::transform([](auto&& data) noexcept {
                                          return data.get_coord();
                                        }) | std::views::common);
    std::vector<std::vector<MatchPair>> matches(imgs_data.size());
    run(
        imgs_data.size(),
        [this, &knn, &matches](int idx) noexcept {
          auto neighbors = knn.find_nearest_neighbour(idx);
          for(auto&& neighbour : neighbors) {
            if(idx < neighbour) {
              matches[idx].emplace_back(idx, neighbour);
            } else {
              matches[idx].emplace_back(neighbour, idx);
            }
          }
        },
        progress);
    auto                view = matches | std::views::join | std::views::common;
    std::set<MatchPair> match_set(view.begin(), view.end());
    return {match_set.begin(), match_set.end()};
  }

public:

  Pipeline(const fs::path& input_dir, fs::path output_dir, fs::path temporary_save_path) :
      output_dir(std::move(output_dir)), temporary_save_path(std::move(temporary_save_path)) {
    std::transform(
        fs::directory_iterator(input_dir),
        fs::directory_iterator(),
        std::back_inserter(img_paths),
        [](const auto& entry) noexcept { return entry.path(); });
    progress.reset(static_cast<int>(img_paths.size()));
  }

  void get_image_info() {
    run(
        img_paths.size(),
        [this](int idx) noexcept {
          auto&& img_path = img_paths[idx];
          if(!ImgDataFactory::validate(img_path)) {
            return;
          }
          imgs_data.push_back(ImgDataFactory::build(img_path, temporary_save_path));
        },
        progress);
    imgs_data.find_and_set_reference_coord();
  }

  void rotate_rectify() {
    run(
        imgs_data.size(),
        [this](int idx) noexcept {
          imgs_data[idx].rotate_rectify();
#ifdef ENABLE_VISUALIZE_OUTPUT
          cv::imwrite(
              temporary_save_path / imgs_data[idx].rotated_img().get_img_name().string(),
              imgs_data[idx].rotated_img().get().get());
#endif
        },
        progress);
  }

  void match(int neighbor_proposal = 8) {
    THIS_MESSAGE("Finding image pairs with neighbor proposal {}", neighbor_proposal);
    auto match_pairs_ = find_neighbors(neighbor_proposal);
    THIS_MESSAGE("Found {} image pairs", match_pairs_.size());
    if(FEATURE_EXTRACTION_METHOD == method_t::SUPERPOINT) {
      THIS_MESSAGE("Using SuperPoint feature extraction");
      Matcher matcher = matcher_factory<SuperPointExtractor>(temporary_save_path);
      matcher.match(match_pairs_, imgs_data, progress);
    } else if(FEATURE_EXTRACTION_METHOD == method_t::DISK) {
      THIS_MESSAGE("Using DISK feature extraction");
      Matcher matcher = matcher_factory<DiskExtractor>(temporary_save_path);
      matcher.match(match_pairs_, imgs_data, progress);
    } else {
      THIS_LOG_ERROR("Unknown feature extraction method");
      return;
    }
    match_pairs.clear();
    std::ranges::move(
        match_pairs_ | std::views::filter([](auto&& pair) noexcept { return pair.valid; }),
        std::back_inserter(match_pairs));
  }

  void triangulate() {
    cv::Mat     r;
    cv::Mat     t;
    cv::Mat     k;
    const auto& img = imgs_data[30];
    r               = img.R_w2c();
    t               = img.t_w2c();
    k               = img.K();
    auto res        = triangulation(match_pairs, imgs_data, progress);
    std::cout << "R: " << r << '\n';
    std::cout << "t: " << t << '\n';
    std::cout << "K: " << k << '\n';
    filter_outliers(&res);
    smooth_surface(&res);
    ba(imgs_data, &res);
    r = img.R_w2c();
    t = img.t_w2c();
    k = img.K();
    std::cout << "R: " << r << '\n';
    std::cout << "t: " << t << '\n';
    std::cout << "K: " << k << '\n';
  }

  void stitch() {
    THIS_MESSAGE("Stitching images");
    Stitcher stitcher(temporary_save_path);
    auto     stitched_img = stitcher.stitch(imgs_data, progress);
    if(stitched_img.empty()) {
      THIS_LOG_ERROR("Stitching failed");
      return;
    }
    fs::path stitched_img_path = output_dir / "stitched_image.jpg";
    cv::imwrite(stitched_img_path.string(), stitched_img);
    THIS_MESSAGE("Stitched image saved to {}", stitched_img_path.string());
  }
};

} // namespace Ortho

#endif
