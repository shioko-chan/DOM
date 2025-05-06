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
#include "algo/dsm.hpp"
#include "algo/filter.hpp"
#include "algo/knn.hpp"
#include "algo/tri.hpp"
#include "config.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "nn/matcher.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

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

public:

  Pipeline(const fs::path& input_dir, fs::path output_dir, fs::path temporary_save_path) :
      output_dir(std::move(output_dir)), temporary_save_path(std::move(temporary_save_path)) {
    for(const auto& entry : fs::directory_iterator(input_dir)) {
      img_paths.push_back(entry.path());
    }
  }

  void get_image_info() { imgs_data.delay_initialize(img_paths, temporary_save_path, progress); }

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
    auto match_pairs_ = find_neighbors(imgs_data, neighbor_proposal);
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
    k               = imgs_data.K();
#ifdef ENABLE_VISUALIZE_OUTPUT
    auto res = triangulation(match_pairs, imgs_data, progress, temporary_save_path);
#else
    auto res = triangulation(match_pairs, imgs_data, progress);
#endif
    std::cout << "R: " << r << '\n';
    std::cout << "t: " << t << '\n';
    std::cout << "K: " << k << '\n';
    std::cout << "res: " << res.size() << '\n';

    THIS_MESSAGE("Filtering outliers");
#ifdef ENABLE_VISUALIZE_OUTPUT
    filter_outliers(&res, temporary_save_path / "f1.pcd");
#else
    filter_outliers(&res);
#endif

    THIS_MESSAGE("Smoothing surface");
#ifdef ENABLE_VISUALIZE_OUTPUT
    smooth_surface(&res, temporary_save_path / "s1.pcd");
#else
    smooth_surface(&res);
#endif
    filter_near_observes(imgs_data, &res);
    filter_too_few_points(&res);
    std::unordered_set<int> observation_ids;
    for(const auto& tri_res : res) {
      for(const auto& [idx, _] : tri_res.pnt2d_idx_vec) {
        observation_ids.insert(idx);
      }
    }
    for(int idx = 0; idx < imgs_data.size(); ++idx) {
      if(!observation_ids.contains(idx)) {
        imgs_data[idx].set_invalid();
      }
    }
    ba(imgs_data, &res);
    export_pcd(temporary_save_path / "ba.pcd", tri_res_vec2point_cloud(res));

    r = img.R_w2c();
    t = img.t_w2c();
    k = imgs_data.K();
    std::cout << "R: " << r << '\n';
    std::cout << "t: " << t << '\n';
    std::cout << "K: " << k << '\n';
    std::cout << "d:" << imgs_data.D() << "\n";

    THIS_MESSAGE("Generating DSM");
    auto    cloud = tri_res_vec2point_cloud(res);
    cv::Mat dsm   = pointcloud_to_dsm(cloud);
    save_dsm_as_image(dsm, temporary_save_path / "dsm.png");
  }

  // void stitch() {
  //   THIS_MESSAGE("Stitching images");
  //   Stitcher stitcher(temporary_save_path);
  //   auto     stitched_img = stitcher.stitch(imgs_data, progress);
  //   if(stitched_img.empty()) {
  //     THIS_LOG_ERROR("Stitching failed");
  //     return;
  //   }
  //   fs::path stitched_img_path = output_dir / "stitched_image.jpg";
  //   cv::imwrite(stitched_img_path.string(), stitched_img);
  //   THIS_MESSAGE("Stitched image saved to {}", stitched_img_path.string());
  // }
};

} // namespace Ortho

#endif
