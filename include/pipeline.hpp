#ifndef SKYMERGE_PIPELINE_HPP
#define SKYMERGE_PIPELINE_HPP

#include <filesystem>
#include <ranges>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "algo/ba.hpp"
#include "algo/filter.hpp"
#include "algo/knn.hpp"
#include "algo/stitch.hpp"
#include "algo/tri.hpp"
#include "config.hpp"
#include "ds/dsm.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "nn/matcher.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace SkyMerge {

namespace fs = std::filesystem;

class Pipeline {
private:

  struct Exiv2XmpParserInitializer {
    Exiv2XmpParserInitializer() noexcept { Exiv2::XmpParser::initialize(); }

    Exiv2XmpParserInitializer(const Exiv2XmpParserInitializer&)                    = delete;
    Exiv2XmpParserInitializer(Exiv2XmpParserInitializer&&)                         = delete;
    auto operator=(const Exiv2XmpParserInitializer&) -> Exiv2XmpParserInitializer& = delete;
    auto operator=(Exiv2XmpParserInitializer&&) -> Exiv2XmpParserInitializer&      = delete;

    ~Exiv2XmpParserInitializer() noexcept { Exiv2::XmpParser::terminate(); }
  };

  Progress                  progress;
  std::vector<fs::path>     img_paths;
  fs::path                  output_dir, temporary_save_path;
  Exiv2XmpParserInitializer exiv2_xmp_parser_initializer;

public:

  Pipeline(const fs::path& input_dir, fs::path output_dir, fs::path temporary_save_path) noexcept :
      output_dir(std::move(output_dir)), temporary_save_path(std::move(temporary_save_path)) {
    for(const auto& entry : fs::directory_iterator(input_dir)) {
      img_paths.push_back(entry.path());
    }
  }

  [[nodiscard]] auto get_image_info() noexcept -> ImgsData { return {img_paths, temporary_save_path, progress}; }

  void rotate_rectify(ImgsData& imgs_data) noexcept {
    run(
        imgs_data.size(),
        [&imgs_data, this](int idx) noexcept {
          imgs_data[idx].rotate_rectify();
#ifdef ENABLE_VISUALIZE_OUTPUT
          cv::imwrite(
              temporary_save_path / imgs_data[idx].rotated_img().get_img_name().string(),
              imgs_data[idx].rotated_img().get().get());
#endif
        },
        progress);
  }

  [[nodiscard]] auto match(ImgsData& imgs_data, int neighbor_proposal = 8) noexcept -> MatchPairs {
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
      return {};
    }
    auto view = match_pairs_ | std::views::filter([](auto&& pair) noexcept { return pair.valid; });
    return {view.begin(), view.end()};
  }

  [[nodiscard]] auto triangulate(ImgsData& imgs_data, MatchPairs& match_pairs) noexcept -> DSM {
#ifdef ENABLE_VISUALIZE_OUTPUT
    auto res = triangulation(match_pairs, imgs_data, progress, temporary_save_path);
    THIS_MESSAGE("Filtering outliers statistical");
    Filter::filter_outliers_statistical(&res, temporary_save_path / "f1.pcd");
    THIS_MESSAGE("Filtering outliers radius");
    Filter::filter_outliers_radius(&res, temporary_save_path / "f2.pcd");
    Filter::filter_near_observes(imgs_data, &res);
    Filter::filter_too_few_points(&res);
    Filter::filter_invalid_image(res, imgs_data);
    THIS_MESSAGE("Smoothing surface");
    auto smoothed = Filter::smooth_surface(&res, temporary_save_path / "s1.pcd");
    BA::ba(imgs_data, &res);
    export_pcd(temporary_save_path / "ba.pcd", tri_res_vec2point_cloud(res));
    Filter::filter_outliers_radius(&res, temporary_save_path / "f3.pcd");
#else
    auto res = triangulation(match_pairs, imgs_data, progress);
    THIS_MESSAGE("Filtering outliers statistical");
    Filter::filter_outliers_statistical(&res);
    THIS_MESSAGE("Smoothing surface");
    Filter::smooth_surface(&res);
    THIS_MESSAGE("Filtering outliers radius");
    Filter::filter_outliers_radius(&res);
    Filter::filter_near_observes(imgs_data, &res);
    Filter::filter_too_few_points(&res);
    Filter::filter_invalid_image(res, imgs_data);
    BA::ba(imgs_data, &res);
    Filter::filter_outliers_radius(&res);
#endif
    THIS_MESSAGE("Generating DSM");
    return DSM{tri_res_vec2point_cloud(res), RESOLUTION};
  }

  void stitch(ImgsData& imgs_data, DSM& dsm) {
    THIS_MESSAGE("Stitching images");
    cv::Mat  texture       = DSMStitcher::stitch(imgs_data, dsm, progress);
    fs::path panorama_path = output_dir / "stitched_image.jpg";
    cv::imwrite(panorama_path.string(), texture);
    THIS_MESSAGE("Stitched image saved to {}", panorama_path.string());
  }
};

} // namespace SkyMerge

#endif
