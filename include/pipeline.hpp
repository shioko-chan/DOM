#ifndef SKYMERGE_PIPELINE_HPP
#define SKYMERGE_PIPELINE_HPP

#include <filesystem>
#include <ranges>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/impl/point_types.hpp>

#include "algo/ba.hpp"
#include "algo/filter.hpp"
#include "algo/knn.hpp"
#include "algo/stitch.hpp"
#include "algo/tracks.hpp"
#include "algo/tri.hpp"
#include "config.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "nn/matcher.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"
#include "types.hpp"

namespace SkyMerge {

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

  Progress                           progress;
  std::vector<std::filesystem::path> img_paths;
  std::filesystem::path              output_dir, temporary_save_path;
  Exiv2XmpParserInitializer          exiv2_xmp_parser_initializer;

public:

  Pipeline(
      const std::filesystem::path& input_dir,
      std::filesystem::path        output_dir,
      std::filesystem::path        temporary_save_path) noexcept :
      output_dir(std::move(output_dir)), temporary_save_path(std::move(temporary_save_path)) {
    for(const auto& entry : std::filesystem::directory_iterator(input_dir)) {
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
    THIS_MESSAGE("[Pipeline] Finding image pairs with neighbor proposal {}", neighbor_proposal);
    auto match_pairs_ = find_neighbors(imgs_data, neighbor_proposal);
    THIS_MESSAGE("[Pipeline] Found {} image pairs", match_pairs_.size());
    if(FEATURE_EXTRACTION_METHOD == method_t::SUPERPOINT) {
      THIS_MESSAGE("[Pipeline] Using SuperPoint feature extraction");
      Matcher matcher = matcher_factory<SuperPointExtractor>(temporary_save_path);
      matcher.match(match_pairs_, imgs_data, progress);
    } else if(FEATURE_EXTRACTION_METHOD == method_t::DISK) {
      THIS_MESSAGE("[Pipeline] Using DISK feature extraction");
      Matcher matcher = matcher_factory<DiskExtractor>(temporary_save_path);
      matcher.match(match_pairs_, imgs_data, progress);
    } else {
      THIS_LOG_ERROR("[Pipeline] Unknown feature extraction method");
      return {};
    }
    auto view = match_pairs_ | std::views::filter([](auto&& pair) noexcept { return pair.valid; });
    for(const auto& img_data : imgs_data) {
      img_data.rotated_img().release_mem();
    }
    return {view.begin(), view.end()};
  }

  [[nodiscard]] auto triangulate(ImgsData& imgs_data, MatchPairs& match_pairs) noexcept {
#ifdef ENABLE_VISUALIZE_OUTPUT
    auto tracks = build_track(match_pairs, progress);
    // Filter::filter_track_too_few_observations(&tracks, 2);
    auto track_point_vec = triangulation(tracks, imgs_data, progress, temporary_save_path);
    THIS_LOG_INFO("[Pipeline] Filtering outliers using statistical method");
    // Filter::filter_outliers_statistical(&track_point_vec, temporary_save_path / "fs1.pcd");
    THIS_LOG_INFO("[Pipeline] Filtering outliers using radius method");
    // Filter::filter_outliers_radius(&track_point_vec, temporary_save_path / "f2.pcd");
    // Filter::filter_near_observations(&track_point_vec, imgs_data);
    // Filter::filter_track_too_few_observations(&track_point_vec);
    // Filter::filter_invalid_image(track_point_vec, imgs_data);
    BA::ba(imgs_data, &track_point_vec, 3.0);
    export_pcd(temporary_save_path / "ba.pcd", track_point_vec2point_cloud(track_point_vec));
    // Filter::filter_reprojection_error(&track_point_vec, imgs_data, 3.0);
    // Filter::filter_track_too_few_observations(&track_point_vec);
    Filter::filter_outliers_radius(&track_point_vec, temporary_save_path / "f3.pcd");
    Filter::filter_outliers_statistical(&track_point_vec, temporary_save_path / "fs2.pcd");
      // Filter::filter_invalid_image(track_point_vec, imgs_data);
      // BA::ba(imgs_data, &track_point_vec, -1);
      // export_pcd(temporary_save_path / "ba1.pcd", track_point_vec2point_cloud(track_point_vec));
      // Filter::filter_reprojection_error(&track_point_vec, imgs_data, 1.0);
      // Filter::filter_track_too_few_observations(&track_point_vec);
      // Filter::filter_outliers_radius(&track_point_vec, temporary_save_path / "f3.pcd");
      // Filter::filter_outliers_statistical(&track_point_vec, temporary_save_path / "fs3.pcd");
      // Filter::filter_invalid_image(track_point_vec, imgs_data);
#else
    auto tracks = build_track(match_pairs, progress);
    auto track_point_vec = triangulation(tracks, imgs_data, progress);
    THIS_LOG_INFO("[Pipeline] Filtering outliers using statistical method");
    THIS_LOG_INFO("[Pipeline] Filtering outliers using radius method");
    BA::ba(imgs_data, &track_point_vec, 3.0);
    Filter::filter_outliers_radius(&track_point_vec);
    Filter::filter_outliers_statistical(&track_point_vec);
#endif
    return track_point_vec2point_cloud(track_point_vec);
  }

  void stitch(ImgsData& imgs_data, const PointCloudPtr& point_cloud) {
    THIS_MESSAGE("[Pipeline] Stitching images");
    cv::Mat texture = Stitcher::stitch(imgs_data, point_cloud, progress, GRID_LENGTH);
    if(texture.empty()) {
      THIS_LOG_ERROR("[Pipeline] Stitching failed");
      return;
    }
    std::filesystem::path panorama_path = output_dir / "stitched_image.jpg";
    cv::imwrite(panorama_path.string(), texture);
    THIS_MESSAGE("[Pipeline] Stitched image saved to {}", panorama_path.string());
  }
};

} // namespace SkyMerge

#endif
