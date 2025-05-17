#ifndef SUPERPOINT_LIGHTGLUE_MATCHER_HPP
#define SUPERPOINT_LIGHTGLUE_MATCHER_HPP

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "ds/imgdata.hpp"
#include "ds/matchpair.hpp"
#include "nn/extractor.hpp"
#include "nn/ort.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace SkyMerge {

template <typename Extractor>
class Matcher {
private:

  InferEnv              lightglue;
  std::filesystem::path temporary_save_path;
  Extractor             extractor;
  std::vector<float>    kpts0, kpts1, desc0, desc1;

  using IdxIdxScoreUMap = std::unordered_map<size_t, std::pair<size_t, float>>;

  auto filter_matches_by_score_precise(const std::int64_t* matches, const float* scores, size_t cnt) -> Matches {
    IdxIdxScoreUMap                     match_score0;
    IdxIdxScoreUMap                     match_score1;
    const std::span<const float>        scores_span(scores, cnt);
    const std::span<const std::int64_t> matches_span(matches, cnt * 2);
    for(size_t idx = 0; idx < cnt; ++idx) {
      if(scores_span[idx] >= LIGHTGLUE_THRESHOLD) {
        auto idx0 = matches_span[idx * 2];
        auto idx1 = matches_span[(idx * 2) + 1];
        if(!match_score0.contains(idx0) || match_score0[idx0].second < scores_span[idx]) {
          match_score0[idx0] = std::make_pair(idx1, scores_span[idx]);
        }
      }
    }
    for(auto&& [idx0, pair] : match_score0) {
      auto idx1 = pair.first;
      if(!match_score1.contains(idx1) || match_score1[idx1].second < pair.second) {
        match_score1[idx1] = std::make_pair(idx0, pair.second);
      }
    }
    auto view = match_score1 | std::views::transform([](const auto& pair) noexcept {
                  return Match{pair.second.first, pair.first, pair.second.second};
                });
    return Matches{view.begin(), view.end()};
  }

  auto filter_matches_by_score(const std::int64_t* matches, const float* scores, size_t cnt) -> Matches {
    auto view =
        std::views::iota(0UL, cnt)
        | std::views::filter([&scores](const auto& idx) noexcept { return scores[idx] >= LIGHTGLUE_THRESHOLD; })
        | std::views::transform([&matches, &scores](const auto& idx) noexcept {
            return Match{static_cast<size_t>(matches[idx * 2]), static_cast<size_t>(matches[(idx * 2) + 1]), scores[idx]};
          });
    return Matches{view.begin(), view.end()};
  }

  auto infer() {
    OrtValues    res      = lightglue.infer();
    const size_t cnt_     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const auto*  matches_ = res[lightglue.get_output_index("matches0")].GetTensorData<std::int64_t>();
    const auto*  scores_  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
  }

  auto infer_and_filter_by_score() -> Matches {
    OrtValues    res     = lightglue.infer();
    const size_t cnt     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const auto*  matches = res[lightglue.get_output_index("matches0")].GetTensorData<std::int64_t>();
    const auto*  scores  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score(matches, scores, cnt);
  }

  auto infer_and_filter_by_score_precise() -> Matches {
    OrtValues    res     = lightglue.infer();
    const size_t cnt     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const auto*  matches = res[lightglue.get_output_index("matches0")].GetTensorData<std::int64_t>();
    const auto*  scores  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score_precise(matches, scores, cnt);
  }

public:

  Matcher(const std::filesystem::path& temporary_save_path, const std::string& weight) :
      temporary_save_path(temporary_save_path), lightglue("[lightglue]", weight), extractor(temporary_save_path) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(static_cast<int>(imgs_data.size()));
    THIS_MESSAGE("Start extracting features.");
    for(auto& img_data : imgs_data) {
      extractor.extract_features(img_data);
      progress.update();
    }
    THIS_MESSAGE("Feature extraction finished.");
    progress.reset(static_cast<int>(pairs.size()));
    auto batches =
        pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) noexcept { return lhs.first == rhs.first; });
    for(auto&& batch : batches) {
      int      batch_cnt = 0;
      ImgData& lhs_img   = imgs_data[batch.front().first];
      if(!extractor.extract_features(lhs_img)) {
        THIS_LOG_INFO("Image {} has no valid feature!", lhs_img.rotated_img().get_img_name().string());
        continue;
      }
      auto [lhs_w, lhs_h] = lhs_img.rotated_img().get_size();
      auto lhs_features   = extractor.get_features_on_device(lhs_img);
      lightglue.set_input("kpts0", lhs_features.get_kpnts(), {1, static_cast<std::int64_t>(lhs_features.get_len()), 2});
      lightglue.set_input(
          "desc0",
          lhs_features.get_descs(),
          {1, static_cast<std::int64_t>(lhs_features.get_len()), Extractor::descriptor_size});
      for(auto&& pair : batch) {
        batch_cnt += 1;
        ImgData& rhs_img = imgs_data[pair.second];
        if(!extractor.extract_features(rhs_img)) {
          THIS_LOG_INFO("Image {} has no valid feature!", rhs_img.rotated_img().get_img_name().string());
          continue;
        }
        auto [rhs_w, rhs_h] = rhs_img.rotated_img().get_size();
        auto rhs_features   = extractor.get_features_on_device(rhs_img);
        lightglue.set_input("kpts1", rhs_features.get_kpnts(), {1, static_cast<std::int64_t>(rhs_features.get_len()), 2});
        lightglue.set_input(
            "desc1",
            rhs_features.get_descs(),
            {1, static_cast<std::int64_t>(rhs_features.get_len()), Extractor::descriptor_size});
        auto matches = infer_and_filter_by_score_precise();
        THIS_LOG_DEBUG(
            "Image {} and image {} have {} matches after threshold filter!",
            lhs_img.rotated_img().get_img_name().string(),
            rhs_img.rotated_img().get_img_name().string(),
            matches.size());
        if(matches.size() < MATCH_CNT_THRESHOLD) {
          THIS_LOG_INFO(
              "Image {} and image {} have too few matches after threshold filter: {}",
              lhs_img.rotated_img().get_img_name().string(),
              rhs_img.rotated_img().get_img_name().string(),
              matches.size());
          continue;
        }
        auto kpnt_lhs_v = matches | std::views::transform([&lhs_img](const auto& match) noexcept {
                            return lhs_img.get_kpnts()[match.lhs];
                          });
        auto kpnt_rhs_v = matches | std::views::transform([&rhs_img](const auto& match) noexcept {
                            return rhs_img.get_kpnts()[match.rhs];
                          });
        auto score_v    = matches | std::views::transform([](const auto& match) noexcept { return match.score; });
        Points<double>      kpnt_lhs{kpnt_lhs_v.begin(), kpnt_lhs_v.end()};
        Points<double>      kpnt_rhs{kpnt_rhs_v.begin(), kpnt_rhs_v.end()};
        std::vector<double> score{score_v.begin(), score_v.end()};
        cv::Mat             mask;
        cv::findFundamentalMat(kpnt_lhs, kpnt_rhs, cv::FM_RANSAC, 3.0, 0.99, mask);
        for(int idx = 0; idx < matches.size(); ++idx) {
          if(mask.at<uchar>(idx, 0) != 0) {
            pair.matches.push_back(std::move(matches[idx]));
          }
        }
        THIS_LOG_DEBUG(
            "Image {} and image {} have {} matches after RANSAC filter!",
            lhs_img.rotated_img().get_img_name().string(),
            rhs_img.rotated_img().get_img_name().string(),
            pair.matches.size());
        if(pair.matches.size() < MATCH_CNT_THRESHOLD) {
          THIS_LOG_INFO(
              "Image {} and image {} have too few matches after RANSAC filter: {}",
              lhs_img.rotated_img().get_img_name().string(),
              rhs_img.rotated_img().get_img_name().string(),
              pair.matches.size());
          continue;
        }
        pair.valid = true;
      }
      progress.update(batch_cnt);
    }
  }
};

template <typename E>
auto matcher_factory(const std::filesystem::path& temporary_save_path) -> Matcher<E> {
  if constexpr(std::is_same_v<E, DiskExtractor>) {
    return Matcher<E>(temporary_save_path, DISK_LIGHTGLUE_WEIGHT);
  } else if constexpr(std::is_same_v<E, SuperPointExtractor>) {
    return Matcher<E>(temporary_save_path, SUPERPOINT_LIGHTGLUE_WEIGHT);
  } else {
    static_assert(false, "Unknown extractor type");
  }
}

} // namespace SkyMerge
#endif
