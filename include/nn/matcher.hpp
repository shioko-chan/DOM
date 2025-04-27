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
#include "tools/log.hpp"
#include "tools/ort.hpp"
#include "tools/progress.hpp"

namespace Ortho {

namespace fs = std::filesystem;

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
class Matcher {
private:

  using Feature  = typename E::Feature;
  using Features = typename E::Features;

  InferEnv           lightglue;
  fs::path           temporary_save_path;
  E                  extractor;
  std::vector<float> kpts0, kpts1, desc0, desc1;

  auto set_input(
      const Features&     features,
      const std::string&  kpts_name,
      std::vector<float>* kpts,
      const std::string&  desc_name,
      std::vector<float>* desc) -> bool {
    if(features.empty()) {
      return false;
    }
    auto view0 = features | std::views::transform([](const auto& feature) noexcept {
                   return std::array<float, 2>{static_cast<float>(feature.x), static_cast<float>(feature.y)};
                 })
                 | std::views::join | std::views::common;
    kpts->assign(view0.begin(), view0.end());
    auto view1 = features | std::views::transform([&features](const auto& feature) noexcept { return feature.desc; })
                 | std::views::join | std::views::common;
    desc->assign(view1.begin(), view1.end());
    lightglue.set_input(kpts_name, *kpts, {1, static_cast<unsigned int>(features.size()), 2});
    lightglue.set_input(
        desc_name,
        *desc,
        {1, static_cast<unsigned int>(features.size()), static_cast<unsigned int>(Feature::descriptor_size)});
    return true;
  }

  using IdxIdxScoreUMap = std::unordered_map<size_t, std::pair<size_t, float>>;

  auto filter_matches_by_score_precise(const int64_t* matches, const float* scores, size_t cnt) -> Matches {
    IdxIdxScoreUMap                match_score0;
    IdxIdxScoreUMap                match_score1;
    const std::span<const float>   scores_span(scores, cnt);
    const std::span<const int64_t> matches_span(matches, cnt * 2);
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

  auto filter_matches_by_score(const int64_t* matches, const float* scores, size_t cnt) -> Matches {
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
    const auto*  matches_ = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const auto*  scores_  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
  }

  auto infer_and_filter_by_score() -> Matches {
    OrtValues    res     = lightglue.infer();
    const size_t cnt     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const auto*  matches = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const auto*  scores  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score(matches, scores, cnt);
  }

  auto infer_and_filter_by_score_precise() -> Matches {
    OrtValues    res     = lightglue.infer();
    const size_t cnt     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const auto*  matches = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const auto*  scores  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score_precise(matches, scores, cnt);
  }

  static auto feature2point(cv::Size size) {
    auto [width, height] = size;
    const double wf2     = width / 2.;
    const double hf2     = height / 2.;
    const double max2    = std::max(wf2, hf2);
    return std::views::transform([wf2, hf2, max2](const Feature& feature) noexcept {
      return Point<double>{(feature.x * max2) + wf2, (feature.y * max2) + hf2};
    });
  }

  using DMatch    = cv::DMatch;
  using DMatches  = std::vector<DMatch>;
  using KeyPoints = std::vector<cv::KeyPoint>;

  static auto draw_matchlines(
      ImgData&        img_lhs,
      ImgData&        img_rhs,
      const Matches&  matches,
      const Features& features_lhs,
      const Features& features_rhs) -> cv::Mat {
    auto           view_lhs = features_lhs | feature2point(img_lhs.get_size());
    auto           view_rhs = features_rhs | feature2point(img_rhs.get_size());
    auto           view     = matches | std::views::transform([](const auto& match) noexcept {
                  return DMatch(static_cast<int>(match.lhs), static_cast<int>(match.rhs), match.score);
                });
    Points<double> points_lhs{view_lhs.begin(), view_lhs.end()};
    Points<double> points_rhs{view_rhs.begin(), view_rhs.end()};
    DMatches       d_matches{view.begin(), view.end()};
    cv::Mat        img0;
    {
      auto guard = img_lhs.img().get();
      guard.get().copyTo(img0);
    }
    cv::Mat img1;
    {
      auto guard = img_rhs.img().get();
      guard.get().copyTo(img1);
    }
    auto points2keypoints = [](const auto& points) noexcept {
      return points
             | std::views::transform([](const auto& point) noexcept { return cv::KeyPoint(point.x, point.y, 1.); });
    };
    auto      v1_lhs = points2keypoints(points_lhs);
    auto      v1_rhs = points2keypoints(points_rhs);
    KeyPoints keypoints_lhs{view_lhs.begin(), view_lhs.end()};
    KeyPoints keypoints_rhs{v1_rhs.begin(), v1_rhs.end()};
    cv::Mat   res;
    cv::drawMatches(
        img0, keypoints_lhs, img1, keypoints_rhs, d_matches, res, cv::Scalar::all(-1), cv::Scalar(255, 255, 255));
    return res;
  }

public:

  Matcher(const fs::path& temporary_save_path, const std::string& weight) :
      temporary_save_path(temporary_save_path), lightglue("[lightglue]", weight), extractor(temporary_save_path) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(static_cast<int>(pairs.size()));
    auto batches =
        pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) noexcept { return lhs.first == rhs.first; });
    for(auto&& batch : batches) {
      int      batch_cnt    = 0;
      ImgData& lhs_img      = imgs_data[batch.front().first];
      Features lhs_features = std::move(extractor.get_features(lhs_img));
      if(!set_input(lhs_features, "kpts0", &kpts0, "desc0", &desc0)) {
        THIS_LOG_INFO("Image {} has no valid feature!", lhs_img.get_img_name().string());
        continue;
      }
      auto [lhs_w, lhs_h] = lhs_img.get_size();
      for(auto&& pair : batch) {
        batch_cnt += 1;
        ImgData& rhs_img      = imgs_data[pair.second];
        Features rhs_features = std::move(extractor.get_features(rhs_img));
        if(!set_input(rhs_features, "kpts1", &kpts1, "desc1", &desc1)) {
          THIS_LOG_INFO("Image {} has no valid feature!", lhs_img.get_img_name().string());
          continue;
        }
        auto [rhs_w, rhs_h] = rhs_img.get_size();
        auto matches        = infer_and_filter_by_score_precise();
#ifdef ENABLE_MIDDLE_OUTPUT
        cv::imwrite(
            temporary_save_path
                / std::format("{}_{}_matches.jpg", lhs_img.get_img_stem().string(), rhs_img.get_img_stem().string()),
            draw_matchlines(lhs_img, rhs_img, matches, lhs_features, rhs_features));
#endif
        const uint64_t len = matches.size();
        THIS_LOG_DEBUG(
            "Image {} and image {} have {} matches after threshold filter!",
            lhs_img.get_img_name().string(),
            rhs_img.get_img_name().string(),
            len);
        if(matches.size() < MATCH_CNT_THRESHOLD) {
          THIS_LOG_INFO(
              "Image {} and image {} have too few matches after threshold filter: {}",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              matches.size());
          continue;
        }
        auto kpnt_lhs =
            matches
            | std::views::transform([&lhs_features](const auto& match) noexcept { return lhs_features[match.lhs]; })
            | feature2point(lhs_img.get_size());
        auto kpnt_rhs =
            matches
            | std::views::transform([&rhs_features](const auto& match) noexcept { return rhs_features[match.rhs]; })
            | feature2point(rhs_img.get_size());
        auto score     = matches | std::views::transform([](const auto& match) noexcept { return match.score; });
        auto idx_lhs   = lhs_img.get_kpnts().append(kpnt_lhs);
        auto idx_rhs   = rhs_img.get_kpnts().append(kpnt_rhs);
        auto matches_v = std::views::zip(idx_lhs, idx_rhs, score) | std::views::transform([](auto&& idx) noexcept {
                           auto&& [i0, i1, score] = idx;
                           return Match{i0, i1, score};
                         });
        pair.matches.assign(matches_v.begin(), matches_v.end());
        Points<double> kpnt_lhs_f;
        Points<double> kpnt_rhs_f;
        std::ranges::copy(kpnt_lhs, std::back_inserter(kpnt_lhs_f));
        std::ranges::copy(kpnt_rhs, std::back_inserter(kpnt_rhs_f));
        pair.M     = cv::estimateAffinePartial2D(kpnt_lhs_f, kpnt_rhs_f);
        pair.valid = true;
      }
      progress.update(batch_cnt);
    }
  }
};

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
auto matcher_factory(const fs::path& temporary_save_path) -> Matcher<E> {
  if constexpr(std::is_same_v<E, DiskExtractor>) {
    return Matcher<E>(temporary_save_path, DISK_LIGHTGLUE_WEIGHT);
  } else if constexpr(std::is_same_v<E, SuperPointExtractor>) {
    return Matcher<E>(temporary_save_path, SUPERPOINT_LIGHTGLUE_WEIGHT);
  } else {
    static_assert(false, "Unknown extractor type");
  }
}

} // namespace Ortho
#endif
