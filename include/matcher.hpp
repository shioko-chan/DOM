#ifndef SUPERPOINT_LIGHTGLUE_MATCHER_HPP
#define SUPERPOINT_LIGHTGLUE_MATCHER_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "extractor.hpp"
#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "models.h"
#include "ort.hpp"
#include "progress.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
class Matcher {
private:

  using Feature = typename E::Feature;
  using Features = typename E::Features;

  InferEnv           lightglue;
  fs::path           temporary_save_path;
  E                  extractor;
  std::vector<float> kpts0, kpts1, desc0, desc1;

  bool set_input(
      const Features& features,
      const std::string& kpts_name,
      std::vector<float>* kpts,
      const std::string& desc_name,
      std::vector<float>* desc) {
    if (features.empty()) {
      return false;
    }
    auto v = features
      | std::views::transform([](const auto& feature) { return std::array<float, 2>{feature.x, feature.y}; })
      | std::views::join | std::views::common;
    kpts->assign(v.begin(), v.end());
    auto w = features | std::views::transform([&features](const auto& feature) { return feature.desc; })
      | std::views::join | std::views::common;
    desc->assign(w.begin(), w.end());
    lightglue.set_input(kpts_name, *kpts, { 1, static_cast<unsigned int>(features.size()), 2 });
    lightglue.set_input(
        desc_name,
        *desc,
        { 1, static_cast<unsigned int>(features.size()), static_cast<unsigned int>(Feature::descriptor_size) });
    return true;
  }

  Matches filter_matches_by_score_precise(const int64_t* matches, const float* scores, int cnt) {
    std::unordered_map<size_t, std::pair<size_t, float>> match_score0, match_score1;
    for (size_t i = 0; i < cnt; ++i) {
      if (scores[i] >= LIGHTGLUE_THRESHOLD) {
        const size_t idx0 = matches[i * 2], idx1 = matches[i * 2 + 1];
        if (match_score0.count(idx0) == 0 || match_score0[idx0].second < scores[i]) {
          match_score0[idx0] = std::make_pair(idx1, scores[i]);
        }
      }
    }
    for (auto&& [idx0, pair] : match_score0) {
      const size_t idx1 = pair.first;
      if (match_score1.count(idx1) == 0 || match_score1[idx1].second < pair.second) {
        match_score1[idx1] = std::make_pair(idx0, pair.second);
      }
    }
    auto v = match_score1 | std::views::transform([](const auto& pair) {
      return Match { pair.second.first, pair.first, pair.second.second };
             });
    return Matches { v.begin(), v.end() };
  }

  Matches filter_matches_by_score(const int64_t* matches, const float* scores, size_t cnt) {
    auto v =
      std::views::iota(0ul, cnt)
      | std::views::filter([&scores](const auto& idx) { return scores[idx] >= LIGHTGLUE_THRESHOLD; })
      | std::views::transform([&matches, &scores](const auto& idx) {
      return Match { static_cast<size_t>(matches[idx * 2]), static_cast<size_t>(matches[idx * 2 + 1]), scores[idx] };
        });
    return Matches { v.begin(), v.end() };
  }

  auto infer() {
    OrtValues      res = lightglue.infer();
    const size_t   cnt_ = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_ = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float* scores_ = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
  }

  Matches infer_and_filter_by_score() {
    OrtValues      res = lightglue.infer();
    const size_t   cnt_ = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_ = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float* scores_ = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score(matches_, scores_, cnt_);
  }

  Matches infer_and_filter_by_score_precise() {
    OrtValues      res = lightglue.infer();
    const size_t   cnt_ = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_ = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float* scores_ = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score_precise(matches_, scores_, cnt_);
  }

  static auto feature2point(cv::Size size) {
    auto [w, h] = size;
    const float wf2 = w / 2.0f, hf2 = h / 2.0f;
    const float max2 = std::max(wf2, hf2);
    return std::views::transform(
        [wf2, hf2, max2](const auto& feature) { return Point<float>{feature.x* max2 + wf2, feature.y* max2 + hf2}; });
  }

  using cv::DMatch;
  using DMatches = std::vector<cv::DMatch>;

  static cv::Mat draw_matchlines(
      ImgData& img_lhs,
      ImgData& img_rhs,
      const Matches& matches,
      const Features& features_lhs,
      const Features& features_rhs) {
    auto          v_lhs = features_lhs | feature2point(img_lhs.get_size());
    auto          v_rhs = features_rhs | feature2point(img_rhs.get_size());
    auto          v = matches | std::views::transform([](const auto& match) {
      return DMatch(static_cast<int>(match.lhs), static_cast<int>(match.rhs), match.score)
             });
    Points<float> points_lhs { v_lhs.begin(), v_lhs.end() }, points_rhs { v_rhs.begin(), v_rhs.end() };
    DMatches      matches { v.begin(), v.end() };

    cv::Mat img0;
    {
      auto guard = img_lhs.img().get();
      guard.get().copyTo(img0);
    }
    cv::Mat img1;
    {
      auto guard = img_rhs.img().get();
      guard.get().copyTo(img1);
    }
    auto points2keypoints = [](const auto& points) {
      return points | std::views::transform([](const auto& point) { return cv::KeyPoint(point.x, point.y, 1.0f); });
      };
    auto      v_lhs = points2keypoints(points_lhs);
    auto      v_rhs = points2keypoints(points_rhs);
    KeyPoints keypoints_lhs { v_lhs.begin(), v_lhs.end() }, keypoints_rhs { v_rhs.begin(), v_rhs.end() };
    cv::Mat   res;

    cv::drawMatches(img0, keypoints_lhs, img1, keypoints_rhs, matches, res, cv::Scalar::all(-1), cv::Scalar(255, 255, 255));
    return res;
  }

public:

  Matcher(const fs::path& temporary_save_path, const std::string& weight) :
    temporary_save_path(temporary_save_path), lightglue("[lightglue]", weight), extractor(temporary_save_path) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(pairs.size());
    auto batches = pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; });
    for (auto&& batch : batches) {
      int      batch_cnt = 0;
      ImgData& lhs_img = imgs_data[batch.front().first];
      Features lhs_features = std::move(extractor.get_features(lhs_img));
      if (!set_input(lhs_features, "kpts0", &kpts0, "desc0", &desc0)) {
        LOG_INFO("Image {} has no valid feature!", lhs_img.get_img_name().string());
        continue;
      }
      auto [lhs_w, lhs_h] = lhs_img.get_size();
      for (auto&& pair : batch) {
        batch_cnt += 1;
        ImgData& rhs_img = imgs_data[pair.second];
        Features rhs_features = std::move(extractor.get_features(rhs_img));
        if (!set_input(rhs_features, "kpts1", &kpts1, "desc1", &desc1)) {
          LOG_INFO("Image {} has no valid feature!", lhs_img.get_img_name().string());
          continue;
        }
        auto [rhs_w, rhs_h] = rhs_img.get_size();
        auto matches = infer_and_filter_by_score_precise();
#ifdef ENABLE_MIDDLE_OUTPUT
        cv::imwrite(
            temporary_save_path
                / std::format("{}_{}_matches.jpg", lhs_img.get_img_stem().string(), rhs_img.get_img_stem().string()),
            draw_matchlines(lhs_img, rhs_img, matches, lhs_features, rhs_features));
#endif
        const unsigned long len = matches.size();
        LOG_DEBUG(
            "Image {} and image {} have {} matches after threshold filter!",
            lhs_img.get_img_name().string(),
            rhs_img.get_img_name().string(),
            len);
        if (matches.size() < MATCH_CNT_THRESHOLD) {
          LOG_INFO(
              "Image {} and image {} have too few matches after threshold filter: {}",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              matches.size());
          continue;
        }
        auto kpnt_lhs = matches
          | std::views::transform([&lhs_features](const auto& match) { return lhs_features[match.lhs]; })
          | feature2point(lhs_img.get_size());
        auto kpnt_rhs = matches
          | std::views::transform([&rhs_features](const auto& match) { return rhs_features[match.rhs]; })
          | feature2point(rhs_img.get_size());
        auto score = matches | std::views::transform([](const auto& match) { return match.score; });
        auto idx_lhs = lhs_img.kpnts.append(kpnt_lhs);
        auto idx_rhs = rhs_img.kpnts.append(kpnt_rhs);
        auto matches_v = std::views::zip(idx_lhs, idx_rhs, score) | std::views::transform([](auto&& idx) {
          auto&& [i0, i1, score] = idx;
          return Match { i0, i1, score };
                         });
        pair.matches.assign(matches_v.begin(), matches_v.end());
        pair.valid = true;
      }
      progress.update(batch_cnt);
    }
  }
};

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
Matcher<E> matcher_factory(const fs::path& temporary_save_path) {
  if constexpr (std::is_same_v<E, DiskExtractor>) {
    return Matcher<E>(temporary_save_path, DISK_LIGHTGLUE_WEIGHT);
  } else if constexpr (std::is_same_v<E, SuperPointExtractor>) {
    return Matcher<E>(temporary_save_path, SUPERPOINT_LIGHTGLUE_WEIGHT);
  } else {
    static_assert(false, "Unknown extractor type");
  }
}

} // namespace Ortho
#endif
