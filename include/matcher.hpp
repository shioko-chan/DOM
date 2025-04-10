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

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "extractor.hpp"
#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "models.h"
#include "ort.hpp"
#include "progress.hpp"
#include "utility.hpp"

namespace Ortho {

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
class Matcher {
private:

  using Feature  = typename E::Feature;
  using Features = typename E::Features;

  InferEnv            lightglue;
  fs::path            temporary_save_path;
  E                   extractor;
  std::vector<float>  kpts0, kpts1, desc0, desc1;
  Features            lhs_features, rhs_features;
  std::vector<size_t> lhs_idx, rhs_idx;

  void generate_indexs_with_roi_filter(ImgData& lhs_img, ImgData& rhs_img) {
    auto inter = intersection(lhs_img.get_spans(), rhs_img.get_spans());
    if(inter.empty()) {
      INFO("Image {} and image {} have no intersection!", lhs_img.get_img_name().string(), rhs_img.get_img_name().string());
      return;
    }
    auto generate_idx = [](const Points<float>& area, const Features& features, std::vector<size_t>* idx_list) {
      auto v = std::views::iota(0ul, features.size()) | std::views::filter([&area, &features](const size_t& idx) {
                 return cv::pointPolygonTest(area, cv::Point2f(features[idx].pix_x, features[idx].pix_y), false) >= 0;
               });
      idx_list->assign(v.begin(), v.end());
    };
    generate_idx(lhs_img.world2img(inter), lhs_features, &lhs_idx);
    generate_idx(rhs_img.world2img(inter), rhs_features, &rhs_idx);
  }

  void set_input_(
      const Features&            features,
      const std::vector<size_t>& idx,
      const std::string&         kpts_name,
      std::vector<float>*        kpts,
      const std::string&         desc_name,
      std::vector<float>*        desc) {
    auto v = idx | std::views::transform([&features](const auto& idx) {
               return std::array<float, 2>{features[idx].x, features[idx].y};
             })
             | std::views::join | std::views::common;
    kpts->assign(v.begin(), v.end());
    auto w = idx | std::views::transform([&features](const auto& idx) { return features[idx].desc; }) | std::views::join
             | std::views::common;
    desc->assign(w.begin(), w.end());
    lightglue.set_input(kpts_name, *kpts, {1, static_cast<unsigned int>(idx.size()), 2});
    lightglue.set_input(
        desc_name,
        *desc,
        {1, static_cast<unsigned int>(idx.size()), static_cast<unsigned int>(Feature::descriptor_size)});
  }

  void set_input() {
    set_input_(lhs_features, lhs_idx, "kpts0", &kpts0, "desc0", &desc0);
    set_input_(rhs_features, rhs_idx, "kpts1", &kpts1, "desc1", &desc1);
  }

  auto infer_and_filter_by_score() {
    std::vector<Ort::Value> res = lightglue.infer();
    const int      cnt_         = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_     = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float*   scores_      = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score(matches_, scores_, cnt_);
  }

  auto filter_matches_by_score_precise(const int64_t* matches, const float* scores, const int cnt) {
    std::unordered_map<int, std::pair<int, float>> match_score0, match_score1;
    for(int i = 0; i < cnt; ++i) {
      if(scores[i] >= LIGHTGLUE_THRESHOLD) {
        const int idx0 = matches[i * 2], idx1 = matches[i * 2 + 1];
        if(match_score0.count(idx0) == 0 || match_score0[idx0].second < scores[i]) {
          match_score0[idx0] = std::make_pair(idx1, scores[i]);
        }
      }
    }
    for(auto&& [idx0, pair] : match_score0) {
      const int idx1 = pair.first;
      if(match_score1.count(idx1) == 0 || match_score1[idx1].second < pair.second) {
        match_score1[idx1] = std::make_pair(idx0, pair.second);
      }
    }
    auto v = match_score1
             | std::views::transform([](const auto& pair) { return std::make_pair(pair.second.first, pair.first); });
    return std::vector<std::pair<int, int>>(v.begin(), v.end());
  }

  auto filter_matches_by_score(const int64_t* matches, const float* scores, const int cnt) {
    auto v = std::views::iota(0, cnt)
             | std::views::filter([&scores](const auto& idx) { return scores[idx] >= LIGHTGLUE_THRESHOLD; })
             | std::views::transform(
                 [&matches](const auto& idx) { return std::make_pair(matches[idx * 2], matches[idx * 2 + 1]); });
    return std::vector<std::pair<int, int>>(v.begin(), v.end());
  }

public:

  Matcher(const fs::path& temporary_save_path, const std::string& weight) :
      temporary_save_path(temporary_save_path), lightglue("[lightglue]", weight), extractor(temporary_save_path) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(pairs.size());
    auto batches = pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; });
    for(auto&& batch : batches) {
      int      batch_cnt = 0;
      ImgData& lhs_img   = imgs_data[batch.front().first];
      lhs_features       = std::move(extractor.get_features(lhs_img));
      for(auto&& pair : batch) {
        batch_cnt += 1;
        ImgData& rhs_img = imgs_data[pair.second];
        rhs_features     = std::move(extractor.get_features(rhs_img));
        generate_indexs_with_roi_filter(lhs_img, rhs_img);
        set_input();
        auto matches = infer_and_filter_by_score();
        DEBUG(
            "Image {} and image {} have {} matches after threshold filter!",
            lhs_img.get_img_name().string(),
            rhs_img.get_img_name().string(),
            matches.size());

        if(matches.size() < 4) {
          INFO(
              "Image {} and image {}. Not enough matches for RANSAC. At least 4 matches are needed, while only {} matches are found",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              matches.size());
          continue;
        }
        Points<float> points0, points1;
        points0.reserve(matches.size());
        points1.reserve(matches.size());
        for(auto&& [lhs, rhs] : matches) {
          points0.emplace_back(lhs_features[lhs_idx[lhs]].pix_x, lhs_features[lhs_idx[lhs]].pix_y);
          points1.emplace_back(rhs_features[rhs_idx[rhs]].pix_x, rhs_features[rhs_idx[rhs]].pix_y);
        }
        cv::Mat ransac_filter;
        cv::Mat M = cv::estimateAffinePartial2D(points0, points1, ransac_filter, cv::RANSAC, 0.5, 200000ul, 0.99, 100ul);
        M.convertTo(M, CV_32FC1);
        if(M.empty()) {
          INFO(
              "Image {} and image {}. Estimate affine transform failed.",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string());
          continue;
        }
        const int inlier_cnt = cv::countNonZero(ransac_filter);
        DEBUG(
            "Image {} and image {} have {} matches after RANSAC!",
            lhs_img.get_img_name().string(),
            rhs_img.get_img_name().string(),
            inlier_cnt);
        if(inlier_cnt < INLIER_CNT_THRESHOLD) {
          INFO(
              "Image {} and image {}. Not enough inlier matches. Threshold is {} matches, while only {} matches are found",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              INLIER_CNT_THRESHOLD,
              inlier_cnt);
          continue;
        }
        pair.M     = std::move(M);
        pair.valid = true;
      }
      progress.update(batch_cnt);
    }
  }
};

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
Matcher<E> matcher_factory(const fs::path& temporary_save_path) {
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