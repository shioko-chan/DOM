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

#include "extractor.hpp"
#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "ort.hpp"
#include "progress.hpp"
#include "static.h"
#include "utility.hpp"

namespace Ortho {

template <typename E>
  requires std::derived_from<E, Extractor<typename E::Feature>>
class Matcher {
private:

  using Feature  = typename E::Feature;
  using Features = typename E::Features;

  static constexpr float lightglue_threshold  = 0.2f;
  static constexpr int   inlier_cnt_threshold = 25;
  InferEnv               lightglue;
  fs::path               temporary_save_path;
  E                      extractor;
  std::vector<float>     kpts0, kpts1, desc0, desc1;
  Features               lhs_features, rhs_features;
  std::vector<size_t>    lhs_idx, rhs_idx;

  void generate_indexs_with_roi_filter(ImgData& lhs_img, ImgData& rhs_img) {
    auto inter = intersection(lhs_img.get_spans(), rhs_img.get_spans());
    if(inter.empty()) {
      WARN("Image {} and image {} have no intersection!", lhs_img.get_img_name().string(), rhs_img.get_img_name().string());
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
      if(scores[i] >= lightglue_threshold) {
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
             | std::views::filter([&scores](const auto& idx) { return scores[idx] >= lightglue_threshold; })
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
          WARN(
              "Image {} and image {}. Not enough matches for RANSAC. At least 4 matches are needed, while only {} matches are found",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              matches.size());
          continue;
        }
        cv::Mat img1;
        auto [img1_, lock_img1] = lhs_img.get_rotate_rectified();
        img1_.copyTo(img1);
        lock_img1.unlock();
        cv::Mat img2;
        auto [img2_, lock_img2] = rhs_img.get_rotate_rectified();
        img2_.copyTo(img2);
        lock_img2.unlock();
        auto              lhs_pnts = lhs_img.get_spans();
        auto              rhs_pnts = rhs_img.get_spans();
        std::stringstream ss1;
        ss1 << "lhs_pnts: " << lhs_pnts << "rhs_pnts: " << rhs_pnts;
        INFO("{}", ss1.str());
        auto              inter = intersection(lhs_pnts, rhs_pnts);
        std::stringstream ss2;
        ss2 << "inter: " << inter;
        INFO("{}", ss2.str());
        std::stringstream ss;
        ss << "lhs_ world2img: " << lhs_img.world2img(lhs_pnts) << "rhs_ world2img: " << rhs_img.world2img(rhs_pnts);
        INFO("{}", ss.str());
        std::vector<std::vector<cv::Point>> lhs_img_pnts(1), rhs_img_pnts(1);
        for(auto&& p : lhs_img.world2img(inter)) {
          lhs_img_pnts[0].emplace_back(cv::Point{cv::Point(p.x, p.y)});
        }
        for(auto&& p : rhs_img.world2img(inter)) {
          rhs_img_pnts[0].emplace_back(cv::Point{cv::Point(p.x, p.y)});
        }
        std::stringstream ss11;
        ss11 << "lhs_ world2img: " << lhs_img_pnts.size() << " " << lhs_img_pnts[0];
        ss11 << "rhs_ world2img: " << rhs_img_pnts.size() << " " << rhs_img_pnts[0];
        INFO("{}", ss11.str());
        cv::drawContours(img1, lhs_img_pnts, -1, cv::Scalar(0, 255, 0), 2);
        cv::drawContours(img2, rhs_img_pnts, -1, cv::Scalar(0, 255, 0), 2);
        // cv::imshow("lhs_img", img1);
        // cv::imshow("rhs_img", img2);
        // cv::waitKey(0);

        cv::Mat img1_mask;
        auto [img1_mask_, lock_img1_mask] = lhs_img.get_rotate_rectified_mask();
        img1_mask_.copyTo(img1_mask);
        lock_img1_mask.unlock();
        cv::Mat img2_mask;
        auto [img2_mask_, lock_img2_mask] = rhs_img.get_rotate_rectified_mask();
        img2_mask_.copyTo(img2_mask);
        lock_img2_mask.unlock();
        Points<float> points0, points1;
        points0.reserve(matches.size());
        points1.reserve(matches.size());
        for(auto&& [lhs, rhs] : matches) {
          points0.emplace_back(lhs_features[lhs_idx[lhs]].pix_x, lhs_features[lhs_idx[lhs]].pix_y);
          points1.emplace_back(rhs_features[rhs_idx[rhs]].pix_x, rhs_features[rhs_idx[rhs]].pix_y);
        }
        cv::Mat ransac_filter;
        cv::Mat M = cv::estimateAffinePartial2D(points1, points0, ransac_filter, cv::RANSAC, 0.5, 200000ul, 0.99, 100ul);
        std::vector<cv::DMatch> inlier_matches;
        for(size_t i = 0; i < points0.size(); i++) {
          if(ransac_filter.at<unsigned char>(i) != 0) {
            inlier_matches.emplace_back(i, i, 0.0);
          }
        }
        DEBUG(
            "Image {} and image {} have {} matches after RANSAC!",
            lhs_img.get_img_name().string(),
            rhs_img.get_img_name().string(),
            inlier_matches.size());
        if(inlier_matches.size() < inlier_cnt_threshold) {
          WARN(
              "Image {} and image {}. Not enough inlier matches. Threshold is {} matches, while only {} matches are found",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              inlier_cnt_threshold,
              inlier_matches.size());
          continue;
        }
        {
          Points<float> corners =
              {Point<float>(0, 0),
               Point<float>(img2.cols - 1, 0),
               Point<float>(img2.cols - 1, img2.rows - 1),
               Point<float>(0, img2.rows - 1)};
          Points<float> corners1 =
              {Point<float>(0, 0),
               Point<float>(img1.cols - 1, 0),
               Point<float>(img1.cols - 1, img1.rows - 1),
               Point<float>(0, img1.rows - 1)};
          cv::transform(corners, corners, M);
          corners.insert(corners.end(), corners1.begin(), corners1.end());
          auto        v = corners | std::views::transform([](const Point<float>& p) {
                     return Point<int>(abs_ceil(p.x), abs_ceil(p.y));
                   });
          Points<int> corners_int(v.begin(), v.end());
          cv::Rect    rect = cv::boundingRect(corners_int);
          std::for_each(corners.begin(), corners.end(), [&rect](Point<float>& p) {
            p.x -= rect.x;
            p.y -= rect.y;
          });
          cv::Mat result1(rect.height, rect.width, img1.type(), cv::Scalar(0, 0, 0));
          img1.copyTo(result1(cv::Rect(corners[4].x, corners[4].y, img1.cols, img1.rows)));
          cv::Mat       result2(rect.height, rect.width, img1.type(), cv::Scalar(0, 0, 0));
          Points<float> from =
              {Point<float>(0, 0),
               Point<float>(img2.cols - 1, 0),
               Point<float>(img2.cols - 1, img2.rows - 1),
               Point<float>(0, img2.rows - 1)};
          Points<float> to(corners.begin(), corners.begin() + 4);
          cv::Mat       M = cv::estimateAffinePartial2D(from, to);
          cv::warpAffine(img2, result2, M, result1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
          cv::Mat avg;
          cv::addWeighted(result1, 0.5, result2, 0.5, 0, avg);
          cv::Mat mask1(rect.height, rect.width, img1_mask.type(), cv::Scalar(0));
          img1_mask.copyTo(mask1(cv::Rect(corners[4].x, corners[4].y, img1_mask.cols, img1_mask.rows)));
          cv::Mat mask2(rect.height, rect.width, img1_mask.type(), cv::Scalar(0));
          cv::warpAffine(img2_mask, mask2, M, result1.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
          cv::Mat res;
          result1.copyTo(res, mask1);
          result2.copyTo(res, mask2);
          avg.copyTo(res, mask1 & mask2);
          if(!fs::exists(temporary_save_path / "foo")) {
            fs::create_directories(temporary_save_path / "foo");
          }
          cv::imwrite(
              temporary_save_path / "foo"
                  / (lhs_img.get_img_stem().string() + "_" + rhs_img.get_img_stem().string() + "_avg.jpg"),
              res);
        }
        auto v0 = points0 | std::views::transform([](const Point<float>& p) { return cv::KeyPoint(p.x, p.y, 1); });
        auto v1 = points1 | std::views::transform([](const Point<float>& p) { return cv::KeyPoint(p.x, p.y, 1); });
        std::vector<cv::KeyPoint> kpts0(v0.begin(), v0.end()), kpts1(v1.begin(), v1.end());
        // cv::drawKeypoints(img1, kpts0, img1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // cv::drawKeypoints(img2, kpts1, img2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::Mat resultImg;
        cv::drawMatches(img1, kpts0, img2, kpts1, inlier_matches, resultImg);
        cv::imwrite(
            temporary_save_path / "foo"
                / (lhs_img.get_img_stem().string() + "_" + rhs_img.get_img_stem().string() + "_matches.jpg"),
            resultImg);
        pair.valid = true;
        pair.M     = M;
      }
      progress.update(batch_cnt);
    }
  }
};

template <typename E>
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