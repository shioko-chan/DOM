#ifndef SUPERPOINT_LIGHTGLUE_MATCHER_HPP
#define SUPERPOINT_LIGHTGLUE_MATCHER_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
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

class Matcher {
private:

  static constexpr float   lightglue_threshold  = 0.2f;
  static constexpr int     inlier_cnt_threshold = 150;
  InferEnv                 lightglue;
  fs::path                 temporary_save_path;
  Extractor                extractor;
  std::vector<cv::Point2f> keypoints_lhs, keypoints_rhs;
  std::vector<float>       kpts0, kpts1, desc0, desc1;

  std::pair<int, int> get_img_size(const ImgData& img_data) {
    auto [img, lock] = img_data.img.rotate();
    return {img.cols, img.rows};
  }

  auto to_pixel(const int w, const int h) {
    const float w2 = w / 2.0f, h2 = h / 2.0f;
    return std::views::transform(
        [w2, h2](auto chunk) { return cv::Point2f(*chunk.begin() * w2 + w2, *(chunk.begin() + 1) * h2 + h2); });
  }

  void set_img(
      ImgData&                  img_data,
      std::vector<cv::Point2f>& keypoints_,
      std::vector<float>&       kpts_,
      std::vector<float>&       desc_,
      const std::string&        kpts_name,
      const std::string&        desc_name) {
    auto [kpts, desc] = extractor.infer_keypoints_and_descriptors(img_data);
    const int64_t len = kpts.size() / 2;
    auto [w, h]       = get_img_size(img_data);
    auto v            = kpts | std::views::chunk(2) | to_pixel(w, h);
    keypoints_.assign(v.begin(), v.end());
    kpts_ = std::move(kpts);
    desc_ = std::move(desc);
    lightglue.set_input(kpts_name, kpts_, {1, len, 2});
    lightglue.set_input(desc_name, desc_, {1, len, 256});
  }

  void set_lhs_img(ImgData& img_data) { set_img(img_data, keypoints_lhs, kpts0, desc0, "kpts0", "desc0"); }

  void set_rhs_img(ImgData& img_data) { set_img(img_data, keypoints_rhs, kpts1, desc1, "kpts1", "desc1"); }

  auto filter_matches_by_score(const int64_t* matches, const float* scores, const int cnt) {
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

    // auto v = std::views::iota(0, cnt) | std::views::filter([&scores](const auto& idx) { return scores[idx] >=
    // lightglue_threshold; })|
    //          std::views::transform([&matches](const auto& idx) { return std::make_pair(matches[idx * 2], matches[idx
    //          * 2 + 1]); });
    // return std::vector<std::pair<int, int>>(v.begin(), v.end());
  }

  auto infer_and_filter_by_score() {
    std::vector<Ort::Value> res = lightglue.infer();

    const int      cnt_     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_ = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float*   scores_  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();

    return filter_matches_by_score(matches_, scores_, cnt_);
  }

public:

  Matcher(fs::path temporary_save_path) :
      temporary_save_path(temporary_save_path), lightglue("[lightglue]", LIGHTGLUE_WEIGHT),
      extractor(temporary_save_path) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(pairs.size());

    auto batches = pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; });

    for(auto&& batch : batches) {
      int      batch_cnt = 0;
      ImgData& lhs_img   = imgs_data[batch.front().first];
      set_lhs_img(lhs_img);

      for(auto&& pair : batch) {
        batch_cnt += 1;

        ImgData& rhs_img = imgs_data[pair.second];
        set_rhs_img(rhs_img);

        auto matches = infer_and_filter_by_score();

        DEBUG(
            "Image {} and image {} have {} matches after threshold filter!",
            lhs_img.img.get_img_name().string(),
            rhs_img.img.get_img_name().string(),
            matches.size());

        if(matches.size() < 4) {
          WARN(
              "Image {} and image {}. Not enough matches for RANSAC. At least 4 matches are needed, while only {} matches are found",
              lhs_img.img.get_img_name().string(),
              rhs_img.img.get_img_name().string(),
              matches.size());
          continue;
        }

        cv::Mat img1, img2;
        auto [img1_, lock_img1] = lhs_img.img.rotate();
        auto [img2_, lock_img2] = rhs_img.img.rotate();
        img1_.copyTo(img1);
        img2_.copyTo(img2);
        lock_img1.unlock();
        lock_img2.unlock();

        std::vector<cv::Point2f> points0, points1;
        points0.reserve(matches.size());
        points1.reserve(matches.size());
        for(auto&& [lhs, rhs] : matches) {
          points0.push_back(keypoints_lhs[lhs]);
          points1.push_back(keypoints_rhs[rhs]);
        }

        cv::Mat ransac_filter;
        cv::Mat M = cv::estimateAffinePartial2D(points1, points0, ransac_filter, cv::RANSAC, 3.0);

        std::vector<cv::DMatch> inlier_matches;
        for(size_t i = 0; i < points0.size(); i++) {
          if(ransac_filter.at<unsigned char>(i) != 0) {
            inlier_matches.emplace_back(i, i, 0.0);
          }
        }
        DEBUG(
            "Image {} and image {} have {} matches after RANSAC!",
            lhs_img.img.get_img_name().string(),
            rhs_img.img.get_img_name().string(),
            inlier_matches.size());

        if(inlier_matches.size() < inlier_cnt_threshold) {
          WARN(
              "Image {} and image {}. Not enough inlier matches. Threshold is {} matches, while only {} matches are found",
              lhs_img.img.get_img_name().string(),
              rhs_img.img.get_img_name().string(),
              inlier_cnt_threshold,
              inlier_matches.size());
          continue;
        }

        {
          std::vector<cv::Point2f> corners =
              {cv::Point2f(0, 0), cv::Point2f(img2.cols, 0), cv::Point2f(img2.cols, img2.rows), cv::Point2f(0, img2.rows)};
          std::vector<cv::Point2f> corners1 =
              {cv::Point2f(0, 0), cv::Point2f(img1.cols, 0), cv::Point2f(img1.cols, img1.rows), cv::Point2f(0, img1.rows)};
          cv::transform(corners, corners, M);
          corners.insert(corners.end(), corners1.begin(), corners1.end());
          float min_x = Ortho::min_x(corners);
          float min_y = Ortho::min_y(corners);
          std::for_each(corners.begin(), corners.end(), [min_x, min_y](cv::Point2f& p) {
            p.x -= min_x;
            p.y -= min_y;
          });
          int width  = std::ceil(Ortho::max_x(corners));
          int height = std::ceil(Ortho::max_y(corners));

          cv::Mat result(height, width, img1.type(), cv::Scalar(0, 0, 0));
          img1.copyTo(result(cv::Rect(corners[4].x, corners[4].y, img1.cols, img1.rows)));

          cv::Mat result1(height, width, img1.type(), cv::Scalar(0, 0, 0));

          cv::warpAffine(img2, result1, M, result.size(), cv::INTER_LINEAR, cv::BORDER_TRANSPARENT);
          cv::Mat mask = (result1 == 0);
          cv::Mat res;
          result.copyTo(res, mask);
          cv::Mat mask1 = (result == 0);
          result1.copyTo(res, mask1);
          cv::Mat avg;
          cv::addWeighted(result, 0.5, result1, 0.5, 0, avg);
          cv::Mat mask2;
          cv::bitwise_or(mask, mask1, mask2);
          avg.copyTo(res, ~mask2);
          if(!fs::exists(temporary_save_path / "foo")) {
            fs::create_directories(temporary_save_path / "foo");
          }

          cv::imwrite(
              temporary_save_path / "foo"
                  / (lhs_img.img.get_img_stem().string() + "_" + rhs_img.img.get_img_stem().string() + "_avg.jpg"),
              avg);
        }

        auto v0 = points0 | std::views::transform([](const cv::Point2f& p) { return cv::KeyPoint(p.x, p.y, 1); });
        auto v1 = points1 | std::views::transform([](const cv::Point2f& p) { return cv::KeyPoint(p.x, p.y, 1); });
        std::vector<cv::KeyPoint> kpts0(v0.begin(), v0.end()), kpts1(v1.begin(), v1.end());

        // cv::drawKeypoints(img1, kpts0, img1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        // cv::drawKeypoints(img2, kpts1, img2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::Mat resultImg;
        cv::drawMatches(img1, kpts0, img2, kpts1, inlier_matches, resultImg);

        cv::imwrite(
            temporary_save_path / "foo"
                / (lhs_img.img.get_img_stem().string() + "_" + rhs_img.img.get_img_stem().string() + "_matches.jpg"),
            resultImg);

        pair.valid = true;
        pair.M     = M;
      }
      progress.update(batch_cnt);
    }
  }
};
} // namespace Ortho
#endif