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

#include "ba.hpp"
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

  using Feature   = typename E::Feature;
  using Features  = typename E::Features;
  using Matches   = std::vector<cv::DMatch>;
  using KeyPoints = std::vector<cv::KeyPoint>;

  InferEnv           lightglue;
  fs::path           temporary_save_path;
  E                  extractor;
  std::vector<float> kpts0, kpts1, desc0, desc1;
  Features           lhs_features, rhs_features;

  bool set_input(
      const Features&     features,
      const std::string&  kpts_name,
      std::vector<float>* kpts,
      const std::string&  desc_name,
      std::vector<float>* desc) {
    if(features.empty()) {
      return false;
    }
    auto v = features
             | std::views::transform([](const auto& feature) { return std::array<float, 2>{feature.x, feature.y}; })
             | std::views::join | std::views::common;
    kpts->assign(v.begin(), v.end());
    auto w = features | std::views::transform([&features](const auto& feature) { return feature.desc; })
             | std::views::join | std::views::common;
    desc->assign(w.begin(), w.end());
    lightglue.set_input(kpts_name, *kpts, {1, static_cast<unsigned int>(features.size()), 2});
    lightglue.set_input(
        desc_name,
        *desc,
        {1, static_cast<unsigned int>(features.size()), static_cast<unsigned int>(Feature::descriptor_size)});
    return true;
  }

  Matches infer_and_filter_by_score() {
    std::vector<Ort::Value> res = lightglue.infer();
    const int      cnt_         = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_     = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float*   scores_      = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score(matches_, scores_, cnt_);
  }

  Matches infer_and_filter_by_score_precise() {
    std::vector<Ort::Value> res = lightglue.infer();
    const int      cnt_         = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
    const int64_t* matches_     = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
    const float*   scores_      = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();
    return filter_matches_by_score_precise(matches_, scores_, cnt_);
  }

  Matches filter_matches_by_score_precise(const int64_t* matches, const float* scores, int cnt) {
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
             | std::views::transform([](const auto& pair) { return cv::DMatch{pair.second.first, pair.first, 0}; });
    return Matches{v.begin(), v.end()};
  }

  Matches filter_matches_by_score(const int64_t* matches, const float* scores, size_t cnt) {
    auto v = std::views::iota(0ul, cnt)
             | std::views::filter([&scores](const auto& idx) { return scores[idx] >= LIGHTGLUE_THRESHOLD; })
             | std::views::transform([&matches](const auto& idx) {
                 return cv::DMatch{static_cast<int>(matches[idx * 2]), static_cast<int>(matches[idx * 2 + 1]), 0};
               });
    return Matches{v.begin(), v.end()};
  }

  static cv::Mat draw_matchlines(
      ImgData&             img_lhs,
      ImgData&             img_rhs,
      const Matches&       matches,
      const Points<float>& points_lhs,
      const Points<float>& points_rhs) {
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
    KeyPoints keypoints_lhs{v_lhs.begin(), v_lhs.end()}, keypoints_rhs{v_rhs.begin(), v_rhs.end()};
    cv::Mat   res;

    cv::drawMatches(img0, keypoints_lhs, img1, keypoints_rhs, matches, res, cv::Scalar::all(-1), cv::Scalar(255, 255, 255));
    return res;
  }

  static auto features2points(const auto& features, cv::Size size) {
    auto [w, h]     = size;
    const float wf2 = w / 2.0f, hf2 = h / 2.0f;
    const float max2 = std::max(wf2, hf2);
    return features | std::views::transform([wf2, hf2, max2](const auto& feature) {
             return Point<float>{feature.x * max2 + wf2, feature.y * max2 + hf2};
           });
  };

  static cv::Mat draw_matchlines(
      ImgData&        img_lhs,
      ImgData&        img_rhs,
      const Matches&  matches,
      const Features& features_lhs,
      const Features& features_rhs) {
    auto v_lhs = features2points(features_lhs, img_lhs.get_size());
    auto v_rhs = features2points(features_rhs, img_rhs.get_size());
    return draw_matchlines(
        img_lhs, img_rhs, matches, Points<float>{v_lhs.begin(), v_lhs.end()}, Points<float>{v_rhs.begin(), v_rhs.end()});
  }

  static cv::Mat
  draw_matchlines(ImgData& img_lhs, ImgData& img_rhs, const Points<float>& points_lhs, const Points<float>& points_rhs) {
    if(points_lhs.size() != points_rhs.size()) {
      LOG_ERROR("Points size mismatch: {} vs {}", points_lhs.size(), points_rhs.size());
      return cv::Mat();
    }
    size_t  len = points_lhs.size();
    auto    v   = std::views::iota(0ul, len) | std::views::transform([&points_lhs, &points_rhs](const auto& idx) {
               return cv::DMatch{static_cast<int>(idx), static_cast<int>(idx), 0};
             });
    Matches matches{v.begin(), v.end()};
    return draw_matchlines(img_lhs, img_rhs, matches, points_lhs, points_rhs);
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
      if(!set_input(lhs_features, "kpts0", &kpts0, "desc0", &desc0)) {
        LOG_INFO("Image {} has no valid feature!", lhs_img.get_img_name().string());
        continue;
      }
      auto [lhs_w, lhs_h] = lhs_img.get_size();
      for(auto&& pair : batch) {
        batch_cnt += 1;
        ImgData& rhs_img = imgs_data[pair.second];
        rhs_features     = std::move(extractor.get_features(rhs_img));
        if(!set_input(rhs_features, "kpts1", &kpts1, "desc1", &desc1)) {
          LOG_INFO("Image {} has no valid feature!", lhs_img.get_img_name().string());
          continue;
        }
        auto [rhs_w, rhs_h] = rhs_img.get_size();
        auto matches        = infer_and_filter_by_score();
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
        if(matches.empty()) {
          LOG_INFO(
              "Image {} and image {} have no valid matches!",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string());
          continue;
        }
        Points<float> points0, points1;
        {
          auto p0v = features2points(
              matches | std::views::transform([this](const auto& match) { return lhs_features[match.queryIdx]; }),
              lhs_img.get_size());
          auto p1v = features2points(
              matches | std::views::transform([this](const auto& match) { return rhs_features[match.trainIdx]; }),
              rhs_img.get_size());
          points0.assign(p0v.begin(), p0v.end());
          points1.assign(p1v.begin(), p1v.end());
        }
        // {
        //   cv::Mat inlier_mask;
        //   cv::findEssentialMat(
        //       points0, points1, lhs_img.K(), lhs_img.D(), rhs_img.K(), rhs_img.D(), cv::RANSAC, 0.999, 1.0,
        //       inlier_mask);
        //   auto p0v =
        //       std::views::iota(0ul, len)
        //       | std::views::filter([&inlier_mask](const auto& idx) { return inlier_mask.at<unsigned char>(idx) != 0;
        //       }) | std::views::transform([&points0](const auto& idx) { return points0[idx]; });
        //   auto p1v =
        //       std::views::iota(0ul, len)
        //       | std::views::filter([&inlier_mask](const auto& idx) { return inlier_mask.at<unsigned char>(idx) != 0;
        //       }) | std::views::transform([&points1](const auto& idx) { return points1[idx]; });
        //   points0.assign(p0v.begin(), p0v.end());
        //   points1.assign(p1v.begin(), p1v.end());
        //   LOG_INFO(
        //       "Image {} and image {} have {} matches after essential matrix filter!",
        //       lhs_img.get_img_name().string(),
        //       rhs_img.get_img_name().string(),
        //       points0.size());
        // }

        for(const auto& [p0, p1] : std::views::zip(points0, points1)) {
          cv::Mat t  = lhs_img.t() - rhs_img.t();
          cv::Mat v1 = lhs_img.R().t() * lhs_img.K().inv() * p0;
          cv::Mat v2 = rhs_img.R().t() * rhs_img.K().inv() * p1;
          float   a = v1.at<float>(0), b = v1.at<float>(1);
          float   c = v2.at<float>(0), d = v2.at<float>(1);
          float   t1 = t.at<float>(0), t2 = t.at<float>(1), t3 = t.at<float>(2);
          float   z2   = (b * t1 - a * t2) / (a * d - b * c);
          float   z1_1 = (z2 * c + t1) / a;
          float   z1_2 = (z2 * d + t2) / b;
          LOG_INFO("z1: {}, z2+t3={}", z1_1, z2 + t3);
          if(std::abs(z1_1 - z1_2) > 1e-2) {
            LOG_ERROR("z1_1: {}, z1_2: {}", z1_1, z1_2);
            continue;
          }
          Point3<float> p0w = mat2point3(v1 * z1_1 - lhs_img.t()), p1w = mat2point3(v2 * z2 - rhs_img.t());
          std::cout << "p0w: " << p0w << std::endl;
          std::cout << "p1w: " << p1w << std::endl;
          Point<float> p0__ = mat2point(lhs_img.K() * lhs_img.R() * (p0w + lhs_img.t())),
                       p1__ = mat2point(rhs_img.K() * rhs_img.R() * (p1w + rhs_img.t()));
          Point<float> p0_  = mat2point(lhs_img.projection_matrix() * p0w),
                       p1_  = mat2point(rhs_img.projection_matrix() * p1w);
          std::cout << "p0_: " << p0_ << "p0__:" << p0__ << "p0: " << p0 << std::endl;
          std::cout << "p1_: " << p1_ << "p1__:" << p1__ << "p1: " << p1 << std::endl;
          LOG_INFO(
              "error lhs: {}, rhs: {}", std::hypot(p0_.x - p0.x, p0_.y - p0.y), std::hypot(p1_.x - p1.x, p1_.y - p1.y));
        }

        {
          cv::Mat R = rhs_img.R() * lhs_img.R().t(), t = rhs_img.t() - lhs_img.t();
          // clang-format off
          cv::Mat Ess = (cv::Mat_<float>(3, 3) <<
                        0,-t.at<float>(2),t.at<float>(1),
                        t.at<float>(2),0,-t.at<float>(0),
                        -t.at<float>(1),t.at<float>(0),0)
                        * R;
          // clang-format on
          // cv::Mat F = rhs_img.K().inv().t() * Ess * lhs_img.K().inv();

          cv::Mat match_img = draw_matchlines(lhs_img, rhs_img, points0, points1);
          cv::imwrite(
              temporary_save_path
                  / std::format("{}_{}_matches0.jpg", lhs_img.get_img_stem().string(), rhs_img.get_img_stem().string()),
              match_img);

          cv::Mat mask;
          cv::Mat F   = cv::findFundamentalMat(points0, points1, mask, cv::FM_RANSAC);
          auto    p0v = std::views::iota(0ul, len)
                     | std::views::filter([&mask](const auto& idx) { return mask.at<unsigned char>(idx) != 0; })
                     | std::views::transform([&points0](const auto& idx) { return points0[idx]; });
          auto p1v = std::views::iota(0ul, len)
                     | std::views::filter([&mask](const auto& idx) { return mask.at<unsigned char>(idx) != 0; })
                     | std::views::transform([&points1](const auto& idx) { return points1[idx]; });
          points0.assign(p0v.begin(), p0v.end());
          points1.assign(p1v.begin(), p1v.end());
          cv::correctMatches(F, points0, points1, points0, points1);
          LOG_INFO(
              "Image {} and image {} have {} matches after fundamental matrix filter!",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string(),
              points0.size());
          match_img = draw_matchlines(lhs_img, rhs_img, points0, points1);
          cv::imwrite(
              temporary_save_path
                  / std::format("{}_{}_matches1.jpg", lhs_img.get_img_stem().string(), rhs_img.get_img_stem().string()),
              match_img);
        }

        cv::Mat points3h;
        try {
          cv::triangulatePoints(lhs_img.projection_matrix(), rhs_img.projection_matrix(), points0, points1, points3h);
        } catch(const std::exception& e) {
          LOG_ERROR("Error in triangulation: {}", e.what());
          continue;
        }
        std::vector<Point3<float>> obj;
        for(int i = 0; i < points3h.cols; i++) {
          points3h.col(i) /= points3h.at<float>(3, i);
          obj.emplace_back(points3h.at<float>(0, i), points3h.at<float>(1, i), points3h.at<float>(2, i));
        }

        float error_sum = 0;
        for(const auto& [p0, p1, wp] : std::views::zip(points0, points1, obj)) {
          if(std::isnan(wp.x) || std::isnan(wp.y) || std::isnan(wp.z)) {
            continue;
          }
          cv::Mat proj_lhs        = lhs_img.projection_matrix() * wp;
          cv::Mat proj_rhs        = rhs_img.projection_matrix() * wp;
          float   projected_lhs_x = proj_lhs.at<float>(0) / proj_lhs.at<float>(2);
          float   projected_lhs_y = proj_lhs.at<float>(1) / proj_lhs.at<float>(2);
          float   projected_rhs_x = proj_rhs.at<float>(0) / proj_rhs.at<float>(2);
          float   projected_rhs_y = proj_rhs.at<float>(1) / proj_rhs.at<float>(2);
          float   e1              = std::hypot(projected_lhs_x - p0.x, projected_lhs_y - p0.y);
          float   e2              = std::hypot(projected_rhs_x - p1.x, projected_rhs_y - p1.y);
          LOG_DEBUG(" e1 : {}, e2 : {} ", e1, e2);
          error_sum += std::pow(projected_lhs_x - p0.x, 2) + std::pow(projected_lhs_y - p0.y, 2)
                       + std::pow(projected_rhs_x - p1.x, 2) + std::pow(projected_rhs_y - p1.y, 2);
        }
        LOG_INFO("BEFORE {}", error_sum);
        std::cout << lhs_img.projection_matrix() << std::endl;
        error_sum = 0;
        auto [R_lhs, R_rhs, t_lhs, t_rhs, K_lhs, K_rhs, world_pnts] = Ortho::ba(points0, points1, lhs_img, rhs_img, obj);
        cv::Mat projection_lhs = get_projection_matrix(R_lhs, t_lhs, K_lhs),
                projection_rhs = get_projection_matrix(R_rhs, t_rhs, K_rhs);
        Points<float> points0_f, points1_f;
        //
        Points<float> src, dst;
        //

        cv::Mat img;
        {
          auto guard = lhs_img.img().get();
          guard.get().copyTo(img);
        }
        for(const auto& [p0, p1, wp] : std::views::zip(points0, points1, world_pnts)) {
          if(std::isnan(wp.x) || std::isnan(wp.y) || std::isnan(wp.z)) {
            continue;
          }
          if(std::isnan(wp.x) || std::isnan(wp.y) || std::isnan(wp.z)) {
            continue;
          }
          cv::Mat proj_lhs        = projection_lhs * wp;
          cv::Mat proj_rhs        = projection_rhs * wp;
          float   projected_lhs_x = proj_lhs.at<float>(0) / proj_lhs.at<float>(2);
          float   projected_lhs_y = proj_lhs.at<float>(1) / proj_lhs.at<float>(2);
          float   projected_rhs_x = proj_rhs.at<float>(0) / proj_rhs.at<float>(2);
          float   projected_rhs_y = proj_rhs.at<float>(1) / proj_rhs.at<float>(2);
          error_sum += std::pow(projected_lhs_x - p0.x, 2) + std::pow(projected_lhs_y - p0.y, 2)
                       + std::pow(projected_rhs_x - p1.x, 2) + std::pow(projected_rhs_y - p1.y, 2);
          if(std::hypot(projected_lhs_x - p0.x, projected_lhs_y - p0.y) > 10) {
            LOG_DEBUG(
                "Point {} is too far from the original point!",
                std::hypot(projected_lhs_x - p0.x, projected_lhs_y - p0.y));
            // continue;
          }
          src.push_back(p0);
          dst.emplace_back(wp.x, wp.y);
          cv::putText(
              img,
              std::format("({:.2f}, {:.2f}, {:.2f})", wp.x, wp.y, wp.z),
              cv::Point(p0.x + 3, p0.y + 3),
              cv::FONT_HERSHEY_SIMPLEX,
              0.5,
              cv::Scalar(0, 255, 0),
              1);
          cv::circle(img, cv::Point(p0.x, p0.y), 3, cv::Scalar(0, 255, 0), -1);
        }
        cv::Rect rect = cv::boundingRect(dst);
        std::for_each(dst.begin(), dst.end(), [rect](auto&& p) {
          p.x -= rect.x;
          p.y -= rect.y;
        });
        LOG_INFO("AFTER {}", error_sum);
        // LOG_INFO("{} matches!", points0_f.size());
        LOG_INFO("{} matches!", src.size());
        if(src.empty() || dst.empty()) {
          LOG_INFO(
              "Image {} and image {} have no valid matches after triangulation!",
              lhs_img.get_img_name().string(),
              rhs_img.get_img_name().string());
          continue;
        }
        // cv::Mat dst_img;
        // try {
        //   cv::Mat M = cv::findHomography(src, dst, cv::RANSAC);
        //   cv::warpPerspective(img, dst_img, M, rect.size(), cv::INTER_LINEAR);
        // } catch(const std::exception& e) {
        //   LOG_ERROR("Error in ortho-rectification: {}", e.what());
        //   return;
        // }
        cv::imwrite(
            temporary_save_path
                / std::format("{}_mark{}", lhs_img.get_img_stem().string(), lhs_img.get_img_extension().string()),
            img);
        // cv::imwrite(
        //     temporary_save_path
        //         / std::format("{}_ortho{}", lhs_img.get_img_stem().string(), lhs_img.get_img_extension().string()),
        //     dst_img);
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