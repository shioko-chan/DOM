#ifndef SUPERPOINT_LIGHTGLUE_MATCHER_HPP
#define SUPERPOINT_LIGHTGLUE_MATCHER_HPP

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "ort.hpp"
#include "progress.hpp"
#include "static.h"
#include "utility.hpp"

namespace fs = std::filesystem;

namespace Ortho {
class Matcher {
private:

  static constexpr float superpoint_threshold = 0.2f, lightglue_threshold = 0.5f;
  static constexpr int   inlier_cnt_threshold = 300, superpoint_keypoint_threshold = 500;

  struct KeypointsAndDescriptors : public CacheElem<fs::path> {
  private:

    std::size_t keypoints_size, descriptors_size;

    key_type path;

  public:

    KeypointsAndDescriptors(
        const key_type&           path,
        const std::vector<float>& keypoints,
        const std::vector<float>& descriptors) :
        CacheElem(true), path(path), keypoints(keypoints), descriptors(descriptors) {}

    std::vector<float> keypoints, descriptors;

    void swap_in() override {
      if(!keypoints.empty() || !descriptors.empty()) {
        throw std::runtime_error("Error: Keypoint or Descriptor is already in memory");
      }
      if(!fs::exists(path)) {
        throw std::runtime_error("Error: " + path.string() + " does not exist");
      }
      std::ifstream ifs(path.string(), std::ios::binary);
      if(!ifs.is_open()) {
        throw std::runtime_error("Error: " + path.string() + " could not be opened");
      }
      ifs.read(reinterpret_cast<char*>(keypoints.data()), keypoints_size);
      ifs.read(reinterpret_cast<char*>(descriptors.data()), descriptors_size);
      if(ifs.fail()) {
        throw std::runtime_error("Error: " + path.string() + " could not be read");
      }
      ifs.close();
    }

    void swap_out() override {
      if(keypoints.empty() || descriptors.empty()) {
        throw std::runtime_error("Error: Keypoint or Descriptor is not in memory");
      }
      if(!fs::exists(path)) {
        throw std::runtime_error("Error: " + path.string() + " does not exist");
      }
      std::ofstream ofs(path.string(), std::ios::binary);
      if(!ofs.is_open()) {
        throw std::runtime_error("Error: " + path.string() + " could not be opened");
      }
      ofs.write(reinterpret_cast<const char*>(keypoints.data()), keypoints.size() * sizeof(float));
      ofs.write(reinterpret_cast<const char*>(descriptors.data()), descriptors.size() * sizeof(float));
      keypoints_size   = keypoints.size() * sizeof(float);
      descriptors_size = descriptors.size() * sizeof(float);
      if(ofs.fail()) {
        throw std::runtime_error("Error: " + path.string() + " could not be written");
      }
      ofs.close();
      keypoints.clear();
      descriptors.clear();
      std::vector<float>().swap(keypoints);
      std::vector<float>().swap(descriptors);
    }

    const inline key_type& get_key() const override { return path; }

    inline std::size_t size() const override {
      return keypoints.size() * sizeof(float) + descriptors.size() * sizeof(float);
    }
  };

  static inline LRU<KeypointsAndDescriptors> cache{4ul * (1ul << 30)};

  InferEnv superpoint, lightglue;
  fs::path temporary_save_path;

  cv::Mat preprocess(const cv::Mat& img_) {
    cv::Mat img_processed;
    cv::cvtColor(img_, img_processed, cv::COLOR_BGR2GRAY);
    img_processed.convertTo(img_processed, CV_32FC1, 1.0 / 255.0);
    return img_processed;
  }

  std::pair<std::vector<float>, std::vector<float>> infer_keypoints_and_descriptors(ImgData& img_data) {
    fs::path path = temporary_save_path / (img_data.img.get_img_stem().string() + ".desc");

    auto elem = cache.get(path);
    if(elem.has_value()) {
      auto&& [kp_desc, lock]       = elem.value();
      std::vector<float> keypoints = kp_desc.keypoints, descriptors = kp_desc.descriptors;
      lock.unlock();
      return {keypoints, descriptors};
    }

    std::vector<float> keypoints, descriptors;

    auto&& [img, lock_img] = img_data.img.rotate();
    cv::Mat img_reshaped   = img.clone();
    decimate_keep_aspect_ratio(&img_reshaped, {1024, 1024});
    lock_img.unlock();

    auto&& [mask, lock_mask] = img_data.img.rotate_mask();
    cv::Mat mask_processed   = mask.clone();
    decimate_keep_aspect_ratio(&mask_processed, {1024, 1024});
    lock_mask.unlock();

    cv::Mat img_processed = preprocess(img_reshaped);
    img_reshaped.release();

    std::vector<float> img_vec(img_processed.begin<float>(), img_processed.end<float>());
    const int64_t      h = img_processed.rows, w = img_processed.cols;
    img_processed.release();

    superpoint.set_input("image", img_vec, {1, 1, h, w});
    auto&& res = superpoint.infer();
    img_vec.clear();

    const int      cnt    = res[superpoint.get_output_index("keypoints")].GetTensorTypeAndShapeInfo().GetShape()[1];
    const int64_t* kps    = res[superpoint.get_output_index("keypoints")].GetTensorData<int64_t>();
    const float *  scores = res[superpoint.get_output_index("scores")].GetTensorData<float>(),
                *descs    = res[superpoint.get_output_index("descriptors")].GetTensorData<float>();

    std::vector<float> filtered_keypoints, filtered_descriptors;
    float              wf2 = w / 2.0f, hf2 = h / 2.0f;
    INFO("keypoints: {}", cnt);
    for(int j = 0; j < cnt; ++j) {
      if(scores[j] >= superpoint_threshold && mask_processed.at<uchar>(kps[j * 2 + 1], kps[j * 2]) != 0) {
        filtered_keypoints.insert(filtered_keypoints.end(), {(kps[j * 2] - wf2) / wf2, (kps[j * 2 + 1] - hf2) / hf2});
        filtered_descriptors.insert(filtered_descriptors.end(), descs + j * 256, descs + (j + 1) * 256);
      }
    }

    cache.put(KeypointsAndDescriptors(path, filtered_keypoints, filtered_descriptors));

    return {filtered_keypoints, filtered_descriptors};
  }

public:

  Matcher(fs::path temporary_save_path) :
      temporary_save_path(temporary_save_path), superpoint("[superpoint]", SUPERPOINT_WEIGHT),
      lightglue("[lightglue]", LIGHTGLUE_WEIGHT) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(pairs.size());

    auto batches = pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; });

    for(auto&& batch : batches) {
      int       batch_cnt = 0;
      const int i         = batch.front().first;

      auto&& [keypoints0, descriptors0] = infer_keypoints_and_descriptors(imgs_data[i]);

      if(keypoints0.size() < superpoint_keypoint_threshold) {
        WARN(
            "Image \"{}\": Not enough keypoints, threshold is {} points.",
            imgs_data[i].img.get_img_name().string(),
            superpoint_keypoint_threshold);
        continue;
      }

      lightglue.set_input("kpts0", keypoints0, {1, static_cast<int64_t>(keypoints0.size()) / 2, 2});
      lightglue.set_input("desc0", descriptors0, {1, static_cast<int64_t>(keypoints0.size()) / 2, 256});

      for(auto&& pair : batch) {
        const int j = pair.second;

        auto&& [keypoints1, descriptors1] = infer_keypoints_and_descriptors(imgs_data[i]);

        if(keypoints1.size() < superpoint_keypoint_threshold) {
          WARN(
              "Image \"{}\": Not enough keypoints, threshold is {} points.",
              imgs_data[i].img.get_img_name().string(),
              superpoint_keypoint_threshold);
          continue;
        }

        lightglue.set_input("kpts1", keypoints1, {1, static_cast<int64_t>(keypoints1.size()) / 2, 2});
        lightglue.set_input("desc1", descriptors1, {1, static_cast<int64_t>(keypoints1.size()) / 2, 256});

        std::vector<Ort::Value> res = lightglue.infer();

        const int      cnt     = res[lightglue.get_output_index("matches0")].GetTensorTypeAndShapeInfo().GetShape()[0];
        const int64_t* matches = res[lightglue.get_output_index("matches0")].GetTensorData<int64_t>();
        const float*   scores  = res[lightglue.get_output_index("mscores0")].GetTensorData<float>();

        INFO("all matches: {}", cnt);

        std::map<int, std::pair<int, float>> match_score0, match_score1;

        for(int j = 0; j < cnt; ++j) {
          if(scores[j] >= lightglue_threshold) {
            const int idx0 = matches[j * 2], idx1 = matches[j * 2 + 1];
            if(idx0 < 0 || idx1 < 0) {
              WARN("Warning: Invalid match");
              continue;
            }
            if(idx0 >= keypoints0.size() / 2 || idx1 >= keypoints1.size() / 2) {
              WARN("Warning: Index out of range");
              continue;
            }
            if(match_score0.count(idx0) == 0 || match_score0[idx0].second < scores[j]) {
              match_score0[idx0] = std::make_pair(idx1, scores[j]);
            }
          }
        }
        for(auto&& [idx0, pair] : match_score0) {
          const int idx1 = pair.first;
          if(match_score1.count(idx1) == 0 || match_score1[idx1].second < pair.second) {
            match_score1[idx1] = std::make_pair(idx0, pair.second);
          }
        }
        std::vector<cv::Point2f> points0, points1;
        std::vector<cv::DMatch>  all_matches;
        for(auto&& [idx1, pair] : match_score1) {
          const int idx0 = pair.first;
          points0.emplace_back(keypoints0[idx0 * 2], keypoints0[idx0 * 2 + 1]);
          points1.emplace_back(keypoints1[idx1 * 2], keypoints1[idx1 * 2 + 1]);
        }
        if(points0.size() < 4 || points1.size() < 4) {
          WARN("Warning: Not enough matches");
          continue;
        }

        std::vector<unsigned char> mask(points0.size());

        cv::Mat M = cv::estimateAffinePartial2D(points1, points0, mask, cv::RANSAC, 1.0);
        // cv::Mat M = cv::findHomography(points1, points0, cv::RANSAC, 1.0, mask);

        std::vector<cv::DMatch> inlier_matches;
        for(size_t i = 0; i < mask.size(); i++) {
          if(mask[i]) {
            inlier_matches.emplace_back(i, i, 0.0);
          }
        }
        if(inlier_matches.size() < inlier_cnt_threshold) {
          WARN("Warning: Not enough inlier matches");
          continue;
        }
        cv::Mat resultImg, img1, img2;
        imgs_data[i].img.rotate().first.copyTo(img1);
        imgs_data[j].img.rotate().first.copyTo(img2);

        {
          std::vector<cv::Point2f> corners =
              {cv::Point2f(0, 0), cv::Point2f(img2.cols, 0), cv::Point2f(img2.cols, img2.rows), cv::Point2f(0, img2.rows)};
          std::vector<cv::Point2f> corners1 =
              {cv::Point2f(0, 0), cv::Point2f(img1.cols, 0), cv::Point2f(img1.cols, img1.rows), cv::Point2f(0, img1.rows)};
          cv::transform(corners, corners, M);
          corners.insert(corners.end(), corners1.begin(), corners1.end());
          float min_x = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                          return a.x < b.x;
                        })->x;
          float min_y = std::min_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                          return a.y < b.y;
                        })->y;
          std::for_each(corners.begin(), corners.end(), [min_x, min_y](cv::Point2f& p) {
            p.x -= min_x;
            p.y -= min_y;
          });
          int width =
              std::ceil(std::max_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                          return a.x < b.x;
                        })->x);
          int height =
              std::ceil(std::max_element(corners.begin(), corners.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
                          return a.y < b.y;
                        })->y);
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

          cv::imwrite(temporary_save_path / "foo" / (std::to_string(i) + "_" + std::to_string(j) + "_avg.jpg"), avg);
        }
        std::vector<cv::KeyPoint> kpts1, kpts2;
        for(size_t i = 0; i < keypoints0.size() / 2; i++) {
          kpts1.emplace_back(
              cv::KeyPoint((keypoints0[i * 2] + 1) * img1.cols / 2, (keypoints0[i * 2 + 1] + 1) * img1.rows / 2, 1));
        }
        for(size_t i = 0; i < keypoints1.size() / 2; i++) {
          kpts2.emplace_back(
              cv::KeyPoint((keypoints1[i * 2] + 1) * img2.cols / 2, (keypoints1[i * 2 + 1] + 1) * img2.rows / 2, 1));
        }

        INFO("inlier matches: {}", inlier_matches.size());
        cv::drawKeypoints(img1, kpts1, img1, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawKeypoints(img2, kpts2, img2, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::drawMatches(img1, kpts1, img2, kpts2, inlier_matches, resultImg);

        cv::imwrite(temporary_save_path / "foo" / (std::to_string(i) + "_" + std::to_string(j) + ".jpg"), resultImg);

        pair.valid = true;
        pair.M     = M;

        batch_cnt += 1;
      }
      progress.update(batch_cnt);
    }
  }
};
} // namespace Ortho
#endif