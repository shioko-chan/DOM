#ifndef ORTHO_FEATURE_EXTRACTOR_HPP
#define ORTHO_FEATURE_EXTRACTOR_HPP

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "ort.hpp"
#include "static.h"
#include "utility.hpp"

namespace fs = std::filesystem;

namespace Ortho {
class Extractor {
private:

  static constexpr float superpoint_threshold       = 0.05f;
  static constexpr int   superpoint_keypoint_maxcnt = 1024;

  InferEnv superpoint;
  fs::path temporary_save_path;

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
      keypoints.resize(keypoints_size);
      descriptors.resize(descriptors_size);
      ifs.read(reinterpret_cast<char*>(keypoints.data()), keypoints_size * sizeof(float));
      ifs.read(reinterpret_cast<char*>(descriptors.data()), descriptors_size * sizeof(float));
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
      keypoints_size   = keypoints.size();
      descriptors_size = descriptors.size();
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

public:

  Extractor(fs::path temporary_save_path) :
      temporary_save_path(temporary_save_path), superpoint("[superpoint]", SUPERPOINT_WEIGHT) {}

  auto infer_keypoints_and_descriptors(ImgData& img_data) {
    fs::path path = temporary_save_path / (img_data.img.get_img_stem().string() + ".desc");

    auto elem = cache.get(path);
    if(elem.has_value()) {
      auto&& [kp_desc, lock] = elem.value();
      std::vector<float> keypoints(kp_desc.keypoints), descriptors(kp_desc.descriptors);
      lock.unlock();
      return std::make_pair(keypoints, descriptors);
    }

    std::vector<float> keypoints, descriptors;

    auto [img, lock_img] = img_data.img.rotate();
    cv::Mat img_reshaped = img.clone();
    decimate_keep_aspect_ratio(&img_reshaped, {1024, 1024});
    lock_img.unlock();

    cv::Mat img_processed;
    cv::cvtColor(img_reshaped, img_processed, cv::COLOR_BGR2GRAY);
    img_processed.convertTo(img_processed, CV_32FC1, 1.0f / 255.0f);
    img_reshaped.release();

    auto [mask, lock_mask] = img_data.img.rotate_mask();
    cv::Mat mask_processed = mask.clone();
    decimate_keep_aspect_ratio(&mask_processed, {1024, 1024});
    lock_mask.unlock();

    std::vector<float> img_vec(img_processed.begin<float>(), img_processed.end<float>());
    const int64_t      h = img_processed.rows, w = img_processed.cols;

    img_processed.release();

    superpoint.set_input("image", img_vec, {1, 1, h, w});

    auto res = superpoint.infer();
    img_vec.clear();

    const int      cnt    = res[superpoint.get_output_index("keypoints")].GetTensorTypeAndShapeInfo().GetShape()[1];
    const int64_t* kps    = res[superpoint.get_output_index("keypoints")].GetTensorData<int64_t>();
    const float *  scores = res[superpoint.get_output_index("scores")].GetTensorData<float>(),
                *descs    = res[superpoint.get_output_index("descriptors")].GetTensorData<float>();

    DEBUG("Image {} has {} keypoints detected!", img_data.img.get_img_name().string(), cnt);
    auto v = std::views::iota(0, cnt) | std::views::filter([&scores, &mask_processed, &kps](const auto& idx) {
               return scores[idx] >= superpoint_threshold
                      && mask_processed.at<unsigned char>(kps[idx * 2 + 1], kps[idx * 2]) != 0;
             });
    std::vector<size_t> indices(v.begin(), v.end());
    if(indices.size() > superpoint_keypoint_maxcnt) {
      std::nth_element(
          indices.begin(),
          indices.begin() + superpoint_keypoint_maxcnt,
          indices.end(),
          [&scores](const size_t& lhs, const size_t& rhs) { return scores[lhs] > scores[rhs]; });
      indices.resize(superpoint_keypoint_maxcnt);
    }
    const float wf2 = w / 2.0f, hf2 = h / 2.0f;

    auto v1 = indices | std::views::transform([&kps, wf2, hf2](const size_t& idx) {
                return std::array{(kps[idx * 2] - wf2) / wf2, (kps[idx * 2 + 1] - hf2) / hf2};
              })
              | std::views::join | std::views::common;
    auto v2 = indices | std::views::transform([&descs](const size_t& idx) {
                return std::views::iota(0, 256)
                       | std::views::transform([idx, &descs](const size_t& j) { return descs[idx * 256 + j]; });
              })
              | std::views::join | std::views::common;

    std::vector<float> filtered_keypoints(v1.begin(), v1.end()), filtered_descriptors(v2.begin(), v2.end());
    DEBUG(
        "Image {} has {} keypoints after threshold filter.",
        img_data.img.get_img_name().string(),
        filtered_keypoints.size() / 2);
    cache.put(KeypointsAndDescriptors(path, filtered_keypoints, filtered_descriptors));
    return std::make_pair(filtered_keypoints, filtered_descriptors);
  }
};
} // namespace Ortho
#endif