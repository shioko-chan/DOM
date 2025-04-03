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

  static inline cv::Size resolution{1024, 1024};

  InferEnv env;
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

protected:

  Extractor(const fs::path& temporary_save_path, const std::string& name, const std::string& model_path) :
      temporary_save_path(temporary_save_path), env(std::format("[{}]", name), model_path) {}

  inline void reshape(cv::Mat* img) { decimate_keep_aspect_ratio(img, resolution); }

  virtual void preprocess(cv::Mat* img) const = 0;

  virtual inline int64_t get_channels() const = 0;

  virtual inline float get_threshold() const = 0;

  virtual inline int64_t get_keypoint_maxcnt() const = 0;

public:

  virtual inline int64_t get_descriptor_width() const = 0;

  std::pair<std::vector<float>, std::vector<float>> infer_keypoints_and_descriptors(ImgData& img_data) {
    fs::path path = temporary_save_path / (img_data.img.get_img_stem().string() + ".desc");

    auto elem = cache.get(path);
    if(elem.has_value()) {
      auto&& [kp_desc, lock] = elem.value();
      std::vector<float> keypoints(kp_desc.keypoints), descriptors(kp_desc.descriptors);
      lock.unlock();
      return {keypoints, descriptors};
    }

    auto [img, lock_img]  = img_data.img.rotate();
    cv::Mat img_processed = img.clone();
    lock_img.unlock();
    reshape(&img_processed);
    preprocess(&img_processed);

    auto [mask, lock_mask] = img_data.img.rotate_mask();
    cv::Mat mask_processed = mask.clone();
    lock_mask.unlock();
    reshape(&mask_processed);

    std::vector<float> img_vec(img_processed.begin<float>(), img_processed.end<float>());
    const int64_t      h = mask_processed.rows, w = mask_processed.cols;
    img_processed.release();

    env.set_input("image", img_vec, {1, get_channels(), h, w});

    if(img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    auto res = env.infer();
    if(img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    img_vec.clear();

    const int      cnt    = res[env.get_output_index("keypoints")].GetTensorTypeAndShapeInfo().GetShape()[1];
    const int64_t* kps    = res[env.get_output_index("keypoints")].GetTensorData<int64_t>();
    const float *  scores = res[env.get_output_index("scores")].GetTensorData<float>(),
                *descs    = res[env.get_output_index("descriptors")].GetTensorData<float>();

    DEBUG("Image {} has {} keypoints detected!", img_data.img.get_img_name().string(), cnt);

    auto v = std::views::iota(0, cnt) | std::views::filter([this, &scores, &mask_processed, &kps](const auto& idx) {
               return scores[idx] >= get_threshold()
                      && mask_processed.at<unsigned char>(kps[idx * 2 + 1], kps[idx * 2]) != 0;
             });
    std::vector<size_t> indices(v.begin(), v.end());
    if(indices.size() > get_keypoint_maxcnt()) {
      std::nth_element(
          indices.begin(),
          indices.begin() + get_keypoint_maxcnt(),
          indices.end(),
          [&scores](const size_t& lhs, const size_t& rhs) { return scores[lhs] > scores[rhs]; });
      indices.resize(get_keypoint_maxcnt());
    }
    const float wf2 = w / 2.0f, hf2 = h / 2.0f;

    auto v1 = indices | std::views::transform([&kps, wf2, hf2](const size_t& idx) {
                return std::array{(kps[idx * 2] - wf2) / wf2, (kps[idx * 2 + 1] - hf2) / hf2};
              })
              | std::views::join | std::views::common;
    auto v2 = indices | std::views::transform([this, &descs](const size_t& idx) {
                return std::views::iota(0, get_descriptor_width())
                       | std::views::transform(
                           [this, idx, &descs](const size_t& j) { return descs[idx * get_descriptor_width() + j]; });
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

class SuperPointExtractor : public Extractor {
private:

  static constexpr float superpoint_threshold       = 0.05f;
  static constexpr int   superpoint_keypoint_maxcnt = 1024;

  void preprocess(cv::Mat* img) const override {
    cv::cvtColor(*img, *img, cv::COLOR_BGR2GRAY);
    img->convertTo(*img, CV_32FC1, 1.0f / 255.0f);
  }

  inline int64_t get_channels() const override { return 1; }

  inline float get_threshold() const override { return superpoint_threshold; }

  inline int64_t get_keypoint_maxcnt() const override { return superpoint_keypoint_maxcnt; }

public:

  inline int64_t get_descriptor_width() const override { return 256; }

  SuperPointExtractor(const fs::path& temporary_save_path) :
      Extractor(temporary_save_path, "superpoint", SUPERPOINT_WEIGHT) {}
};

class DiskExtractor : public Extractor {
private:

  static constexpr float disk_threshold       = 0.05f;
  static constexpr int   disk_keypoint_maxcnt = 1024;

  void preprocess(cv::Mat* img) const override {
    if(!img->isContinuous()) {
      *img = img->clone();
    }
    std::vector<cv::Mat> channels;
    cv::split(*img, channels);
    img->create(3, channels[0].rows * channels[0].cols, CV_32FC1);
    channels[2].reshape(1, 1).convertTo(img->row(0), CV_32FC1, 1.0f / 255.0f);
    channels[1].reshape(1, 1).convertTo(img->row(1), CV_32FC1, 1.0f / 255.0f);
    channels[0].reshape(1, 1).convertTo(img->row(2), CV_32FC1, 1.0f / 255.0f);
  }

  inline int64_t get_channels() const override { return 3; }

  inline float get_threshold() const override { return disk_threshold; }

  inline int64_t get_keypoint_maxcnt() const override { return disk_keypoint_maxcnt; }

public:

  inline int64_t get_descriptor_width() const override { return 128; }

  DiskExtractor(const fs::path& temporary_save_path) : Extractor(temporary_save_path, "disk", DISK_WEIGHT) {}
};

} // namespace Ortho
#endif