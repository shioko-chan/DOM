#ifndef ORTHO_FEATURE_EXTRACTOR_HPP
#define ORTHO_FEATURE_EXTRACTOR_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "models.h"
#include "ort.hpp"
#include "utility.hpp"

namespace fs = std::filesystem;

namespace Ortho {

template <size_t N>
struct Feature {
public:

  static constexpr size_t descriptor_size = N;

  static constexpr size_t size = sizeof(float) * 2 + sizeof(int) * 2 + sizeof(float) * descriptor_size;

  float                              x, y;
  int                                pix_x, pix_y;
  std::array<float, descriptor_size> desc;

  friend std::ofstream& operator<<(std::ofstream& ofs, const Feature& f) {
    ofs.write(reinterpret_cast<const char*>(&f.x), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&f.y), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&f.pix_x), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&f.pix_y), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(f.desc.data()), descriptor_size * sizeof(float));
    return ofs;
  }

  friend std::ifstream& operator>>(std::ifstream& ifs, Feature& f) {
    ifs.read(reinterpret_cast<char*>(&f.x), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&f.y), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&f.pix_x), sizeof(int));
    ifs.read(reinterpret_cast<char*>(&f.pix_y), sizeof(int));
    ifs.read(reinterpret_cast<char*>(f.desc.data()), descriptor_size * sizeof(float));
    return ifs;
  }
};

template <typename F>
  requires std::same_as<F, Feature<F::descriptor_size>>
class Extractor {
public:

  using Feature  = F;
  using Features = std::vector<Feature>;

  static constexpr size_t descriptor_size = Feature::descriptor_size;

private:

  static inline cv::Size resolution{1024, 1024};

  InferEnv env;
  fs::path temporary_save_path;

  struct Features_ : public ManagementUnit<fs::path> {
  private:

    size_t len;

    key_type path;

  public:

    Features features;

    Features_(const key_type& path, const Features& features) : ManagementUnit(true), path(path), features(features) {}

    void swap_in() override {
      if(!features.empty()) {
        throw std::runtime_error("Error: Features is already in memory");
      }
      if(!fs::exists(path)) {
        throw std::runtime_error("Error: " + path.string() + " does not exist");
      }
      std::ifstream ifs(path.string(), std::ios::binary);
      if(!ifs.is_open()) {
        throw std::runtime_error("Error: " + path.string() + " could not be opened");
      }
      features.resize(len);
      for(auto&& f : features) {
        ifs >> f;
      }
      if(ifs.fail()) {
        throw std::runtime_error("Error: " + path.string() + " could not be read");
      }
      ifs.close();
    }

    void swap_out() override {
      if(features.empty()) {
        throw std::runtime_error("Error: Features is already on disk");
      }
      if(!fs::exists(path)) {
        throw std::runtime_error("Error: " + path.string() + " does not exist");
      }
      std::ofstream ofs(path.string(), std::ios::binary | std::ios::trunc);
      if(!ofs.is_open()) {
        throw std::runtime_error("Error: " + path.string() + " could not be opened");
      }
      len = features.size();
      for(auto&& f : features) {
        ofs << f;
      }
      if(ofs.fail()) {
        throw std::runtime_error("Error: " + path.string() + " could not be written");
      }
      ofs.close();
      Features().swap(features);
    }

    const inline key_type& get_key() const override { return path; }

    inline size_t size() const override { return features.size() * Feature::size; }
  };

  static inline LRU<Features_> cache{4ul * (1ul << 30)};

protected:

  Extractor(const fs::path& temporary_save_path, const std::string& name, const std::string& model_path) :
      temporary_save_path(temporary_save_path), env(std::format("[{}]", name), model_path) {}

  inline void reshape(cv::Mat* img) { decimate_keep_aspect_ratio(img, resolution); }

  virtual void preprocess(cv::Mat* img) const = 0;

  virtual inline int64_t get_channels() const = 0;

  virtual inline float get_threshold() const = 0;

  virtual inline int64_t get_keypoint_maxcnt() const = 0;

public:

  Features get_features(ImgData& img_data) {
    fs::path path = temporary_save_path / (img_data.get_img_stem().string() + ".desc");
    auto     elem = cache.get(path);
    if(elem.has_value()) {
      auto&& [features_, lock] = elem.value();
      Features features(features_.features);
      lock.unlock();
      return features;
    }
    auto [img, lock_img]  = img_data.get_rotate_rectified();
    cv::Mat img_processed = img.clone();
    lock_img.unlock();
    reshape(&img_processed);
    preprocess(&img_processed);
    auto [mask, lock_mask] = img_data.get_rotate_rectified_mask();
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
    DEBUG("Image {} has {} keypoints detected!", img_data.get_img_name().string(), cnt);
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
    auto        u = indices | std::views::transform([kps, descs, wf2, hf2](const size_t& idx) {
               std::array<float, descriptor_size> descriptor;
               std::copy_n(descs + idx * descriptor_size, descriptor_size, descriptor.begin());
               return Feature{
                          .x     = (kps[idx * 2] - wf2) / wf2,
                          .y     = (kps[idx * 2 + 1] - hf2) / hf2,
                          .pix_x = static_cast<int>(kps[idx * 2]),
                          .pix_y = static_cast<int>(kps[idx * 2 + 1]),
                          .desc  = std::move(descriptor)};
             });
    Features    filtered_features(u.begin(), u.end());
    DEBUG(
        "Image {} has {} keypoints after threshold filter.",
        img_data.get_img_name().string(),
        filtered_features.size() / 2);
    cache.put(Features_(path, filtered_features));
    return filtered_features;
  }
};

class SuperPointExtractor : public Extractor<Feature<256>> {
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

  SuperPointExtractor(const fs::path& temporary_save_path) :
      Extractor(temporary_save_path, "superpoint", SUPERPOINT_WEIGHT) {}
};

class DiskExtractor : public Extractor<Feature<128>> {
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

  DiskExtractor(const fs::path& temporary_save_path) : Extractor(temporary_save_path, "disk", DISK_WEIGHT) {}
};

} // namespace Ortho
#endif