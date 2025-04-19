#ifndef ORTHO_FEATURE_EXTRACTOR_HPP
#define ORTHO_FEATURE_EXTRACTOR_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <ranges>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "imgdata.hpp"
#include "log.hpp"
#include "mem.hpp"
#include "models.h"
#include "ort.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {

template <size_t N>
struct Feature {
  static constexpr size_t descriptor_size = N;

  float                              x, y;
  std::array<float, descriptor_size> desc;

  friend std::ofstream& operator<<(std::ofstream& ofs, const Feature& f) {
    ofs.write(reinterpret_cast<const char*>(&f.x), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(&f.y), sizeof(float));
    ofs.write(reinterpret_cast<const char*>(f.desc.data()), descriptor_size * sizeof(float));
    return ofs;
  }

  friend std::ifstream& operator>>(std::ifstream& ifs, Feature& f) {
    ifs.read(reinterpret_cast<char*>(&f.x), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&f.y), sizeof(float));
    ifs.read(reinterpret_cast<char*>(f.desc.data()), descriptor_size * sizeof(float));
    return ifs;
  }
};

template <typename F>
  requires std::same_as<F, Feature<F::descriptor_size>>
struct Features {
public:

  Features() = default;

  Features(std::initializer_list<F> init) : features(init) {}

  template <std::input_iterator I>
  Features(I first, I last) : features(first, last) {}

  F& operator[](size_t i) noexcept { return features[i]; }

  const F& operator[](size_t i) const noexcept { return features[i]; }

  std::vector<F>& get() noexcept { return features; }

  const std::vector<F>& get() const noexcept { return features; }

  size_t size() const noexcept { return features.size(); }

  bool empty() const noexcept { return features.empty(); }

  void resize(size_t size) noexcept { features.resize(size); }

  void clear() noexcept { features.clear(); }

  void reserve(size_t size) noexcept { features.reserve(size); }

  auto begin() noexcept { return features.begin(); }

  auto end() noexcept { return features.end(); }

  auto begin() const noexcept { return features.begin(); }

  auto end() const noexcept { return features.end(); }

  auto cbegin() const noexcept { return features.cbegin(); }

  auto cend() const noexcept { return features.cend(); }

  auto rbegin() noexcept { return features.rbegin(); }

  auto rend() noexcept { return features.rend(); }

  auto rbegin() const noexcept { return features.rbegin(); }

  auto rend() const noexcept { return features.rend(); }

  auto crbegin() const noexcept { return features.crbegin(); }

  auto crend() const noexcept { return features.crend(); }

  template <typename T>
    requires std::same_as<std::decay_t<T>, F>
  void push_back(T&& feature) noexcept {
    features.push_back(std::forward<T>(feature));
  }

  void pop_back() noexcept { features.pop_back(); }

  friend std::ofstream& operator<<(std::ofstream& ofs, const Features& features) noexcept {
    size_t len = features.get().size();
    ofs << len;
    for (auto&& f : features) {
      ofs << f;
    }
    return ofs;
  }

  friend std::ifstream& operator>>(std::ifstream& ifs, Features& features) noexcept {
    size_t len;
    ifs >> len;
    features.resize(len);
    for (auto&& f : features) {
      ifs >> f;
    }
    return ifs;
  }

private:

  std::vector<F> features;
};

template <typename F>
  requires std::same_as<F, Feature<F::descriptor_size>>
class Extractor {
public:

  using Feature = F;
  using Features = Ortho::Features<F>;

  static constexpr size_t descriptor_size = Feature::descriptor_size;

  virtual ~Extractor() = default;

private:

  static inline cv::Size resolution { 1024, 1024 };

  InferEnv env;
  fs::path temporary_save_path;

  class FeaturesMem : public ManageAble {
  public:

    template <typename T>
      requires std::same_as<std::decay_t<T>, Features>
    FeaturesMem(T&& features) : features(std::forward<T>(features)) {}

    size_t size() const noexcept override {
      if (features.empty()) {
        return 0;
      }
      return features.size() * sizeof(Feature);
    }

    Features features;
  };

protected:

  Extractor(const fs::path& temporary_save_path, const std::string& name, const std::string& model_path) :
    temporary_save_path(temporary_save_path), env(std::format("[{}]", name), model_path) {
    check_or_create_path(temporary_save_path);
  }

  inline void reshape(cv::Mat* img) const { decimate_keep_aspect_ratio(img, resolution); }

  virtual inline void preprocess(cv::Mat* img) const = 0;

  virtual constexpr int64_t get_channels() const noexcept = 0;

  virtual constexpr float get_threshold() const noexcept = 0;

  virtual constexpr int64_t get_keypoint_maxcnt() const noexcept = 0;

public:

  Features get_features(ImgData& img_data) {
    fs::path path = temporary_save_path / (img_data.get_img_stem().string() + ".desc");
    auto     elem = mem.get_node(path.string());
    if (elem) {
      auto&& elem_guard = *elem;
      Features features(elem_guard.get<FeaturesMem>().features);
      return features;
    }

    auto    img_guard = img_data.get_img();
    cv::Mat img_processed = img_guard.get().clone();
    img_guard.unlock();
    reshape(&img_processed);
    preprocess(&img_processed);

    auto    mask_guard = img_data.get_mask();
    cv::Mat mask_processed = mask_guard.get().clone();
    mask_guard.unlock();
    reshape(&mask_processed);

    const int64_t h = mask_processed.rows, w = mask_processed.cols;

    std::vector<float> img_vec(img_processed.begin<float>(), img_processed.end<float>());

    env.set_input("image", img_vec, { 1, get_channels(), h, w });
    if (img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    auto res = env.infer();
    if (img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    img_vec.clear();
    const size_t   cnt = res[env.get_output_index("keypoints")].GetTensorTypeAndShapeInfo().GetShape()[1];
    const int64_t* kps = res[env.get_output_index("keypoints")].GetTensorData<int64_t>();
    const float* scores = res[env.get_output_index("scores")].GetTensorData<float>(),
      * descs = res[env.get_output_index("descriptors")].GetTensorData<float>();
    LOG_DEBUG("Image {} has {} keypoints detected!", img_data.get_img_name().string(), cnt);
    auto v = std::views::iota(0ul, cnt) | std::views::filter([this, &scores, &mask_processed, &kps](const auto& idx) {
      return scores[idx] >= get_threshold()
        && mask_processed.at<unsigned char>(kps[idx * 2 + 1], kps[idx * 2]) != 0;
             });
    std::vector<size_t> indices(v.begin(), v.end());
    if (indices.size() > get_keypoint_maxcnt()) {
      std::nth_element(
          indices.begin(),
          indices.begin() + get_keypoint_maxcnt(),
          indices.end(),
          [&scores](const size_t& lhs, const size_t& rhs) { return scores[lhs] > scores[rhs]; });
      indices.resize(get_keypoint_maxcnt());
    }
    const float wf2 = w / 2.0f, hf2 = h / 2.0f;
    const float max2 = std::max(wf2, hf2);
    auto        u =
      indices | std::views::transform([kps, descs, wf2, hf2, max2](const size_t& idx) {
      std::array<float, descriptor_size> descriptor;
      std::copy_n(descs + idx * descriptor_size, descriptor_size, descriptor.begin());
      return Feature {
          .x = (kps[idx * 2] - wf2) / max2, .y = (kps[idx * 2 + 1] - hf2) / max2, .desc = std::move(descriptor) };
      });
    Features filtered_features(u.begin(), u.end());
    LOG_DEBUG("Image {} has {} keypoints after filter.", img_data.get_img_name().string(), filtered_features.size() / 2);
    mem.register_node(
        path.string(),
        std::make_unique<FeaturesMem>(filtered_features),
        SwapInFunc([path] {
          std::ifstream ifs(path.string(), std::ios::binary);
          if (!ifs.is_open()) {
            throw std::runtime_error("Error: " + path.string() + " could not be opened");
          }
          Features features;
          ifs >> features;
          ifs.close();
          if (ifs.fail()) {
            throw std::runtime_error("Error: " + path.string() + " could not be read");
          }
          return new FeaturesMem(std::move(features));
          }),
        SwapOutFunc([path](ManageAblePtr ptr) {
          if (ptr) {
            std::ofstream ofs(path.string(), std::ios::binary | std::ios::trunc);
            if (!ofs.is_open()) {
              throw std::runtime_error("Error: " + path.string() + " could not be opened");
            }
            auto& features = dynamic_cast<FeaturesMem*>(ptr.get())->features;
            ofs << features;
            if (ofs.fail()) {
              throw std::runtime_error("Error: " + path.string() + " could not be written");
            }
            ofs.close();
          }
          }));
    return filtered_features;
  }
};

class SuperPointExtractor : public Extractor<Feature<256>> {
private:

  inline void preprocess(cv::Mat* img) const override {
    cv::cvtColor(*img, *img, cv::COLOR_BGR2GRAY);
    img->convertTo(*img, CV_32FC1, 1.0f / 255.0f);
  }

  constexpr int64_t get_channels() const noexcept override { return 1; }

  constexpr float get_threshold() const noexcept override { return SUPERPOINT_THRESHOLD; }

  constexpr int64_t get_keypoint_maxcnt() const noexcept override { return SUPERPOINT_KEYPOINT_MAXCNT; }

public:

  SuperPointExtractor(const fs::path& temporary_save_path) :
    Extractor(temporary_save_path, "superpoint", SUPERPOINT_WEIGHT) {}
};

class DiskExtractor : public Extractor<Feature<128>> {
private:

  inline void preprocess(cv::Mat* img) const override {
    if (!img->isContinuous()) {
      *img = img->clone();
    }
    std::vector<cv::Mat> channels;
    cv::split(*img, channels);
    img->create(3, channels[0].rows * channels[0].cols, CV_32FC1);
    channels[2].reshape(1, 1).convertTo(img->row(0), CV_32FC1, 1.0f / 255.0f);
    channels[1].reshape(1, 1).convertTo(img->row(1), CV_32FC1, 1.0f / 255.0f);
    channels[0].reshape(1, 1).convertTo(img->row(2), CV_32FC1, 1.0f / 255.0f);
  }

  constexpr int64_t get_channels() const noexcept override { return 3; }

  constexpr float get_threshold() const noexcept override { return DISK_THRESHOLD; }

  constexpr int64_t get_keypoint_maxcnt() const noexcept override { return DISK_KEYPOINT_MAXCNT; }

public:

  DiskExtractor(const fs::path& temporary_save_path) : Extractor(temporary_save_path, "disk", DISK_WEIGHT) {}
};

} // namespace Ortho
#endif
