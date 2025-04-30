#ifndef ORTHO_FEATURE_EXTRACTOR_HPP
#define ORTHO_FEATURE_EXTRACTOR_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/mem.hpp"
#include "tools/ort.hpp"
#include "tools/report_error.hpp"
#include "tools/utility.hpp"

namespace Ortho {

namespace fs = std::filesystem;

template <size_t N>
struct alignas(128) Feature {
  static constexpr size_t descriptor_size = N;

  using Desc   = std::array<float, descriptor_size>;
  using Buffer = std::array<char, (N * sizeof(float)) + (2 * sizeof(double))>;
  double x, y;
  Desc   desc;

  friend auto operator<<(std::ofstream& ofs, const Feature& feature) noexcept -> std::ofstream& {
    Buffer buffer;
    std::memcpy(buffer.data(), &feature.x, sizeof(double));
    std::memcpy(buffer.data() + sizeof(double), &feature.y, sizeof(double));
    std::memcpy(buffer.data() + (2 * sizeof(double)), feature.desc.data(), N * sizeof(float));
    ofs.write(buffer.data(), buffer.size());
    return ofs;
  }

  friend auto operator>>(std::ifstream& ifs, Feature& feature) noexcept -> std::ifstream& {
    Buffer buffer;
    ifs.read(buffer.data(), buffer.size());
    std::memcpy(&feature.x, buffer.data(), sizeof(double));
    std::memcpy(&feature.y, buffer.data() + sizeof(double), sizeof(double));
    std::memcpy(feature.desc.data(), buffer.data() + (2 * sizeof(double)), N * sizeof(float));
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
  Features(I first, I last) noexcept : features(first, last) {}

  auto operator[](size_t idx) noexcept -> F& { return features[idx]; }

  auto operator[](size_t idx) const noexcept -> const F& { return features[idx]; }

  auto get() noexcept -> std::vector<F>& { return features; }

  auto get() const noexcept -> const std::vector<F>& { return features; }

  [[nodiscard]] auto size() const noexcept -> size_t { return features.size(); }

  [[nodiscard]] auto empty() const noexcept -> bool { return features.empty(); }

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

  friend auto operator<<(std::ofstream& ofs, const Features& features) noexcept -> std::ofstream& {
    size_t len = features.get().size();
    ofs << len;
    for(auto&& feature : features) {
      ofs << feature;
    }
    return ofs;
  }

  friend auto operator>>(std::ifstream& ifs, Features& features) noexcept -> std::ifstream& {
    size_t len = 0;
    ifs >> len;
    features.resize(len);
    for(auto&& feature : features) {
      ifs >> feature;
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

  Extractor(const Extractor&)                    = delete;
  Extractor(Extractor&&)                         = delete;
  auto operator=(const Extractor&) -> Extractor& = delete;
  auto operator=(Extractor&&) -> Extractor&      = delete;

  using Feature  = F;
  using Features = Ortho::Features<F>;

  static constexpr size_t descriptor_size = Feature::descriptor_size;

  virtual ~Extractor() = default;

private:

  InferEnv env;
  fs::path temporary_save_path;

  class FeaturesMem : public ManageAble {
  public:

    template <typename T>
      requires std::same_as<std::decay_t<T>, Features>
    explicit FeaturesMem(T&& features) noexcept : features_(std::forward<T>(features)) {}

    [[nodiscard]] auto size() const noexcept -> size_t override {
      if(features_.empty()) {
        return 0;
      }
      return features_.size() * sizeof(Feature);
    }

    auto features() const noexcept -> const Features& { return features_; }

    auto features() noexcept -> Features& { return features_; }

  private:

    Features features_;
  };

protected:

  Extractor(const fs::path& temporary_save_path, const std::string& name, const std::string& model_path) noexcept :
      temporary_save_path(temporary_save_path), env(std::format("[{}]", name), model_path) {
    check_or_create_path(temporary_save_path);
  }

  void reshape(cv::Mat* img) const noexcept { decimate_keep_aspect_ratio(img, FEATURE_EXTRACTOR_RESOLUTION_LIM); }

  virtual inline void preprocess(cv::Mat* img) const noexcept = 0;

  [[nodiscard]] virtual constexpr auto get_channels() const noexcept -> int64_t = 0;

  [[nodiscard]] virtual constexpr auto get_threshold() const noexcept -> double = 0;

  [[nodiscard]] virtual constexpr auto get_keypoint_maxcnt() const noexcept -> int64_t = 0;

  [[nodiscard]] virtual constexpr auto get_name() const noexcept -> std::string = 0;

public:

  void register_node(const fs::path& path, const Features& features) noexcept {
    Mem::register_node(
        path.string(),
        std::make_unique<FeaturesMem>(features),
        [path] noexcept {
          std::ifstream ifs(path.string(), std::ios::binary);
          if(!ifs.is_open()) {
            report_error("{} could not be opened.", path.string());
          }
          Features features;
          ifs >> features;
          ifs.close();
          if(ifs.fail()) {
            report_error("{} could not be read.", path.string());
          }
          return std::make_unique<FeaturesMem>(std::move(features));
        },
        [path](ManageAblePtr ptr) noexcept {
          if(ptr) {
            std::ofstream ofs(path.string(), std::ios::binary | std::ios::trunc);
            if(!ofs.is_open()) {
              report_error("{} could not be opened.", path.string());
            }
            auto& features = dynamic_cast<FeaturesMem*>(ptr.get())->features();
            ofs << features;
            if(ofs.fail()) {
              report_error("{} could not be written.", path.string());
            }
            ofs.close();
          }
        });
  }

  auto get_features(ImgData& img_data) -> Features {
    fs::path path =
        temporary_save_path / std::format("{}_{}.desc", img_data.rotated_img().get_img_stem().string(), get_name());
    auto elem = Mem::get_node(path.string());
    if(elem) {
      auto&&   elem_guard = *elem;
      Features features{elem_guard.get<FeaturesMem>().features()};
      return features;
    }
    if(fs::exists(path)) {
      std::ifstream ifs(path.string(), std::ios::binary);
      if(ifs.is_open()) {
        Features features;
        ifs >> features;
        ifs.close();
        if(!ifs.fail()) {
          register_node(path, features);
          return features;
        }
      }
    }
    auto&   img_rotated   = img_data.rotated_img();
    auto    img_guard     = img_rotated.get();
    cv::Mat img_processed = img_guard.get().clone();
    img_guard.unlock();
    reshape(&img_processed);
    preprocess(&img_processed);
    const auto [width, height] = img_rotated.get_size();
    std::vector<float> img_vec(img_processed.begin<float>(), img_processed.end<float>());
    env.set_input("image", img_vec, std::vector<int64_t>{1, get_channels(), height, width});
    if(img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    auto res = env.infer();
    if(img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    img_vec.clear();
    const size_t cnt = res[env.get_output_index("keypoints")].GetTensorTypeAndShapeInfo().GetShape()[1];
    const std::span<const int64_t> kps_span{res[env.get_output_index("keypoints")].GetTensorData<int64_t>(), cnt * 2};
    const std::span<const float>   scores_span{res[env.get_output_index("scores")].GetTensorData<float>(), cnt};
    const std::span<const float>
        descs_span{res[env.get_output_index("descriptors")].GetTensorData<float>(), cnt * descriptor_size};
    THIS_LOG_DEBUG("Image {} has {} keypoints detected!", img_data.get_img_name().string(), cnt);
    auto view0 =
        std::views::iota(0UL, cnt) | std::views::filter([this, &scores_span, &img_rotated, &kps_span](const auto& idx) {
          return scores_span[idx] >= get_threshold()
                 && img_rotated.check_valid_pixel(
                     Point<double>{static_cast<double>(kps_span[idx * 2]), static_cast<double>(kps_span[(idx * 2) + 1])});
        });
    std::vector<size_t> indices(view0.begin(), view0.end());
    if(indices.size() > get_keypoint_maxcnt()) {
      std::nth_element(
          indices.begin(),
          indices.begin() + get_keypoint_maxcnt(),
          indices.end(),
          [&scores_span](const size_t& lhs, const size_t& rhs) { return scores_span[lhs] > scores_span[rhs]; });
      indices.resize(get_keypoint_maxcnt());
    }
    const double   wf2   = width / 2.0;
    const double   hf2   = height / 2.0;
    const double   max2  = std::max(wf2, hf2);
    auto           view1 = indices | std::views::transform([&kps_span, &descs_span, wf2, hf2, max2](const size_t& idx) {
                   std::array<float, descriptor_size> descriptor;
                   std::copy_n(&descs_span[idx * descriptor_size], descriptor_size, descriptor.begin());
                   return Feature{
                                 .x    = (static_cast<double>(kps_span[idx * 2]) - wf2) / max2,
                                 .y    = (static_cast<double>(kps_span[(idx * 2) + 1]) - hf2) / max2,
                                 .desc = std::move(descriptor)};
                 });
    const Features filtered_features(view1.begin(), view1.end());
    THIS_LOG_DEBUG(
        "Image {} has {} keypoints after filter.", img_data.get_img_name().string(), filtered_features.size() / 2);
    if(!fs::exists(path)) {
      std::ofstream ofs(path.string(), std::ios::binary | std::ios::trunc);
      if(ofs.is_open()) {
        ofs << filtered_features;
        ofs.close();
      }
    }
    register_node(path, filtered_features);
    return filtered_features;
  }
};

class SuperPointExtractor : public Extractor<Feature<256>> {
private:

  void preprocess(cv::Mat* img) const noexcept override {
    cv::cvtColor(*img, *img, cv::COLOR_BGR2GRAY);
    img->convertTo(*img, CV_32FC1, 1.F / 255.F);
  }

  [[nodiscard]] constexpr auto get_channels() const noexcept -> int64_t override { return 1; }

  [[nodiscard]] constexpr auto get_threshold() const noexcept -> double override { return SUPERPOINT_THRESHOLD; }

  [[nodiscard]] constexpr auto get_keypoint_maxcnt() const noexcept -> int64_t override {
    return SUPERPOINT_KEYPOINT_MAXCNT;
  }

  [[nodiscard]] constexpr auto get_name() const noexcept -> std::string override { return "superpoint"; }

public:

  explicit SuperPointExtractor(const fs::path& temporary_save_path) :
      Extractor(temporary_save_path, "superpoint", SUPERPOINT_WEIGHT) {}
};

class DiskExtractor : public Extractor<Feature<128>> {
private:

  void preprocess(cv::Mat* img) const noexcept override {
    if(!img->isContinuous()) {
      *img = img->clone();
    }
    std::vector<cv::Mat> channels;
    cv::split(*img, channels);
    img->create(3, channels[0].rows * channels[0].cols, CV_32FC1);
    channels[2].reshape(1, 1).convertTo(img->row(0), CV_32FC1, 1.F / 255.F);
    channels[1].reshape(1, 1).convertTo(img->row(1), CV_32FC1, 1.F / 255.F);
    channels[0].reshape(1, 1).convertTo(img->row(2), CV_32FC1, 1.F / 255.F);
  }

  [[nodiscard]] constexpr auto get_channels() const noexcept -> int64_t override { return 3; }

  [[nodiscard]] constexpr auto get_threshold() const noexcept -> double override { return DISK_THRESHOLD; }

  [[nodiscard]] constexpr auto get_keypoint_maxcnt() const noexcept -> int64_t override { return DISK_KEYPOINT_MAXCNT; }

  [[nodiscard]] constexpr auto get_name() const noexcept -> std::string override { return "disk"; }

public:

  explicit DiskExtractor(const fs::path& temporary_save_path) : Extractor(temporary_save_path, "disk", DISK_WEIGHT) {}
};

} // namespace Ortho
#endif
