#ifndef SKYMERGE_FEATURE_EXTRACTOR_HPP
#define SKYMERGE_FEATURE_EXTRACTOR_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "config.hpp"
#include "ds/imgdata.hpp"
#include "ds/mem.hpp"
#include "nn/ort.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/report.hpp"
#include "tools/utility.hpp"

namespace SkyMerge {

template <size_t N>
class Extractor {
public:

  Extractor(const Extractor&)                    = delete;
  Extractor(Extractor&&)                         = delete;
  auto operator=(const Extractor&) -> Extractor& = delete;
  auto operator=(Extractor&&) -> Extractor&      = delete;

  struct alignas(64) Features {
  public:

    Features() noexcept = default;

    Features(size_t len, std::vector<float> kpnts, std::vector<float> descs) noexcept :
        len(len), kpnts(std::move(kpnts)), descs(std::move(descs)) {
      THIS_ASSERTION_SHOULD_EQ(this->kpnts.size(), len * 2, "Keypoints size mismatch!");
      THIS_ASSERTION_SHOULD_EQ(this->descs.size(), len * N, "Descriptors size mismatch!");
    }

    friend auto operator<<(std::ofstream& ofs, const Features& features) noexcept -> std::ofstream& {
      ofs.write(reinterpret_cast<const char*>(&features.len), sizeof(size_t));
      ofs.write(reinterpret_cast<const char*>(features.kpnts.data()), features.kpnts.size() * sizeof(float));
      ofs.write(reinterpret_cast<const char*>(features.descs.data()), features.descs.size() * sizeof(float));
      return ofs;
    }

    friend auto operator>>(std::ifstream& ifs, Features& features) noexcept -> std::ifstream& {
      ifs.read(reinterpret_cast<char*>(&features.len), sizeof(size_t));
      features.kpnts.resize(features.len * 2);
      features.descs.resize(features.len * N);
      ifs.read(reinterpret_cast<char*>(features.kpnts.data()), features.kpnts.size() * sizeof(float));
      ifs.read(reinterpret_cast<char*>(features.descs.data()), features.descs.size() * sizeof(float));
      return ifs;
    }

    size_t             len{};
    std::vector<float> kpnts;
    std::vector<float> descs;
  };

  static constexpr size_t descriptor_size = N;

  virtual ~Extractor() = default;

private:

  InferEnv              env;
  std::filesystem::path temporary_save_path;

  class FeaturesManaged : public Managed {
  public:

    template <typename T>
      requires std::same_as<std::decay_t<T>, Features>
    explicit FeaturesManaged(T&& features) noexcept : features(std::forward<T>(features)) {}

    [[nodiscard]] auto size() const noexcept -> size_t override {
      return features.len * (descriptor_size + 2) * sizeof(float);
    }

    Features features;
  };

  class FeaturesDeviceManaged : public Managed {
  public:

    FeaturesDeviceManaged(size_t len, float* kpnts, float* descs) noexcept : len(len), kpnts(kpnts), descs(descs) {}

    [[nodiscard]] auto size() const noexcept -> size_t override { return len * (descriptor_size + 2) * sizeof(float); }

    size_t len;
    float* kpnts{nullptr};
    float* descs{nullptr};
  };

  auto infer(ImgData& img_data) -> Features {
    const auto& img_rotated   = img_data.rotated_img();
    auto        img_guard     = img_rotated.get();
    cv::Mat     img_processed = img_guard.get().clone();
    img_guard.unlock();
    preprocess(&img_processed);
    const auto [width, height] = img_rotated.get_size();
    std::vector<float> img_vec{img_processed.begin<float>(), img_processed.end<float>()};
    env.set_input("image", img_vec, std::vector<std::int64_t>{1, get_channels(), height, width});
    if(img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    auto res = env.infer();
    if(img_vec.empty()) {
      throw std::runtime_error("Error: Image is empty");
    }
    img_vec.clear();
    const size_t cnt = res[env.get_output_index("keypoints")].GetTensorTypeAndShapeInfo().GetShape()[1];
    const std::span<const std::int64_t>
        kps_span{res[env.get_output_index("keypoints")].GetTensorData<std::int64_t>(), cnt * 2};
    const std::span<const float> scores_span{res[env.get_output_index("scores")].GetTensorData<float>(), cnt};
    const std::span<const float>
        descs_span{res[env.get_output_index("descriptors")].GetTensorData<float>(), cnt * descriptor_size};
    THIS_LOG_DEBUG("Image {} has {} keypoints detected!", img_data.rotated_img().get_img_name().string(), cnt);
    auto view0 =
        std::views::iota(0UL, cnt) | std::views::filter([this, &scores_span, &img_rotated, &kps_span](const auto& idx) {
          return scores_span[idx] >= get_threshold()
                 && img_rotated.check_valid_pixel(
                     Point<float>{static_cast<float>(kps_span[idx * 2]), static_cast<float>(kps_span[(idx * 2) + 1])});
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

    auto kpnt_v =
        indices | std::views::transform([&kps_span](const size_t& idx) {
          return Point<double>{static_cast<double>(kps_span[idx * 2]), static_cast<double>(kps_span[(idx * 2) + 1])};
        });
    Points<double> kpnts{kpnt_v.begin(), kpnt_v.end()};
    img_data.set_kpnts(kpnts);

    const float wf2  = static_cast<float>(width) / 2.0F;
    const float hf2  = static_cast<float>(height) / 2.0F;
    const float max2 = std::max(wf2, hf2);

    std::vector<float> kpnts_filtered;
    kpnts_filtered.reserve(indices.size() * 2);
    std::vector<float> descs_filtered;
    descs_filtered.reserve(indices.size() * descriptor_size);
    for(auto idx : indices) {
      kpnts_filtered.push_back((static_cast<float>(kps_span[idx * 2]) - wf2) / max2);
      kpnts_filtered.push_back((static_cast<float>(kps_span[(idx * 2) + 1]) - hf2) / max2);
      descs_filtered
          .insert(descs_filtered.end(), &descs_span[idx * descriptor_size], &descs_span[(idx + 1) * descriptor_size]);
    }
    Features filtered_features{indices.size(), std::move(kpnts_filtered), std::move(descs_filtered)};
    THIS_LOG_DEBUG(
        "Image {} has {} keypoints after filter.",
        img_data.rotated_img().get_img_name().string(),
        filtered_features.size() / 2);
    return filtered_features;
  }

  void register_node(const std::filesystem::path& path, const Features& features) const noexcept {
    HostMem::register_node(
        path.string(),
        std::make_unique<FeaturesManaged>(features),
        [path] noexcept {
          std::ifstream ifs(path.string(), std::ios::binary);
          if(!ifs.is_open()) {
            terminate_with_error("{} could not be opened.", path.string());
          }
          Features features;
          ifs >> features;
          ifs.close();
          if(ifs.fail()) {
            terminate_with_error("{} could not be read.", path.string());
          }
          return std::make_unique<FeaturesManaged>(std::move(features));
        },
        [path](ManagedPtr ptr) noexcept {
          if(ptr) {
            std::ofstream ofs(path.string(), std::ios::binary | std::ios::trunc);
            if(!ofs.is_open()) {
              terminate_with_error("{} could not be opened.", path.string());
            }
            auto& features = dynamic_cast<FeaturesManaged*>(ptr.get())->features;
            ofs << features;
            if(ofs.fail()) {
              terminate_with_error("{} could not be written.", path.string());
            }
            ofs.close();
          }
        });

    DeviceMem::register_node(
        path.string(),
        nullptr,
        [path] noexcept {
          auto guard = HostMem::get_node(path.string());
          if(!guard) {
            terminate_with_error("{} not found in host memory.", path.string());
          }
          const auto& features = guard->get<FeaturesManaged>().features;
          float*      kpnts    = nullptr;
          float*      descs    = nullptr;

          cudaError_t err = cudaSuccess;
          err             = cudaMalloc(&kpnts, features.kpnts.size() * sizeof(float));
          if(err != cudaSuccess) {
            terminate_with_error("cudaMalloc failed for kpnts: {}", cudaGetErrorString(err));
          }
          err = cudaMalloc(&descs, features.descs.size() * sizeof(float));
          if(err != cudaSuccess) {
            cudaFree(kpnts);
            terminate_with_error("cudaMalloc failed for descs: {}", cudaGetErrorString(err));
          }
          err = cudaMemcpy(kpnts, features.kpnts.data(), features.kpnts.size() * sizeof(float), cudaMemcpyHostToDevice);
          if(err != cudaSuccess) {
            cudaFree(kpnts);
            cudaFree(descs);
            terminate_with_error("cudaMemcpy failed for kpnts: {}", cudaGetErrorString(err));
          }
          err = cudaMemcpy(descs, features.descs.data(), features.descs.size() * sizeof(float), cudaMemcpyHostToDevice);
          if(err != cudaSuccess) {
            cudaFree(kpnts);
            cudaFree(descs);
            terminate_with_error("cudaMemcpy failed for descs: {}", cudaGetErrorString(err));
          }
          return std::make_unique<FeaturesDeviceManaged>(features.len, kpnts, descs);
        },
        [path](ManagedPtr ptr) noexcept {
          if(ptr) {
            auto* features = dynamic_cast<FeaturesDeviceManaged*>(ptr.get());
            cudaFree(features->kpnts);
            cudaFree(features->descs);
          }
        });
  }

protected:

  Extractor(const std::filesystem::path& temporary_save_path, const std::string& name, const std::string& model_path) noexcept
      : temporary_save_path(temporary_save_path), env(std::format("[{}]", name), model_path) {
    check_or_create_path(temporary_save_path);
  }

  virtual inline void preprocess(cv::Mat* img) const noexcept = 0;

  [[nodiscard]] virtual constexpr auto get_channels() const noexcept -> std::int64_t = 0;

  [[nodiscard]] virtual constexpr auto get_threshold() const noexcept -> double = 0;

  [[nodiscard]] virtual constexpr auto get_keypoint_maxcnt() const noexcept -> std::int64_t = 0;

public:

  auto extract_features(ImgData& img_data) -> bool {
    std::filesystem::path path =
        temporary_save_path / std::format("{}.desc", img_data.rotated_img().get_img_stem().string());
    if(HostMem::contain_node(path.string())) {
      return true;
    }
    auto filtered_features = infer(img_data);
    if(filtered_features.len == 0) {
      THIS_LOG_INFO("Image {} has no valid feature!", img_data.rotated_img().get_img_name().string());
      return false;
    }
    register_node(path, filtered_features);
    return true;
  }

  struct DeviceFeaturesRefGuard {
    DeviceFeaturesRefGuard(DeviceFeaturesRefGuard&&) noexcept = default;

    DeviceFeaturesRefGuard(const DeviceFeaturesRefGuard&)                    = delete;
    auto operator=(const DeviceFeaturesRefGuard&) -> DeviceFeaturesRefGuard& = delete;
    auto operator=(DeviceFeaturesRefGuard&&) -> DeviceFeaturesRefGuard&      = delete;

    ~DeviceFeaturesRefGuard() noexcept = default;

    explicit DeviceFeaturesRefGuard(RefGuard&& refguard) noexcept : refguard(std::move(refguard)) {}

    auto get_descs() noexcept -> std::span<float> {
      return {refguard.get<FeaturesDeviceManaged>().descs, refguard.get<FeaturesDeviceManaged>().len * descriptor_size};
    }

    auto get_kpnts() noexcept -> std::span<float> {
      return {refguard.get<FeaturesDeviceManaged>().kpnts, refguard.get<FeaturesDeviceManaged>().len * 2};
    }

    auto get_len() noexcept -> size_t { return refguard.get<FeaturesDeviceManaged>().len; }

    void unlock() noexcept { refguard.unlock(); }

  private:

    RefGuard refguard;
  };

  [[nodiscard]] auto get_features_on_device(const ImgData& img_data) const -> DeviceFeaturesRefGuard {
    std::filesystem::path path =
        temporary_save_path / std::format("{}.desc", img_data.rotated_img().get_img_stem().string());
    return DeviceFeaturesRefGuard{*DeviceMem::get_node(path.string())};
  }
};

class SuperPointExtractor : public Extractor<256> {
private:

  void preprocess(cv::Mat* img) const noexcept override {
    cv::cvtColor(*img, *img, cv::COLOR_BGR2GRAY);
    img->convertTo(*img, CV_32FC1, 1.F / 255.F);
  }

  [[nodiscard]] constexpr auto get_channels() const noexcept -> std::int64_t override { return 1; }

  [[nodiscard]] constexpr auto get_threshold() const noexcept -> double override { return SUPERPOINT_THRESHOLD; }

  [[nodiscard]] constexpr auto get_keypoint_maxcnt() const noexcept -> std::int64_t override {
    return SUPERPOINT_KEYPOINT_MAXCNT;
  }

public:

  explicit SuperPointExtractor(const std::filesystem::path& temporary_save_path) :
      Extractor(temporary_save_path, "superpoint", SUPERPOINT_WEIGHT) {}
};

class DiskExtractor : public Extractor<128> {
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

  [[nodiscard]] constexpr auto get_channels() const noexcept -> std::int64_t override { return 3; }

  [[nodiscard]] constexpr auto get_threshold() const noexcept -> double override { return DISK_THRESHOLD; }

  [[nodiscard]] constexpr auto get_keypoint_maxcnt() const noexcept -> std::int64_t override {
    return DISK_KEYPOINT_MAXCNT;
  }

public:

  explicit DiskExtractor(const std::filesystem::path& temporary_save_path) :
      Extractor(temporary_save_path, "disk", DISK_WEIGHT) {}
};

} // namespace SkyMerge
#endif
