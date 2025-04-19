#ifndef ORTHO_IMGDATA_HPP
#define ORTHO_IMGDATA_HPP

#include <algorithm>
#include <cmath>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "pose_intrinsic.hpp"
#include "rotate_rectify.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {

struct Kpnts {
public:

  size_t append(const Point<float>& kpnt) {
    auto it = kpnts_map.find(kpnt);
    if (it == kpnts_map.end()) {
      size_t idx = kpnts_map.size();
      kpnts_map.emplace(kpnt, idx);
      kpnts_map_rev.emplace(idx, kpnt);
      return idx;
    } else {
      return it->second;
    }
  }

  template <std::ranges::range R>
  auto append(const R& kpnts) {
    return kpnts | std::views::transform([this](const auto& kpnt) { return append(kpnt); });
  }

  Point<float> get(size_t idx) const {
    auto it = kpnts_map_rev.find(idx);
    if (it == kpnts_map_rev.end()) {
      throw std::runtime_error("Error: Keypoint not found");
    }
    return it->second;
  }

  size_t size() { return kpnts_map.size(); }

private:

  PointUMap<float, size_t>    kpnts_map;
  PointUMapRev<size_t, float> kpnts_map_rev;
};

class ImgData {
  friend class ImgsData;
public:

  ImgData() = default;

  ImgData(Pose&& pose, ExifXmp&& exif_xmp, fs::path img_path, fs::path temp_save_path, cv::Size img_size) :
    pose(std::move(pose)), exif_xmp(std::move(exif_xmp)), img_path(img_path), temp_save_path(temp_save_path),
    img_size(std::move(img_size)) {
    check_or_create_path(temp_save_path);
  }

  Image img() {
    if (!img_rotated.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated;
  }

  Image mask() {
    if (!img_rotated_mask.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated_mask;
  }

  ImgRefGuard get_img() {
    if (!img_rotated.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated.get();
  }

  ImgRefGuard get_mask() {
    if (!img_rotated_mask.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated_mask.get();
  }

  cv::Size get_size() {
    if (img_size_rotated.area() == 0) {
      rotate_rectify();
    }
    return img_size_rotated;
  }

  void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree) {
    pose.set_reference(latitude_ref_degree, longitude_ref_degree);
    reference_set = true;
  }

  const Angle& get_latitude() const { return pose.latitude; }

  const Angle& get_longitude() const { return pose.longitude; }

  cv::Mat R_proj() const { return pose.R_proj(); }

  cv::Mat t_proj() const { return pose.t_proj(); }

  cv::Mat K() {
    Intrinsic intrinsic {
        static_cast<float>(img_size_rotated.width),
        static_cast<float>(img_size_rotated.height),
        exif_xmp.exif_data().findKey(Exiv2::ExifKey("Exif.Photo.FocalLength"))->toFloat() };
    return intrinsic.K();
  }

  cv::Mat D() { return cv::Mat::zeros(5, 1, CV_32F); }

  cv::Mat projection_matrix() { return get_projection_matrix(R_proj(), t_proj(), K()); }

  const Point<float>& get_coord() const { return pose.coord; }

  const fs::path& get_img_path() const { return img_path; }

  fs::path get_img_name() const { return img_path.filename(); }

  fs::path get_img_stem() const { return img_path.stem(); }

  fs::path get_img_extension() const { return img_path.extension(); }

  void rotate_rectify() {
    if (!reference_set) {
      throw std::runtime_error("Error: Reference coordinate not set");
    }
    cv::Mat img = cv::imread(img_path.string());
    if (img.empty()) {
      throw std::runtime_error(img_path.string() + " could not be read");
    }
    auto&& [rotate_img, mask] = Ortho::rotate_rectify(img_size, pose.R_proj().t_proj(), img);
    img_size_rotated = rotate_img.size();
    this->img_rotated.delay_initialize(
        temp_save_path / std::format("{}_r{}", img_path.stem().string(), img_path.extension().string()),
        std::move(rotate_img));
    this->img_rotated_mask.delay_initialize(
        temp_save_path / std::format("{}_rm{}", img_path.stem().string(), img_path.extension().string()),
        std::move(mask));
  }

  Kpnts kpnts;

private:

  cv::Size img_size, img_size_rotated;
  fs::path temp_save_path, img_path;
  Pose     pose;
  Image    img_rotated, img_rotated_mask;
  ExifXmp  exif_xmp;
  bool     reference_set { false };

};

class ImgsData {
public:

  ImgsData() = default;

  ImgsData(std::initializer_list<ImgData> init) : imgs_data(init) {}

  template <std::input_iterator I>
  ImgsData(I first, I last) : imgs_data(first, last) {}

  ImgData& operator[](size_t i) { return imgs_data[i]; }

  const ImgData& operator[](size_t i) const { return imgs_data[i]; }

  std::vector<ImgData>& get() { return imgs_data; }

  const std::vector<ImgData>& get() const { return imgs_data; }

  size_t size() const { return imgs_data.size(); }

  bool empty() const { return imgs_data.empty(); }

  void resize(size_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.resize(size);
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.clear();
  }

  void reserve(size_t size) {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.reserve(size);
  }

  auto begin() noexcept { return imgs_data.begin(); }

  auto end() noexcept { return imgs_data.end(); }

  auto begin() const noexcept { return imgs_data.begin(); }

  auto end() const noexcept { return imgs_data.end(); }

  auto cbegin() const noexcept { return imgs_data.cbegin(); }

  auto cend() const noexcept { return imgs_data.cend(); }

  auto rbegin() noexcept { return imgs_data.rbegin(); }

  auto rend() noexcept { return imgs_data.rend(); }

  auto rbegin() const noexcept { return imgs_data.rbegin(); }

  auto rend() const noexcept { return imgs_data.rend(); }

  auto crbegin() const noexcept { return imgs_data.crbegin(); }

  auto crend() const noexcept { return imgs_data.crend(); }

  template <typename T>
    requires std::same_as<std::decay_t<T>, ImgData>
  void push_back(T&& data) {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.push_back(std::forward<T>(data));
  }

  void pop_back() {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.pop_back();
  }

  void find_and_set_reference_coord() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<float>          latitudes, longitudes;
    for (auto&& data : imgs_data) {
      latitudes.push_back(data.latitude.degrees());
      longitudes.push_back(data.longitude.degrees());
    }
    size_t n = latitudes.size() / 2;
    std::nth_element(latitudes.begin(), latitudes.begin() + n, latitudes.end());
    std::nth_element(longitudes.begin(), longitudes.begin() + n, longitudes.end());
    float latitude_ref = latitudes[n];
    float longitude_ref = longitudes[n];
    for (auto&& data : imgs_data) {
      data.set_reference(latitude_ref, longitude_ref);
    }
  }

private:

  std::vector<ImgData> imgs_data;
  std::mutex           mutex;
};

class ImgDataFactory {
private:

  static inline const std::unordered_set<std::string> extensions =
  { ".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP" };

  struct ExifKey {
    static inline const std::string              width = "Exif.Photo.PixelXDimension";
    static inline const std::string              height = "Exif.Photo.PixelYDimension";
    static inline const std::vector<std::string> keys = { width, height };
  };

public:

  static bool validate(const fs::path& path) {
    if (!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      LOG_WARN("Error: {} is not a valid image file", path.string());
      return false;
    }
    ExifXmp exif_xmp(path);
    if (!PoseFactory::validate(exif_xmp)) {
      return false;
    }
    const auto& exif_data = exif_xmp.exif_data();
    for (const auto& key : ExifKey::keys) {
      if (exif_data.findKey(Exiv2::ExifKey(key)) == exif_data.end()) {
        LOG_WARN("{}: Key {} not found in Exif data", path.string(), key);
        return false;
      }
    }
    return true;
  }

  static ImgData build(const fs::path& path, fs::path temp_save_path) {
    ExifXmp      exif_xmp(path);
    const auto& exif_data = exif_xmp.exif_data();
    unsigned int w = exif_data.findKey(Exiv2::ExifKey(ExifKey::width))->toUint32(),
      h = exif_data.findKey(Exiv2::ExifKey(ExifKey::height))->toUint32();
    Pose pose = PoseFactory::build(exif_xmp.xmp_data());
    return ImgData { std::move(pose), std::move(exif_xmp), path, temp_save_path, cv::Size(w, h) };
  }
};
} // namespace Ortho

#endif
