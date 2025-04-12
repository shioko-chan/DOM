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
#include <unordered_set>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "pose_intrinsic.hpp"
#include "rotate_rectify.hpp"
#include "utility.hpp"

namespace fs = std::filesystem;

namespace Ortho {

struct ImgData {
public:

  ImgData() = default;

  ImgData(
      Pose&&      pose,
      Intrinsic&& intrinsic,
      ExifXmp&&   exif_xmp,
      fs::path    img_path,
      fs::path    temp_save_path,
      cv::Size    img_size) :
      pose(std::move(pose)), intrinsic(std::move(intrinsic)), exif_xmp(std::move(exif_xmp)), img_path(img_path),
      temp_save_path(temp_save_path), img_size(std::move(img_size)) {
    check_or_create_path(temp_save_path);
  }

  Image rotate_rectified() {
    if(!img_rotated.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated;
  }

  Image rotate_rectified_mask() {
    if(!img_rotated_mask.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated_mask;
  }

  ImgRefGuard get_rotate_rectified() {
    if(!img_rotated.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated.get();
  }

  ImgRefGuard get_rotate_rectified_mask() {
    if(!img_rotated_mask.is_initialized()) {
      rotate_rectify();
    }
    return img_rotated_mask.get();
  }

  const Points<float>& get_spans() {
    if(ground_points.empty()) {
      rotate_rectify();
    }
    return ground_points;
  }

  Points<float> world2img(const Points<float>& points) {
    if(!world2img_) {
      rotate_rectify();
    }
    return world2img_(points);
  }

  void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree, const float& altitude_ref_) {
    pose.set_reference(latitude_ref_degree, longitude_ref_degree, altitude_ref_);
    reference_set = true;
  }

  const Angle& get_latitude() const { return pose.latitude; }

  const Angle& get_longitude() const { return pose.longitude; }

  float get_altitude() const { return pose.altitude; }

  const Angle& get_yaw() const { return pose.yaw; }

  const Angle& get_pitch() const { return pose.pitch; }

  const Angle& get_roll() const { return pose.roll; }

  const cv::Mat& get_rotation_matrix() const { return pose.R(); }

  const Point<float>& get_coord() const { return pose.coord; }

  const cv::Mat& get_camera_matrix() const { return intrinsic.K(); }

  const cv::Mat& get_distortion_coefficients() const { return intrinsic.D(); }

  const fs::path& get_img_path() const { return img_path; }

  fs::path get_img_name() const { return img_path.filename(); }

  fs::path get_img_stem() const { return img_path.stem(); }

  fs::path get_img_extension() const { return img_path.extension(); }

  friend std::ostream& operator<<(std::ostream& os, const ImgData& data) {
    os << "Camera Matrix: " << data.intrinsic << "\n"
       << "Pose: " << data.pose << "\n";
    return os;
  }

  void rotate_rectify() {
    if(!reference_set) {
      throw std::runtime_error("Error: Reference coordinate not set");
    }
    cv::Mat img = cv::imread(img_path.string());
    if(img.empty()) {
      throw std::runtime_error(img_path.string() + " could not be read");
    }
    auto&& [rotate_img, mask, ground_points, world2img] = Ortho::rotate_rectify(img_size, pose, intrinsic, img);
    this->ground_points                                 = std::move(ground_points);
    this->world2img_                                    = std::move(world2img);
    this->img_rotated.delay_initialize(
        temp_save_path / std::format("{}_r{}", img_path.stem().string(), img_path.extension().string()),
        std::move(rotate_img));
    this->img_rotated_mask.delay_initialize(
        temp_save_path / std::format("{}_rm{}", img_path.stem().string(), img_path.extension().string()),
        std::move(mask));
  }

private:

  cv::Size       img_size;
  fs::path       temp_save_path, img_path;
  Pose           pose;
  Intrinsic      intrinsic;
  Image          img_rotated, img_rotated_mask;
  ExifXmp        exif_xmp;
  Points<float>  ground_points;
  PointsPipeline world2img_;
  bool           reference_set{false};
};

struct ImgsData {
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
    std::vector<float>          latitudes, longitudes, altitudes;
    for(auto&& data : imgs_data) {
      latitudes.push_back(data.get_latitude().degrees());
      longitudes.push_back(data.get_longitude().degrees());
      altitudes.push_back(data.get_altitude());
    }
    size_t n = latitudes.size() / 2;
    std::nth_element(latitudes.begin(), latitudes.begin() + n, latitudes.end());
    std::nth_element(longitudes.begin(), longitudes.begin() + n, longitudes.end());
    std::nth_element(altitudes.begin(), altitudes.begin() + n, altitudes.end());
    float latitude_ref  = latitudes[n];
    float longitude_ref = longitudes[n];
    float altitude_ref  = altitudes[n];
    for(auto&& data : imgs_data) {
      data.set_reference(latitude_ref, longitude_ref, altitude_ref);
    }
  }

private:

  std::vector<ImgData> imgs_data;
  std::mutex           mutex;
};

class ImgDataFactory {
private:

  static inline const std::unordered_set<std::string> extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

  struct ExifKey {
    static inline const std::string              width  = "Exif.Photo.PixelXDimension";
    static inline const std::string              height = "Exif.Photo.PixelYDimension";
    static inline const std::vector<std::string> keys   = {width, height};
  };

public:

  static bool validate(const fs::path& path) {
    if(!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      WARN("Error: {} is not a valid image file", path.string());
      return false;
    }
    ExifXmp exif_xmp(path);
    if(!IntrinsicFactory::validate(exif_xmp)) {
      return false;
    }
    if(!PoseFactory::validate(exif_xmp)) {
      return false;
    }
    const auto& exif_data = exif_xmp.exif_data();
    for(const auto& key : ExifKey::keys) {
      if(exif_data.findKey(Exiv2::ExifKey(key)) == exif_data.end()) {
        WARN("{}: Key {} not found in Exif data", path.string(), key);
        return false;
      }
    }
    return true;
  }

  static ImgData build(const fs::path& path, fs::path temp_save_path) {
    ExifXmp      exif_xmp(path);
    const auto&  exif_data = exif_xmp.exif_data();
    unsigned int w         = exif_data.findKey(Exiv2::ExifKey(ExifKey::width))->toUint32(),
                 h         = exif_data.findKey(Exiv2::ExifKey(ExifKey::height))->toUint32();
    Intrinsic intrinsic    = IntrinsicFactory::build(exif_xmp.exif_data(), w, h);
    Pose      pose         = PoseFactory::build(exif_xmp.xmp_data());
    return ImgData{std::move(pose), std::move(intrinsic), std::move(exif_xmp), path, temp_save_path, cv::Size(w, h)};
  }
};
} // namespace Ortho

#endif