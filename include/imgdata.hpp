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

  ImgData(Pose&& pose, Intrinsic&& intrinsic, Image&& img, ExifXmp&& exif_xmp, fs::path temp_save_path) :
      pose(std::move(pose)), intrinsic(std::move(intrinsic)), img(std::move(img)), exif_xmp(std::move(exif_xmp)),
      temp_save_path(temp_save_path) {}

  ImgRefGuard get_original_img() const { return img.get(); }

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

  const fs::path& get_img_path() const { return img.get_img_path(); }

  fs::path get_img_name() const { return img.get_img_name(); }

  fs::path get_img_stem() const { return img.get_img_stem(); }

  fs::path get_img_extension() const { return img.get_img_extension(); }

  friend std::ostream& operator<<(std::ostream& os, const ImgData& data) {
    os << "Camera Matrix: " << data.intrinsic << "\n"
       << "Pose: " << data.pose << "\n";
    return os;
  }

private:

  void rotate_rectify() {
    if(!reference_set) {
      throw std::runtime_error("Error: Reference coordinate not set");
    }
    auto img_guard = img.get();
    auto&& [rotate_img, mask, ground_points, world2img] =
        Ortho::rotate_rectify(img_guard.get().size(), pose, intrinsic, img_guard.get());
    this->ground_points = std::move(ground_points);
    this->world2img_    = std::move(world2img);
    this->img_rotated.delay_initialize(
        temp_save_path / (img.get_img_stem().string() + "_rotated" + img.get_img_extension().string()),
        std::move(rotate_img));
    this->img_rotated_mask.delay_initialize(
        temp_save_path / (img.get_img_stem().string() + "_rotated_mask" + img.get_img_extension().string()),
        std::move(mask));
  }

  fs::path       temp_save_path;
  Pose           pose;
  Intrinsic      intrinsic;
  Image          img, img_rotated, img_rotated_mask;
  ExifXmp        exif_xmp;
  Points<float>  ground_points;
  PointsPipeline world2img_;
  bool           reference_set = false;
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

public:

  static std::optional<ImgData> build(fs::path path, fs::path temp_save_path) {
    if(!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      ERROR("Error: {} is not a valid image file", path.string());
      return std::nullopt;
    }
    ExifXmp exif_xmp(path);
    if(!IntrinsicFactory::validate(exif_xmp)) {
      return std::nullopt;
    }
    if(!PoseFactory::validate(exif_xmp)) {
      return std::nullopt;
    }
    auto img            = Image(path);
    auto imgref         = img.get();
    auto [w, h]         = imgref.get().size();
    Intrinsic intrinsic = IntrinsicFactory::build(exif_xmp.exif_data(), w, h);
    Pose      pose      = PoseFactory::build(exif_xmp.xmp_data());
    return ImgData{std::move(pose), std::move(intrinsic), std::move(img), std::move(exif_xmp), temp_save_path};
  }
};
} // namespace Ortho

#endif