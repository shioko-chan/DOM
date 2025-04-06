#ifndef IMGDATA_HPP
#define IMGDATA_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
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

using MatRefLockPair = TRefLockPair<cv::Mat>;

struct ImgData {
public:

  ImgData() = default;

  ImgData(Pose&& pose, Intrinsic&& intrinsic, Image&& img) :
      pose(std::move(pose)), intrinsic(std::move(intrinsic)), img(std::move(img)) {}

  MatRefLockPair get_original_img() const {
    auto [img_, lock] = img.img().value();
    return {img_.get(), std::move(lock)};
  }

  MatRefLockPair get_rotate_rectified() {
    auto res = img.rotate_rectified();
    if(res.has_value()) {
      auto&& [img_, lock] = res.value();
      return {img_.get(), std::move(lock)};
    }
    rotate_rectify();
    auto [img_, lock] = img.rotate_rectified().value();
    return {img_.get(), std::move(lock)};
  }

  MatRefLockPair get_rotate_rectified_mask() {
    auto res = img.rotate_rectified_mask();
    if(res.has_value()) {
      auto&& [img_, lock] = res.value();
      return {img_.get(), std::move(lock)};
    }
    rotate_rectify();
    auto [img_, lock] = img.rotate_rectified_mask().value();
    return {img_.get(), std::move(lock)};
  }

  const Points<float>& get_spans() {
    if(!ground_points.empty()) {
      return ground_points;
    }
    rotate_rectify();
    return ground_points;
  }

  Points<float> world2img(const Points<float>& points) {
    if(world2img_) {
      return world2img_(points);
    }
    rotate_rectify();
    return world2img_(points);
  }

  void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree, const float& altitude_ref_) {
    pose.set_reference(latitude_ref_degree, longitude_ref_degree, altitude_ref_);
    reference_set = true;
  }

  const Angle& get_latitude() const { return pose.latitude; }

  const Angle& get_longitude() const { return pose.longitude; }

  float get_altitude() const { return pose.altitude; }

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
    auto [img__, lock]                                  = img.img().value();
    auto&& img_                                         = img__.get();
    auto&& [rotate_img, mask, ground_points, world2img] = Ortho::rotate_rectify(img_.size(), pose, intrinsic, img_);
    lock.unlock();
    this->ground_points = std::move(ground_points);
    this->world2img_    = std::move(world2img);
    this->img.set_rotate_rectified(rotate_img);
    this->img.set_rotate_rectified_mask(mask);
  }

  Pose           pose;
  Intrinsic      intrinsic;
  Image          img;
  Points<float>  ground_points;
  PointsPipeline world2img_;
  bool           reference_set = false;
};

struct ImgsData {
public:

  ImgsData() = default;

  ImgData& operator[](size_t i) { return imgs_data[i]; }

  size_t size() const { return imgs_data.size(); }

  void resize(size_t size) { imgs_data.resize(size); }

  void find_and_set_reference_coord() {
    std::vector<float> latitude, longitude, altitude;
    const size_t       size         = imgs_data.size();
    float              latitude_ref = 0.0f, longitude_ref = 0.0f, altitude_ref = std::numeric_limits<float>::min();
    for(auto&& data : imgs_data) {
      latitude_ref += data.get_latitude().degrees();
      longitude_ref += data.get_longitude().degrees();
      altitude_ref = std::max(altitude_ref, data.get_altitude());
    }
    latitude_ref /= size;
    longitude_ref /= size;
    for(auto&& data : imgs_data) {
      data.set_reference(latitude_ref, longitude_ref, altitude_ref);
    }
  }

  std::vector<ImgData>& get() { return imgs_data; }

private:

  std::vector<ImgData> imgs_data;
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
    Image  img(path, temp_save_path);
    auto&& res = IntrinsicFactory::validate_exif_xmp(img.exif_data(), img.xmp_data());
    if(res.has_value()) {
      std::cerr << res.value();
      return std::nullopt;
    }
    res = PoseFactory::validate_exif_xmp(img.exif_data(), img.xmp_data());
    if(res.has_value()) {
      std::cerr << res.value();
      return std::nullopt;
    }
    auto [img_, lock] = img.img().value();
    auto [w, h]       = img_.get().size();
    lock.unlock();
    Intrinsic intrinsic = IntrinsicFactory::build(img.exif_data(), w, h);
    Pose      pose      = PoseFactory::build(img.xmp_data());
    return ImgData{std::move(pose), std::move(intrinsic), std::move(img)};
    // return {{pose, intrinsic, img}};
  }
};
} // namespace Ortho

#endif