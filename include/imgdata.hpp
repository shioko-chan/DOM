#ifndef ORTHO_IMGDATA_HPP
#define ORTHO_IMGDATA_HPP

#include <algorithm>
#include <array>
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
#include "rotate_rectify.hpp"
#include "types.hpp"
#include "utility.hpp"

namespace Ortho {

struct Angle {
public:

  static constexpr float PI = 3.1415926535897932384626433832795;

  explicit Angle() {}

  explicit Angle(const float& degrees) : value(to_radians(degrees)) {}

  inline float radians() const { return value; }

  inline float degrees() const { return to_degrees(value); }

  inline void set_degrees(const float& degrees) { value = to_radians(degrees); }

  inline void set_radians(const float& radians) { value = radians; }

  friend std::ostream& operator<<(std::ostream& os, const Angle& prop) {
    os << prop.value << "(" << prop.radians() << "rad, " << prop.degrees() << "deg)";
    return os;
  }

private:

  inline static float to_degrees(float radians) { return radians * 180.0f / PI; }

  inline static float to_radians(float degrees) { return degrees * PI / 180.0f; }

  float value = 0.0f;
};

struct Kpnts {
public:

  size_t append(const Point<float>& kpnt) {
    auto it = kpnts_map.find(kpnt);
    if(it == kpnts_map.end()) {
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
    if(it == kpnts_map_rev.end()) {
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
  friend class ImgDataFactory;

public:

  ImgData() = default;

  ImgData(
      float    yaw_,
      float    pitch_,
      float    roll_,
      float    latitude_,
      float    longitude_,
      float    altitude_,
      float    focal_35mm_,
      fs::path img_path,
      fs::path temp_save_path) :
      latitude{latitude_}, longitude{longitude_}, altitude{altitude_}, focal_35mm{focal_35mm_}, img_path{img_path},
      temp_save_path{temp_save_path} {
    check_or_create_path(temp_save_path);
    Angle   yaw{yaw_}, pitch{pitch_}, roll{roll_};
    cv::Mat R_   = Rz(-yaw.radians()) * Ry(-pitch.radians()) * Rx(-roll.radians()) * Ry(-Angle::PI / 2);
    Q_proj_array = rotate2qarray(R_.t());
  }

  Image img() const {
    if(!rotated_rectified) {
      throw std::runtime_error("Not rectified yet!");
    }
    return img_rotated;
  }

  Image mask() const {
    if(!rotated_rectified) {
      throw std::runtime_error("Not rectified yet!");
    }
    return img_rotated_mask;
  }

  ImgRefGuard get_img() const {
    if(!rotated_rectified) {
      throw std::runtime_error("Not rectified yet!");
    }
    return img_rotated.get();
  }

  ImgRefGuard get_mask() const {
    if(!rotated_rectified) {
      throw std::runtime_error("Not rectified yet!");
    }
    return img_rotated_mask.get();
  }

  cv::Size get_size() const {
    if(!rotated_rectified) {
      throw std::runtime_error("Not rectified yet!");
    }
    return img_size;
  }

  void rotate_rectify() {
    if(!reference_set) {
      throw std::runtime_error("Reference coordinate not set!");
    }
    cv::Mat img = cv::imread(img_path.string());
    if(img.empty()) {
      throw std::runtime_error(img_path.string() + " could not be read");
    }
    const auto [w, h] = img.size();
    set_by_camera_params(w, h, focal_35mm);
    cv::Mat proj_mat          = get_projection_matrix(R_proj(), t_proj(), K_proj());
    auto&& [rotate_img, mask] = Ortho::rotate_rectify(&proj_mat, R_bproj(), img);
    cv::Mat c_m, r_m, t_m;
    cv::decomposeProjectionMatrix(proj_mat, c_m, r_m, t_m);
    set_by_K_proj(c_m);
    set_by_R_proj(r_m);
    t_m = r_m.t() * (t_m.rowRange(0, 3) / t_m.at<double>(3, 0));
    set_by_t_proj(t_m);
    img_size = rotate_img.size();
    this->img_rotated.delay_initialize(
        temp_save_path / std::format("{}_r{}", img_path.stem().string(), img_path.extension().string()),
        std::move(rotate_img));
    this->img_rotated_mask.delay_initialize(
        temp_save_path / std::format("{}_rm{}", img_path.stem().string(), img_path.extension().string()),
        std::move(mask));
    rotated_rectified = true;
  }

  fs::path get_img_path() const { return img_path; }

  fs::path get_img_name() const { return img_path.filename(); }

  fs::path get_img_stem() const { return img_path.stem(); }

  fs::path get_img_extension() const { return img_path.extension(); }

  cv::Mat K_proj() const { return array2camera(camera_array); }

  cv::Mat K_bproj() const { return array2camera(camera_array).inv(); }

  cv::Mat D() const { return array2distort(distort_array); }

  CameraArray& camera_array_raw() { return camera_array; }

  DistortArray& distort_array_raw() { return distort_array; }

  void set_by_camera_params(float w, float h, float focal_35mm) {
    float aspect_ratio = w * 1.0f / h;
    float ref_width    = (aspect_ratio >= 1.5) ? 36.0f : 24.0f * aspect_ratio;
    float f            = (ref_width == 36.0f) ? (w / 36.0f * focal_35mm) : (h / 24.0f * focal_35mm);
    camera_array       = {f, f, w / 2.0f, h / 2.0f};
  }

  void set_by_K_proj(cv::InputArray K) { camera_array = camera2array(K); }

  void set_by_K_bproj(cv::InputArray K) { camera_array = camera2array(K.getMat().inv()); }

  void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree) {
    const auto latitude_r = Angle(latitude_ref_degree), longitude_r = Angle(longitude_ref_degree);

    // WGS84
    const double a          = 6378137.0;
    const double f          = 1 / 298.257223563;
    const double e_sq       = 2 * f - f * f;
    const double sin_phi_sq = std::pow(std::sin(latitude_r.radians()), 2);
    const double M          = a * (1 - e_sq) / std::pow(1 - e_sq * sin_phi_sq, 1.5);
    const double N          = a / std::sqrt(1 - e_sq * sin_phi_sq);
    const float  x          = N * (longitude.radians() - longitude_r.radians()) * std::cos(latitude_r.radians());
    const float  y          = M * (latitude.radians() - latitude_r.radians());

    coord         = Point<float>(x, y);
    t_proj_array  = {-coord.x, -coord.y, altitude};
    reference_set = true;
  }

  cv::Mat R_proj() const { return qarray2rotate(Q_proj_array); }

  cv::Mat t_proj() const { return array2translate(t_proj_array); }

  cv::Mat R_bproj() const { return qarray2rotate(Q_proj_array).t(); }

  cv::Mat t_bproj() const { return -array2translate(t_proj_array); }

  RotateQArray& Q_proj_array_raw() { return Q_proj_array; }

  TranslateArray& t_proj_array_raw() { return t_proj_array; }

  void set_by_R_proj(cv::InputArray R) { Q_proj_array = rotate2qarray(R); }

  void set_by_R_bproj(cv::InputArray R) { Q_proj_array = rotate2qarray(R.getMat().t()); }

  void set_by_t_proj(cv::InputArray t) { t_proj_array = translate2array(t); }

  void set_by_t_bproj(cv::InputArray t) { t_proj_array = translate2array(-t.getMat()); }

  Kpnts& get_kpnts() { return kpnts; }

  const Point<float>& get_coord() { return coord; }

private:

  static cv::Mat Rx(float radians) {
    // clang-format off
    return (cv::Mat_<float>(3, 3) <<
      1, 0, 0,
      0, std::cos(radians), std::sin(radians),
      0, -std::sin(radians), std::cos(radians));
    // clang-format on
  }

  static cv::Mat Ry(float radians) {
    // clang-format off
    return (cv::Mat_<float>(3, 3) <<
     std::cos(radians), 0, -std::sin(radians),
     0, 1, 0,
     std::sin(radians), 0, std::cos(radians));
    // clang-format on
  }

  static cv::Mat Rz(float radians) {
    // clang-format off
    return (cv::Mat_<float>(3, 3) <<
    std::cos(radians), std::sin(radians), 0,
   -std::sin(radians), std::cos(radians), 0,
    0, 0, 1);
    // clang-format on
  }

  Kpnts kpnts;

  bool reference_set{false}, rotated_rectified{false};

  cv::Size img_size;
  fs::path temp_save_path, img_path;
  Image    img_rotated, img_rotated_mask;

  Angle        latitude, longitude;
  float        altitude;
  Point<float> coord;

  RotateQArray   Q_proj_array;
  TranslateArray t_proj_array;

  CameraArray  camera_array;
  DistortArray distort_array{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  float focal_35mm;
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
    for(auto&& data : imgs_data) {
      latitudes.push_back(data.latitude.degrees());
      longitudes.push_back(data.longitude.degrees());
    }
    size_t n = latitudes.size() / 2;
    std::nth_element(latitudes.begin(), latitudes.begin() + n, latitudes.end());
    std::nth_element(longitudes.begin(), longitudes.begin() + n, longitudes.end());
    float latitude_ref  = latitudes[n];
    float longitude_ref = longitudes[n];
    for(auto&& data : imgs_data) {
      data.set_reference(latitude_ref, longitude_ref);
    }
  }

private:

  std::vector<ImgData> imgs_data;
  std::mutex           mutex;
};

class ImgDataFactory {
private:

  static inline const std::unordered_set<std::string> img_extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

  struct ExifKey {
    static inline const std::string              focal_length_35mm = "Exif.Photo.FocalLengthIn35mmFilm";
    static inline const std::vector<std::string> keys              = {focal_length_35mm};
  };

  struct XmpKey {
    static inline const std::string yaw = "Xmp.drone-dji.GimbalYawDegree", pitch = "Xmp.drone-dji.GimbalPitchDegree",
                                    roll = "Xmp.drone-dji.GimbalRollDegree", latitude = "Xmp.drone-dji.GpsLatitude",
                                    longitude         = "Xmp.drone-dji.GpsLongitude",
                                    altitude          = "Xmp.drone-dji.AbsoluteAltitude";
    static inline const std::vector<std::string> keys = {yaw, pitch, roll, latitude, longitude, altitude};
  };

public:

  static bool validate(const fs::path& path) {
    if(!fs::is_regular_file(path) || img_extensions.count(path.extension().string()) == 0) {
      LOG_WARN("Error: {} is not a valid image file", path.string());
      return false;
    }
    ExifXmp               exif_xmp(path);
    const Exiv2::XmpData& xmp_data = exif_xmp.xmp_data();
    for(const auto& key : XmpKey::keys) {
      if(xmp_data.findKey(Exiv2::XmpKey(key)) == xmp_data.end()) {
        LOG_WARN("{}: Key {} not found in XMP data", exif_xmp.get_img_path().string(), key);
        return false;
      }
    }
    const auto& exif_data = exif_xmp.exif_data();
    for(const auto& key : ExifKey::keys) {
      if(exif_data.findKey(Exiv2::ExifKey(key)) == exif_data.end()) {
        LOG_WARN("{}: Key {} not found in Exif data", exif_xmp.get_img_path().string(), key);
        return false;
      }
    }
    return true;
  }

  static ImgData build(const fs::path& path, fs::path temp_save_path) {
    ExifXmp exif_xmp(path);
    auto&   xmp_data = exif_xmp.xmp_data();

    auto& exif_data = exif_xmp.exif_data();

    return ImgData{
        xmp_data[XmpKey::yaw].toFloat(),
        xmp_data[XmpKey::pitch].toFloat(),
        xmp_data[XmpKey::roll].toFloat(),
        xmp_data[XmpKey::latitude].toFloat(),
        xmp_data[XmpKey::longitude].toFloat(),
        xmp_data[XmpKey::altitude].toFloat(),
        exif_data[ExifKey::focal_length_35mm].toFloat(),
        path,
        temp_save_path};
  }
};
} // namespace Ortho

#endif
