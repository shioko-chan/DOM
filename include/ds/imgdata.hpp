#ifndef ORTHO_IMGDATA_HPP
#define ORTHO_IMGDATA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <numbers>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <exiv2/exif.hpp>
#include <exiv2/exiv2.hpp>
#include <exiv2/tags.hpp>
#include <opencv2/opencv.hpp>

#include "algo/rotate_rectify.hpp"
#include "ds/image.hpp"
#include "tools/log.hpp"
#include "tools/report_error.hpp"
#include "tools/utility.hpp"

namespace Ortho {

namespace fs = std::filesystem;

struct Angle {
public:

  explicit Angle() = default;

  explicit Angle(const double& degrees) : value(to_radians(degrees)) {}

  [[nodiscard]] auto radians() const noexcept -> double { return value; }

  [[nodiscard]] auto degrees() const noexcept -> double { return to_degrees(value); }

  void set_degrees(const double& degrees) noexcept { value = to_radians(degrees); }

  void set_radians(const double& radians) noexcept { value = radians; }

  friend auto operator<<(std::ostream& ostream, const Angle& prop) noexcept -> std::ostream& {
    ostream << prop.value << "(" << prop.radians() << "rad, " << prop.degrees() << "deg)";
    return ostream;
  }

private:

  static auto to_degrees(double radians) noexcept -> double { return radians * 180. / std::numbers::pi; }

  static auto to_radians(double degrees) noexcept -> double { return degrees * std::numbers::pi / 180.; }

  double value = 0.;
};

struct alignas(128) Kpnts {
public:

  void set_perspective_matrix(cv::InputArray pers_mat_input) noexcept { this->pers_mat = pers_mat_input.getMat(); }

  auto append(const Point<double>& kpnt) noexcept -> size_t {
    auto kpnts_map_iter = kpnts_map.find(kpnt);
    if(kpnts_map_iter == kpnts_map.end()) {
      size_t idx    = kpnts_map.size();
      auto   origin = mat2point(pers_mat * kpnt);
      kpnts_map.emplace(origin, idx);
      kpnts_map_rev.emplace(idx, origin);
      return idx;
    }
    return kpnts_map_iter->second;
  }

  template <std::ranges::range R>
  auto append(const R& kpnts) noexcept {
    return kpnts | std::views::transform([this](const auto& kpnt) noexcept { return append(kpnt); });
  }

  auto get(size_t idx) const noexcept -> Point<double> {
    auto kpnts_map_rev_iter = kpnts_map_rev.find(idx);
    if(kpnts_map_rev_iter == kpnts_map_rev.end()) {
      report_error("Keypoint not found");
    }
    return kpnts_map_rev_iter->second;
  }

  auto size() noexcept -> size_t { return kpnts_map.size(); }

private:

  cv::Mat                      pers_mat;
  PointUMap<double, size_t>    kpnts_map;
  PointUMapRev<size_t, double> kpnts_map_rev;
};

class ImgData {
  friend class ImgsData;
  friend class ImgDataFactory;

public:

  ImgData() = default;

  ImgData(
      double          yaw_,
      double          pitch_,
      double          roll_,
      double          latitude_,
      double          longitude_,
      double          altitude_,
      double          focal_35mm_,
      fs::path        img_path,
      const fs::path& temp_save_path) noexcept :
      latitude{latitude_}, longitude{longitude_}, altitude{altitude_}, focal_35mm{focal_35mm_},
      temp_save_path{temp_save_path}, img_origin{std::move(img_path)} {
    check_or_create_path(temp_save_path);
    Angle   yaw{yaw_};
    Angle   pitch{pitch_};
    Angle   roll{roll_};
    cv::Mat R_mat = Rz(-yaw.radians()) * Ry(-pitch.radians()) * Rx(-roll.radians()) * Ry(-std::numbers::pi / 2);
    Q_proj_array  = rotate2qarray(R_mat.t());
  }

  auto origin_img() const noexcept -> const OriginImage& { return img_origin; }

  auto origin_img() noexcept -> OriginImage& { return img_origin; }

  auto rotated_img() const noexcept -> const Image& {
    if(!rotated_rectified) {
      report_error("Not rectified yet!");
    }
    return img_rotated;
  }

  auto rotated_img() noexcept -> Image& {
    if(!rotated_rectified) {
      report_error("Not rectified yet!");
    }
    return img_rotated;
  }

  void rotate_rectify() noexcept {
    if(!reference_set) {
      report_error("Reference coordinate not set!");
    }
    auto img                   = img_origin.get();
    const auto [width, height] = img.size();
    set_by_camera_params(width, height, focal_35mm);
    auto [rotate_img, pixel_span, pers_mat] = Ortho::rotate_rectify(R_bproj(), img);
    kpnts.set_perspective_matrix(pers_mat.inv());
    this->img_rotated.delay_initialize(
        temp_save_path
            / std::format("{}_r{}", img_origin.get_img_stem().string(), img_origin.get_img_extension().string()),
        std::move(rotate_img),
        pixel_span);
    rotated_rectified = true;
  }

  auto K_proj() const noexcept -> cv::Mat {
    return (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, 1) * array2camera(camera_array);
  }

  auto K_bproj() const noexcept -> cv::Mat {
    return array2camera(camera_array).inv() * (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, 1);
  }

  auto D() const noexcept -> cv::Mat { return array2distort(distort_array); }

  auto camera_array_raw() noexcept -> CameraArray& { return camera_array; }

  auto camera_array_raw() const noexcept -> const CameraArray& { return camera_array; }

  auto distort_array_raw() noexcept -> DistortArray& { return distort_array; }

  auto distort_array_raw() const noexcept -> const DistortArray& { return distort_array; }

  void set_by_camera_params(double width, double height, double focal_35mm) noexcept {
    double aspect_ratio = width * 1. / height;
    double ref_width    = (aspect_ratio >= 1.5) ? 36. : 24. * aspect_ratio;
    double focal_pix    = (ref_width == 36.) ? (width / 36. * focal_35mm) : (height / 24. * focal_35mm);
    camera_array        = {focal_pix, focal_pix, width / 2., height / 2.};
  }

  void set_reference(double latitude_ref_degree, double longitude_ref_degree) noexcept {
    const auto latitude_r  = Angle(latitude_ref_degree);
    const auto longitude_r = Angle(longitude_ref_degree);
    // WGS84
    const double semi_major_axis = 6378137.0;
    const double flattening      = 1.0 / 298.257223563;
    const double eccentricity_sq = (2.0 * flattening) - (flattening * flattening);
    const double sin_lat_ref_sq  = std::pow(std::sin(latitude_r.radians()), 2);
    const double meridional_radius =
        semi_major_axis * (1.0 - eccentricity_sq) / std::pow(1.0 - (eccentricity_sq * sin_lat_ref_sq), 1.5);
    const double prime_vert_radius   = semi_major_axis / std::sqrt(1.0 - (eccentricity_sq * sin_lat_ref_sq));
    const double delta_longitude_rad = longitude.radians() - longitude_r.radians();
    const double delta_latitude_rad  = latitude.radians() - latitude_r.radians();
    const double projected_x         = prime_vert_radius * delta_longitude_rad * std::cos(latitude_r.radians());
    const double projected_y         = meridional_radius * delta_latitude_rad;
    coord                            = Point<double>(projected_x, projected_y);
    cv::Mat t_c2w_mat                = (cv::Mat_<double>(3, 1) << coord.x, coord.y, -altitude);
    cv::Mat t_w2c_mat                = -R_proj() * t_c2w_mat;
    t_proj_array  = {t_w2c_mat.at<double>(0, 0), t_w2c_mat.at<double>(1, 0), t_w2c_mat.at<double>(2, 0)};
    reference_set = true;
  }

  auto R_proj() const noexcept -> cv::Mat { return qarray2rotate(Q_proj_array); }

  auto t_proj() const noexcept -> cv::Mat { return array2translate(t_proj_array); }

  auto R_bproj() const noexcept -> cv::Mat { return qarray2rotate(Q_proj_array).t(); }

  auto t_bproj() const noexcept -> cv::Mat { return -R_bproj() * array2translate(t_proj_array); }

  auto Q_proj_array_raw() noexcept -> RotateQArray& { return Q_proj_array; }

  auto Q_proj_array_raw() const noexcept -> const RotateQArray& { return Q_proj_array; }

  auto t_proj_array_raw() noexcept -> TranslateArray& { return t_proj_array; }

  auto t_proj_array_raw() const noexcept -> const TranslateArray& { return t_proj_array; }

  auto get_kpnts() const noexcept -> const Kpnts& { return kpnts; }

  auto get_kpnts() noexcept -> Kpnts& { return kpnts; }

  auto get_coord() noexcept -> const Point<double>& { return coord; }

private:

  static auto Rx(double radians) noexcept -> cv::Mat {
    // clang-format off
    cv::Mat R_mat = 
    (cv::Mat_<double>(3, 3) <<
      1, 0, 0,
      0, std::cos(radians), std::sin(radians),
      0, -std::sin(radians), std::cos(radians));
    // clang-format on
    return R_mat;
  }

  static auto Ry(double radians) noexcept -> cv::Mat {
    // clang-format off
    cv::Mat R_mat = 
    (cv::Mat_<double>(3, 3) <<
      std::cos(radians), 0, -std::sin(radians),
      0, 1, 0,
      std::sin(radians), 0, std::cos(radians));
    // clang-format on
    return R_mat;
  }

  static auto Rz(double radians) noexcept -> cv::Mat {
    // clang-format off
    cv::Mat R_mat = 
    (cv::Mat_<double>(3, 3) <<
      std::cos(radians), std::sin(radians), 0,
    -std::sin(radians), std::cos(radians), 0,
      0, 0, 1);
    // clang-format on
    return R_mat;
  }

  Kpnts kpnts;

  bool reference_set{false}, rotated_rectified{false};

  fs::path    temp_save_path;
  Image       img_rotated;
  OriginImage img_origin;

  Angle         latitude, longitude;
  double        altitude{};
  Point<double> coord;

  RotateQArray   Q_proj_array{};
  TranslateArray t_proj_array{};

  CameraArray  camera_array{};
  DistortArray distort_array{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  double focal_35mm{};
};

class ImgsData {
public:

  ImgsData() noexcept = default;

  ImgsData(std::initializer_list<ImgData> init) noexcept : imgs_data(init) {}

  template <std::input_iterator I>
  ImgsData(I first, I last) noexcept : imgs_data(first, last) {}

  [[nodiscard]] auto operator[](size_t idx) noexcept -> ImgData& { return imgs_data[idx]; }

  [[nodiscard]] auto operator[](size_t idx) const noexcept -> const ImgData& { return imgs_data[idx]; }

  [[nodiscard]] auto get() noexcept -> std::vector<ImgData>& { return imgs_data; }

  [[nodiscard]] auto get() const noexcept -> const std::vector<ImgData>& { return imgs_data; }

  [[nodiscard]] auto size() const noexcept -> size_t { return imgs_data.size(); }

  [[nodiscard]] auto empty() const noexcept -> bool { return imgs_data.empty(); }

  void resize(size_t size) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.resize(size);
  }

  void clear() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.clear();
  }

  void reserve(size_t size) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.reserve(size);
  }

  [[nodiscard]] auto begin() noexcept { return imgs_data.begin(); }

  [[nodiscard]] auto end() noexcept { return imgs_data.end(); }

  [[nodiscard]] auto begin() const noexcept { return imgs_data.begin(); }

  [[nodiscard]] auto end() const noexcept { return imgs_data.end(); }

  [[nodiscard]] auto cbegin() const noexcept { return imgs_data.cbegin(); }

  [[nodiscard]] auto cend() const noexcept { return imgs_data.cend(); }

  [[nodiscard]] auto rbegin() noexcept { return imgs_data.rbegin(); }

  [[nodiscard]] auto rend() noexcept { return imgs_data.rend(); }

  [[nodiscard]] auto rbegin() const noexcept { return imgs_data.rbegin(); }

  [[nodiscard]] auto rend() const noexcept { return imgs_data.rend(); }

  [[nodiscard]] auto crbegin() const noexcept { return imgs_data.crbegin(); }

  [[nodiscard]] auto crend() const noexcept { return imgs_data.crend(); }

  template <typename T>
    requires std::same_as<std::decay_t<T>, ImgData>
  void push_back(T&& data) noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.push_back(std::forward<T>(data));
  }

  void pop_back() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    imgs_data.pop_back();
  }

  void find_and_set_reference_coord() noexcept {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<double>         latitudes;
    std::vector<double>         longitudes;
    for(auto&& data : imgs_data) {
      latitudes.push_back(data.latitude.degrees());
      longitudes.push_back(data.longitude.degrees());
    }
    auto nth = static_cast<int64_t>(latitudes.size()) / 2;
    std::nth_element(latitudes.begin(), latitudes.begin() + nth, latitudes.end());
    std::nth_element(longitudes.begin(), longitudes.begin() + nth, longitudes.end());
    double latitude_ref  = latitudes[nth];
    double longitude_ref = longitudes[nth];
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

  static constexpr std::array<std::string_view, 10> img_extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

  struct ExifKey {
    static constexpr std::string_view focal_length_35mm{"Exif.Photo.FocalLengthIn35mmFilm"};
  };

  struct XmpKey {
    static constexpr std::string_view yaw = "Xmp.drone-dji.GimbalYawDegree", pitch = "Xmp.drone-dji.GimbalPitchDegree",
                                      roll = "Xmp.drone-dji.GimbalRollDegree", latitude = "Xmp.drone-dji.GpsLatitude",
                                      longitude = "Xmp.drone-dji.GpsLongitude",
                                      altitude  = "Xmp.drone-dji.AbsoluteAltitude";
  };

public:

  static auto validate(const fs::path& path) noexcept -> bool {
    if(!fs::is_regular_file(path)
       || std::ranges::find(img_extensions, path.extension().string()) == img_extensions.end()) {
      THIS_LOG_WARN("Error: {} is not a valid image file", path.string());
      return false;
    }
    ExifXmp               exif_xmp(path);
    const Exiv2::XmpData& xmp_data = exif_xmp.xmp_data();
    auto validate_xmp = [&xmp_data, path = exif_xmp.get_img_path().string()](std::string_view key_) noexcept {
      std::string key{key_.data(), key_.size()};
      if(xmp_data.findKey(Exiv2::XmpKey(key)) == xmp_data.end()) {
        THIS_LOG_WARN("{}: Key {} not found in XMP data", path, key);
        return false;
      }
      return true;
    };
    const Exiv2::ExifData& exif_data = exif_xmp.exif_data();
    auto validate_exif = [&exif_data, path = exif_xmp.get_img_path().string()](std::string_view key_) noexcept {
      std::string key{key_.data(), key_.size()};
      if(exif_data.findKey(Exiv2::ExifKey(key)) == exif_data.end()) {
        THIS_LOG_WARN("{}: Key {} not found in XMP data", path, key);
        return false;
      }
      return true;
    };
    return validate_xmp(XmpKey::yaw) && validate_xmp(XmpKey::pitch) && validate_xmp(XmpKey::roll)
           && validate_xmp(XmpKey::latitude) && validate_xmp(XmpKey::longitude) && validate_xmp(XmpKey::altitude)
           && validate_exif(ExifKey::focal_length_35mm);
  }

  static auto build(const fs::path& path, const fs::path& temp_save_path) noexcept -> ImgData {
    ExifXmp exif_xmp(path);
    auto&   xmp_data = exif_xmp.xmp_data();

    auto& exif_data = exif_xmp.exif_data();

    return ImgData{
        xmp_data[std::string{XmpKey::yaw}].toFloat(),
        xmp_data[std::string{XmpKey::pitch}].toFloat(),
        xmp_data[std::string{XmpKey::roll}].toFloat(),
        xmp_data[std::string{XmpKey::latitude}].toFloat(),
        xmp_data[std::string{XmpKey::longitude}].toFloat(),
        xmp_data[std::string{XmpKey::altitude}].toFloat(),
        exif_data[std::string{ExifKey::focal_length_35mm}].toFloat(),
        path,
        temp_save_path};
  }
};
} // namespace Ortho

#endif
