#ifndef SKYMERGE_IMGDATA_HPP
#define SKYMERGE_IMGDATA_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <limits>
#include <mutex>
#include <numbers>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <GeographicLib/Geocentric.hpp>
#include <GeographicLib/LocalCartesian.hpp>
#include <exiv2/exiv2.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "algo/rectify.hpp"
#include "config.hpp"
#include "ds/image.hpp"
#include "tools/debug.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"
#include "tools/utility.hpp"

namespace SkyMerge {

struct Angle {
public:

  explicit Angle() noexcept = default;

  explicit Angle(const double& degrees) noexcept : value(to_radians(degrees)) {}

  [[nodiscard]] auto radians() const noexcept -> double { return value; }

  [[nodiscard]] auto degrees() const noexcept -> double { return to_degrees(value); }

  void set_degrees(const double& degrees) noexcept { value = to_radians(degrees); }

  void set_radians(const double& radians) noexcept { value = radians; }

  friend auto operator<<(std::ostream& ostream, const Angle& prop) noexcept -> std::ostream& {
    ostream << prop.value << "(" << prop.radians() << "rad, " << prop.degrees() << "deg)";
    return ostream;
  }

private:

  static auto to_degrees(double radians) noexcept -> double { return radians * 180.0 / std::numbers::pi; }

  static auto to_radians(double degrees) noexcept -> double { return degrees * std::numbers::pi / 180.0; }

  double value = 0.;
};

class ImgData {
  friend class ImgsData;
  friend class ImgDataFactory;

public:

  ImgData() noexcept = default;

  ImgData(
      double                       yaw_,
      double                       pitch_,
      double                       roll_,
      double                       latitude_,
      double                       longitude_,
      double                       altitude_,
      std::filesystem::path        img_path,
      const std::filesystem::path& temp_save_path) noexcept :
      latitude{latitude_}, longitude{longitude_}, altitude{altitude_}, temp_save_path{temp_save_path},
      img_origin{std::move(img_path)} {
    check_or_create_path(temp_save_path);
    Angle   yaw{yaw_};
    Angle   pitch{(pitch_ + 90.0)}; // DJI to nadir
    Angle   roll{roll_};
    cv::Mat R_v_w2c =
        z_rotate_matrix(yaw.radians()) * y_rotate_matrix(pitch.radians()) * x_rotate_matrix(roll.radians());
    A_w2c_array = rotate2axisangle(R_v_w2c.t());
  }

  [[nodiscard]] auto is_valid() const noexcept -> bool { return valid; }

  void set_invalid() noexcept { valid = false; }

  void set_valid() noexcept { valid = true; }

  [[nodiscard]] auto origin_img() const noexcept -> const OriginImage& { return img_origin; }

  [[nodiscard]] auto origin_img() noexcept -> OriginImage& { return img_origin; }

  [[nodiscard]] auto rotated_img() const noexcept -> const Image& {
    THIS_ASSERTION_SHOULD_TRUE(rotated_rectified, "Not rectified yet!");
    return img_rotated;
  }

  [[nodiscard]] auto rotated_img() noexcept -> Image& {
    THIS_ASSERTION_SHOULD_TRUE(rotated_rectified, "Not rectified yet!");
    return img_rotated;
  }

  void rotate_rectify() noexcept {
    THIS_ASSERTION_SHOULD_TRUE(reference_set, "Reference coordinate not set!");
    auto img                                = img_origin.get();
    const auto [width, height]              = img.size();
    auto [rotate_img, pixel_span, pers_mat] = SkyMerge::rotate_rectify(R_c2w(), img);
    perspective_mat                         = pers_mat.inv();
    double scale                            = decimate_keep_aspect_ratio(&rotate_img, FEATURE_EXTRACTOR_RESOLUTION_LIM);
    std::ranges::for_each(pixel_span, [scale](Point<double>& point) noexcept {
      point.x *= scale;
      point.y *= scale;
    });
    this->img_rotated.delay_initialize(
        temp_save_path
            / std::format("{}_r{}", img_origin.get_img_stem().string(), img_origin.get_img_extension().string()),
        std::move(rotate_img),
        pixel_span);
    rotated_rectified = true;
  }

  void set_reference(double latitude_ref_degree, double longitude_ref_degree) noexcept {
    GeographicLib::LocalCartesian
           local_cartesian{latitude_ref_degree, longitude_ref_degree, 0, GeographicLib::Geocentric::WGS84()};
    double east{};
    double north{};
    double upward{};
    local_cartesian.Forward(latitude, longitude, altitude, east, north, upward);
    auto    coord     = Point<double>{north, east};
    cv::Mat t_c2w_mat = (cv::Mat_<double>(3, 1) << coord.x, coord.y, -upward);
    cv::Mat t_w2c_mat = -R_w2c() * t_c2w_mat;
    t_w2c_array       = {t_w2c_mat.at<double>(0, 0), t_w2c_mat.at<double>(1, 0), t_w2c_mat.at<double>(2, 0)};
    reference_set     = true;
  }

  [[nodiscard]] auto R_w2c() const noexcept -> cv::Mat {
    THIS_ASSERTION_SHOULD_FALSE(std::isnan(A_w2c_array[0]), "R not initialized yet!");
    return axisangle2rotate(A_w2c_array);
  }

  [[nodiscard]] auto R_c2w() const noexcept -> cv::Mat {
    THIS_ASSERTION_SHOULD_FALSE(std::isnan(A_w2c_array[0]), "R not initialized yet!");
    return axisangle2rotate(A_w2c_array).t();
  }

  [[nodiscard]] auto A_w2c_array_raw() noexcept -> RotateAxisAngle& { return A_w2c_array; }

  [[nodiscard]] auto A_w2c_array_raw() const noexcept -> const RotateAxisAngle& {
    THIS_ASSERTION_SHOULD_FALSE(std::isnan(A_w2c_array[0]), "R not initialized yet!");
    return A_w2c_array;
  }

  [[nodiscard]] auto t_w2c() const noexcept -> cv::Mat {
    THIS_ASSERTION_SHOULD_FALSE(std::isnan(t_w2c_array[0]), "t not initialized yet!");
    return array2translate(t_w2c_array);
  }

  [[nodiscard]] auto t_c2w() const noexcept -> cv::Mat {
    THIS_ASSERTION_SHOULD_FALSE(std::isnan(t_w2c_array[0]), "t not initialized yet!");
    return -R_c2w() * array2translate(t_w2c_array);
  }

  [[nodiscard]] auto t_w2c_array_raw() noexcept -> TranslateArray& { return t_w2c_array; }

  [[nodiscard]] auto t_w2c_array_raw() const noexcept -> const TranslateArray& {
    THIS_ASSERTION_SHOULD_FALSE(std::isnan(t_w2c_array[0]), "t not initialized yet!");
    return t_w2c_array;
  }

  [[nodiscard]] auto get_kpnts() const noexcept -> const Points<double>& { return kpnts; }

  void set_kpnts(const Points<double>& points) noexcept { cv::perspectiveTransform(points, kpnts, perspective_mat); }

  [[nodiscard]] auto get_coord() const noexcept -> Point<double> {
    THIS_ASSERTION_SHOULD_TRUE(reference_set, "reference not set!");
    cv::Mat t_w2c_mat = (cv::Mat_<double>(3, 1) << t_w2c_array[0], t_w2c_array[1], t_w2c_array[2]);
    cv::Mat t_c2w_mat = -R_c2w() * t_w2c_mat;
    return {t_c2w_mat.at<double>(0, 0), t_c2w_mat.at<double>(1, 0)};
  }

private:

  bool valid{true};

  Points<double> kpnts;

  cv::Mat perspective_mat;

  bool reference_set{false}, rotated_rectified{false};

  std::filesystem::path temp_save_path;
  Image                 img_rotated;
  OriginImage           img_origin;

  double latitude{}, longitude{}, altitude{};

  RotateAxisAngle A_w2c_array{std::numeric_limits<double>::quiet_NaN()};
  TranslateArray  t_w2c_array{std::numeric_limits<double>::quiet_NaN()};
};

class ImgDataFactory {
private:

  static constexpr std::array<std::string_view, 10> img_extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

  struct XmpKey {
    static constexpr std::string_view yaw{"Xmp.drone-dji.GimbalYawDegree"}, pitch{"Xmp.drone-dji.GimbalPitchDegree"},
        roll{"Xmp.drone-dji.GimbalRollDegree"}, latitude{"Xmp.drone-dji.GpsLatitude"},
        longitude{"Xmp.drone-dji.GpsLongitude"}, altitude{"Xmp.drone-dji.RelativeAltitude"};
  };

public:

  static auto check_path(const std::filesystem::path& path) noexcept -> bool {
    if(!std::filesystem::is_regular_file(path)
       || std::ranges::find(img_extensions, path.extension().string()) == img_extensions.end()) {
      THIS_LOG_WARN("Error: {} is not a valid image file", path.string());
      return false;
    }
    return true;
  }

  static auto check_xmp(const Exiv2::XmpData& xmp_data) -> bool {
    auto validate_xmp = [&xmp_data](std::string_view key_) noexcept {
      std::string key{key_};
      if(xmp_data.findKey(Exiv2::XmpKey(key)) == xmp_data.end()) {
        THIS_LOG_WARN("Key {} not found in XMP data", key);
        return false;
      }
      return true;
    };
    return validate_xmp(XmpKey::yaw) && validate_xmp(XmpKey::pitch) && validate_xmp(XmpKey::roll)
           && validate_xmp(XmpKey::latitude) && validate_xmp(XmpKey::longitude) && validate_xmp(XmpKey::altitude);
  }

  static auto build(const std::filesystem::path& path, const std::filesystem::path& temp_save_path) noexcept -> ImgData {
    ExifXmp exif_xmp{path};
    auto&   xmp_data  = exif_xmp.xmp_data();
    auto&   exif_data = exif_xmp.exif_data();
    return ImgData{
        xmp_data[std::string{XmpKey::yaw}].toFloat(),
        xmp_data[std::string{XmpKey::pitch}].toFloat(),
        xmp_data[std::string{XmpKey::roll}].toFloat(),
        xmp_data[std::string{XmpKey::latitude}].toFloat(),
        xmp_data[std::string{XmpKey::longitude}].toFloat(),
        xmp_data[std::string{XmpKey::altitude}].toFloat(),
        path,
        temp_save_path};
  }
};

class ImgsData {
public:

  ImgsData() noexcept = delete;

  ImgsData(
      const std::vector<std::filesystem::path>& img_paths,
      const std::filesystem::path&              temporary_save_path,
      Progress&                                 progress) noexcept {
    if(img_paths.empty()) {
      THIS_LOG_WARN("No image input!");
      return;
    }
    std::vector<double> f_35mms(img_paths.size(), std::numeric_limits<double>::quiet_NaN());
    run(
        img_paths.size(),
        [this, &img_paths, &f_35mms, temporary_save_path](int idx) noexcept {
          const auto& img_path = img_paths[idx];
          if(!ImgDataFactory::check_path(img_path)) {
            return;
          }
          ExifXmp exif_xmp(img_path);
          if(!ImgDataFactory::check_xmp(exif_xmp.xmp_data())) {
            return;
          }
          auto&       exif_data = exif_xmp.exif_data();
          std::string key{ExifKey::focal_length_35mm};
          if(exif_data.findKey(Exiv2::ExifKey(key)) == exif_data.end()) {
            THIS_LOG_WARN("Key {} not found in Exif data", key);
            return;
          }
          f_35mms[idx] = exif_data[std::string{ExifKey::focal_length_35mm}].toFloat();
          push_back(ImgDataFactory::build(img_path, temporary_save_path));
        },
        progress);
    std::unordered_map<int, int> freq_map;
    const double                 approx = 1000.0;
    for(double focal : f_35mms) {
      if(std::isfinite(focal)) {
        freq_map[static_cast<int>(std::round(focal * approx))]++;
      }
    }
    int max_count = 0;
    for(const auto& [focal, count] : freq_map) {
      if(count > max_count) {
        max_count = count;
        f_35mm    = focal;
      }
    }
    f_35mm /= approx;
    if(max_count != imgs_data.size()) {
      THIS_LOG_WARN("Images are not taken from same camera!");
    }
    find_and_set_reference_coord();
  }

  [[nodiscard]] auto operator[](size_t idx) noexcept -> ImgData& { return imgs_data[idx]; }

  [[nodiscard]] auto operator[](size_t idx) const noexcept -> const ImgData& { return imgs_data[idx]; }

  [[nodiscard]] auto get() noexcept -> std::vector<ImgData>& { return imgs_data; }

  [[nodiscard]] auto get() const noexcept -> const std::vector<ImgData>& { return imgs_data; }

  [[nodiscard]] auto size() const noexcept -> size_t { return imgs_data.size(); }

  [[nodiscard]] auto empty() const noexcept -> bool { return imgs_data.empty(); }

  void resize(size_t size) noexcept {
    TempLock lock(mutex);
    imgs_data.resize(size);
  }

  void clear() noexcept {
    TempLock lock(mutex);
    imgs_data.clear();
  }

  void reserve(size_t size) noexcept {
    TempLock lock(mutex);
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
    TempLock lock(mutex);
    imgs_data.push_back(std::forward<T>(data));
  }

  void pop_back() noexcept {
    TempLock lock(mutex);
    imgs_data.pop_back();
  }

  [[nodiscard]] auto K() noexcept -> cv::Mat {
    if(std::isnan(camera_array[0])) {
      initialize_camera_param();
    }
    return array2camera(camera_array);
  }

  [[nodiscard]] auto M() noexcept -> cv::Mat {
    // clang-format off
    return K() * (cv::Mat_<double>(3, 3) << 
     0, 1, 0,
    -1, 0, 0, 
     0, 0, 1);
    // clang-format on
  }

  [[nodiscard]] auto D() const noexcept -> cv::Mat { return array2distort(distort_array); }

  [[nodiscard]] auto camera_array_raw() noexcept -> CameraArray& {
    if(std::isnan(camera_array[0])) {
      initialize_camera_param();
    }
    return camera_array;
  }

  [[nodiscard]] auto distort_array_raw() noexcept -> DistortArray& { return distort_array; }

  [[nodiscard]] auto distort_array_raw() const noexcept -> const DistortArray& { return distort_array; }

  void set_by_camera_params(double width, double height, double focal_35mm) noexcept {
    double aspect_ratio = width * 1.0 / height;
    double ref_width    = (aspect_ratio >= 1.5) ? 36.0 : 24.0 * aspect_ratio;
    double focal_pix    = (aspect_ratio >= 1.5) ? (width / 36.0 * focal_35mm) : (height / 24.0 * focal_35mm);
    camera_array        = {focal_pix, focal_pix, width / 2.0, height / 2.0};
  }

  void initialize_camera_param() {
    if(imgs_data.empty()) {
      THIS_LOG_WARN("No image input!");
      return;
    }
    std::map<std::pair<int, int>, int> freq_map;
    for(auto& img_data : imgs_data) {
      const auto& [w, h] = img_data.origin_img().get_size();
      freq_map[{w, h}]++;
    }
    int                 max_count = 0;
    std::pair<int, int> w_h{};
    for(const auto& [wh, count] : freq_map) {
      if(count > max_count) {
        max_count = count;
        w_h       = wh;
      }
    }
    if(max_count != imgs_data.size()) {
      THIS_LOG_WARN("Images are not in the same size!");
    }
    const auto& [w, h] = w_h;
    set_by_camera_params(w, h, f_35mm);
  }

  void find_and_set_reference_coord() noexcept {
    TempLock            lock(mutex);
    std::vector<double> latitudes;
    std::vector<double> longitudes;
    if(imgs_data.empty()) {
      return;
    }
    for(const auto& data : imgs_data) {
      latitudes.push_back(data.latitude);
      longitudes.push_back(data.longitude);
    }
    auto nth = static_cast<std::int64_t>(latitudes.size()) / 2;
    std::nth_element(latitudes.begin(), latitudes.begin() + nth, latitudes.end());
    std::nth_element(longitudes.begin(), longitudes.begin() + nth, longitudes.end());
    double latitude_ref  = latitudes[nth];
    double longitude_ref = longitudes[nth];
    for(auto& data : imgs_data) {
      data.set_reference(latitude_ref, longitude_ref);
    }
  }

private:

  struct ExifKey {
    static constexpr std::string_view focal_length_35mm{"Exif.Photo.FocalLengthIn35mmFilm"};
  };

  std::vector<ImgData> imgs_data;
  std::mutex           mutex;
  double               f_35mm{};
  CameraArray          camera_array{std::numeric_limits<double>::quiet_NaN()};
  DistortArray         distort_array{0.0, 0.0, 0.0, 0.0, 0.0};
};

} // namespace SkyMerge

#endif
