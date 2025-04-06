#ifndef POSE_INTRINSIC_HPP
#define POSE_INTRINSIC_HPP

#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <optional>
#include <ranges>
#include <unordered_map>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "static.h"
#include "utility.hpp"

namespace fs    = std::filesystem;
namespace views = std::views;

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

struct Pose {
public:

  Angle        yaw, pitch, roll, latitude, longitude;
  float        altitude, altitude_ref;
  Point<float> coord;

  Pose() = default;

  explicit Pose(
      const float& yaw_,
      const float& pitch_,
      const float& roll_,
      const float& latitude_,
      const float& longitude_,
      const float& altitude_) :
      yaw(yaw_), pitch(pitch_), roll(roll_), latitude(latitude_), longitude(longitude_), altitude(altitude_) {
    cv::Mat R_z = Rz(yaw.radians());
    cv::Mat R_y = Ry(pitch.radians());
    cv::Mat R_x = Rx(roll.radians());
    R_          = R_z * R_y * R_x * Ry(Angle::PI / 2);
  }

  // void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree, const float& altitude_ref_) {
  //   altitude_ref           = altitude_ref_;
  //   const auto  latitude_r = Angle(latitude_ref_degree), longitude_r = Angle(longitude_ref_degree);
  //   const float x = 6371000 * (longitude.radians() - longitude_r.radians()) * std::cos(latitude_r.radians()),
  //               y = 6371000 * (latitude.radians() - latitude_r.radians());
  //   coord         = Point<float>(x, y);
  // }

  void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree, const float& altitude_ref_) {
    altitude_ref          = altitude_ref_;
    const auto latitude_r = Angle(latitude_ref_degree), longitude_r = Angle(longitude_ref_degree);
    // WGS84
    const double a = 6378137.0;
    const double f = 1 / 298.257223563;
    const double b = a * (1 - f);
    double       M = a * (1 - f) / std::pow(1 - f * std::pow(std::sin(latitude_r.radians()), 2), 1.5);
    double       N = a / std::sqrt(1 - f * std::pow(std::sin(latitude_r.radians()), 2));
    const float  x = N * (longitude.radians() - longitude_r.radians()) * std::cos(latitude_r.radians());
    const float  y = M * (latitude.radians() - latitude_r.radians());
    coord          = Point<float>(x, y);
  }

  static cv::Mat Rx(float radians) {
    // clang-format off
        return (cv::Mat_<float>(3, 3) << 
          1, 0                ,  0                ,
          0, std::cos(radians), -std::sin(radians),
          0, std::sin(radians),  std::cos(radians));
    // clang-format on
  }

  static cv::Mat Ry(float radians) {
    // clang-format off
        return (cv::Mat_<float>(3, 3) << 
         std::cos(radians), 0, std::sin(radians),
         0                , 1, 0                ,
        -std::sin(radians), 0, std::cos(radians));
    // clang-format on
  }

  static cv::Mat Rz(float radians) {
    // clang-format off
        return (cv::Mat_<float>(3, 3) << 
        std::cos(radians), -std::sin(radians), 0,
        std::sin(radians),  std::cos(radians), 0,
        0                ,  0                , 1);
    // clang-format on
  }

  const cv::Mat& R() const { return R_; }

  friend std::ostream& operator<<(std::ostream& os, const Pose& pose) {
    os << "Yaw: " << pose.yaw << "\n"
       << "Pitch: " << pose.pitch << "\n"
       << "Roll: " << pose.roll << "\n"
       << "Latitude: " << pose.latitude << "\n"
       << "Longitude: " << pose.longitude << "\n"
       << "Altitude: " << pose.altitude << "\n"
       << "R: " << pose.R_ << "\n";
    return os;
  }

private:

  cv::Mat R_;
};

struct Intrinsic {
public:

  Intrinsic() = default;

  explicit Intrinsic(const float w, const float h, const float focal, const float sensor_width = 13.2f) :
      distortion_coefficients((cv::Mat_<float>::zeros(1, 5))) {
    float pix_f = std::max(w, h) * focal / sensor_width;
    // clang-format off
        cv::Mat m_ = (cv::Mat_<float>(3, 3) << 
          pix_f,      0,  w / 2,
              0,  pix_f,  h / 2,
              0,      0,      1
        );
    // clang-format on
    m_.copyTo(camera_matrix);
  }

  const cv::Mat& K() const { return camera_matrix; }

  const cv::Mat& D() const { return distortion_coefficients; }

  friend std::ostream& operator<<(std::ostream& os, const Intrinsic& intrinsic) {
    os << "Camera Matrix: " << intrinsic.camera_matrix << "\n"
       << "Distortion Coefficients: " << intrinsic.distortion_coefficients << "\n";
    return os;
  }

private:

  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
};

class PoseFactory {
private:

  struct XmpKey {
    static inline const std::string yaw = "Xmp.drone-dji.GimbalYawDegree", pitch = "Xmp.drone-dji.GimbalPitchDegree",
                                    roll = "Xmp.drone-dji.GimbalRollDegree", latitude = "Xmp.drone-dji.GpsLatitude",
                                    longitude         = "Xmp.drone-dji.GpsLongitude",
                                    altitude          = "Xmp.drone-dji.RelativeAltitude";
    static inline const std::vector<std::string> keys = {yaw, pitch, roll, latitude, longitude, altitude};
  };

public:

  static std::optional<std::string> validate_exif_xmp(const Exiv2::ExifData& exif_data, const Exiv2::XmpData& xmp_data) {
    for(const auto& key : XmpKey::keys) {
      if(xmp_data.findKey(Exiv2::XmpKey(key)) == xmp_data.end()) {
        std::stringstream ss;
        ss << "Error: Key " << key << " not found\n";
        return ss.str();
      }
    }
    return std::nullopt;
  }

  static Pose build(Exiv2::XmpData& xmp) {
    const float& yaw       = xmp[XmpKey::yaw].toFloat();
    const float& pitch     = xmp[XmpKey::pitch].toFloat();
    const float& roll      = xmp[XmpKey::roll].toFloat();
    const float& latitude  = xmp[XmpKey::latitude].toFloat();
    const float& longitude = xmp[XmpKey::longitude].toFloat();
    const float& altitude  = xmp[XmpKey::altitude].toFloat();
    return Pose(yaw, pitch, roll, latitude, longitude, altitude);
  }
};

class IntrinsicFactory {
private:

  static std::unordered_map<std::string, float> build_sensor_width_database() {
    fs::path sensor_width_database_path(SENSOR_WIDTH_DATABASE);

    std::unordered_map<std::string, float> sensor_width_database;
    if(!fs::exists(sensor_width_database_path)) {
      throw std::runtime_error("Error: Sensor width database not found");
    }
    std::ifstream ifs(sensor_width_database_path);
    if(!ifs.is_open()) {
      throw std::runtime_error("Error: Failed to open sensor width database");
    }
    std::string line;
    while(std::getline(ifs, line)) {
      auto v = line | views::split(';')
               | views::transform([](auto token_range) { return std::string(token_range.begin(), token_range.end()); });
      std::vector<std::string> tokens(v.begin(), v.end());
      if(tokens.size() != 2) {
        continue;
      }
      sensor_width_database.emplace(tokens[0], std::stod(tokens[1]));
    }
    return sensor_width_database;
  }

  static inline const std::unordered_map<std::string, float> sensor_width_database = build_sensor_width_database();

  struct ExifKey {
    static inline const std::string              make         = "Exif.Image.Make";
    static inline const std::string              model        = "Exif.Image.Model";
    static inline const std::string              focal_length = "Exif.Photo.FocalLength";
    static inline const std::vector<std::string> keys         = {make, model, focal_length};
  };

public:

  static std::optional<std::string> validate_exif_xmp(const Exiv2::ExifData& exif_data, const Exiv2::XmpData& xmp_data) {
    for(const auto& key : ExifKey::keys) {
      if(exif_data.findKey(Exiv2::ExifKey(key)) == exif_data.end()) {
        std::stringstream ss;
        ss << "Error: Key " << key << " not found\n";
        return ss.str();
      }
    }
    return std::nullopt;
  }

  static Intrinsic build(Exiv2::ExifData& exif, const float w, const float h) {
    const float       focal = exif[ExifKey::focal_length].toFloat();
    std::stringstream sensor_name;
    sensor_name << exif[ExifKey::make].toString() << " " << exif[ExifKey::model].toString();
    std::string sensor = sensor_name.str();
    if(sensor_width_database.count(sensor) == 0) {
      std::cerr << "Error: Sensor width not found in the database. Sensor name is " << sensor
                << "please add it to sensor_width_camera_database.txt at static directory\n";
      return Intrinsic(w, h, focal);
    }
    return Intrinsic(w, h, focal, sensor_width_database.at(sensor));
  }
};
} // namespace Ortho

namespace std {

template <>
struct formatter<cv::Mat> : formatter<string> {
  template <typename FormatContext>
  auto format(const cv::Mat& mat, FormatContext& ctx) {
    std::stringstream ss;
    ss << mat;
    return format_to(ctx.out(), "{}", ss.str());
  }
};

template <>
struct formatter<Ortho::Angle> : formatter<string> {
  template <typename FormatContext>
  auto format(const Ortho::Angle& angle, FormatContext& ctx) {
    return format_to(ctx.out(), "({} rad, {} degree)", angle.radians(), angle.degrees());
  }
};

template <>
struct formatter<Ortho::Pose> : formatter<string> {
  template <typename FormatContext>
  auto format(const Ortho::Pose& angle, FormatContext& ctx) {
    return format_to(
        ctx.out(),
        "Yaw: {}, Pitch: {}, Roll: {}, Latitude: {}, Longitude: {}, Altitude: {}\nR: {}",
        angle.yaw,
        angle.pitch,
        angle.roll,
        angle.latitude,
        angle.longitude,
        angle.altitude,
        angle.R());
  }
};

template <>
struct formatter<Ortho::Intrinsic> : formatter<string> {
  template <typename FormatContext>
  auto format(const Ortho::Intrinsic& intrinsic, FormatContext& ctx) {
    return format_to(ctx.out(), "Camera Matrix: {}\n Distortion Coefficients: {}", intrinsic.K(), intrinsic.D());
  }
};

} // namespace std
#endif