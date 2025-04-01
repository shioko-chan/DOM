#ifndef POSE_INTRINSIC_HPP
#define POSE_INTRINSIC_HPP

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <ranges>
#include <unordered_map>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "angle.hpp"
#include "static.h"

using std::unordered_map;

namespace fs    = std::filesystem;
namespace views = std::views;

namespace Ortho {
struct Pose {
public:

  Angle       yaw, pitch, roll, latitude, longitude;
  float       altitude, altitude_ref;
  cv::Point2f coord;

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
    cv::Mat m1_ = R_z * R_y * R_x * Ry(Angle::PI / 2);
    m1_.copyTo(R_);
    cv::Mat m2_ = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, -altitude);
    m2_.copyTo(t_);
    cv::hconcat(R_, t_, T_);
  }

  void set_reference(const float& latitude_ref_degree, const float& longitude_ref_degree, const float& altitude_ref_) {
    altitude_ref           = altitude_ref_;
    const auto  latitude_r = Angle(latitude_ref_degree), longitude_r = Angle(longitude_ref_degree);
    const float x = 6371000 * (longitude.radians() - longitude_r.radians()) * std::cos(latitude_r.radians()),
                y = 6371000 * (latitude.radians() - latitude_r.radians());
    coord         = cv::Point2f(x, y);
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

  const cv::Mat& t() const { return t_; }

  const cv::Mat& T() const { return T_; }

  friend ostream& operator<<(ostream& os, const Pose& pose) {
    os << "Yaw: " << pose.yaw << "\n"
       << "Pitch: " << pose.pitch << "\n"
       << "Roll: " << pose.roll << "\n"
       << "Latitude: " << pose.latitude << "\n"
       << "Longitude: " << pose.longitude << "\n"
       << "Altitude: " << pose.altitude << "\n"
       << "R: " << pose.R_ << "\n"
       << "t: " << pose.t_ << "\n";
    return os;
  }

private:

  cv::Mat R_, t_, T_;
};

struct Intrinsic {
public:

  Intrinsic() = default;

  explicit Intrinsic(const float& w, const float& h, const float& focal, const float& sensor_width = 13.2f) :
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

  friend ostream& operator<<(ostream& os, const Intrinsic& intrinsic) {
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

  static std::optional<std::string> validate_xmp(const Exiv2::XmpData& data) {
    for(const auto& key : XmpKey::keys) {
      if(data.findKey(Exiv2::XmpKey(key)) == data.end()) {
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

  static unordered_map<std::string, float> build_sensor_width_database() {
    fs::path sensor_width_database_path(SENSOR_WIDTH_DATABASE);

    unordered_map<std::string, float> sensor_width_database;
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

  static inline const unordered_map<std::string, float> sensor_width_database = build_sensor_width_database();

  struct ExifKey {
    static inline const std::string              make         = "Exif.Image.Make";
    static inline const std::string              model        = "Exif.Image.Model";
    static inline const std::string              focal_length = "Exif.Photo.FocalLength";
    static inline const std::string              width        = "Exif.Image.ImageWidth";
    static inline const std::string              height       = "Exif.Image.ImageLength";
    static inline const std::vector<std::string> keys         = {make, model, focal_length, width, height};
  };

public:

  static std::optional<std::string> validate_exif(const Exiv2::ExifData& data) {
    for(const auto& key : ExifKey::keys) {
      if(data.findKey(Exiv2::ExifKey(key)) == data.end()) {
        std::stringstream ss;
        ss << "Error: Key " << key << " not found\n";
        return ss.str();
      }
    }
    return std::nullopt;
  }

  static Intrinsic build(Exiv2::ExifData& exif) {
    const float &w = exif[ExifKey::width].toFloat(), &h = exif[ExifKey::height].toFloat(),
                &focal = exif[ExifKey::focal_length].toFloat();
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
#endif