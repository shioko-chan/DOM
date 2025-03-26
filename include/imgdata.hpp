#ifndef IMGDATA_HPP
#define IMGDATA_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#ifndef SENSOR_WIDTH_DATABASE
  #define SENSOR_WIDTH_DATABASE "sensor_width_database.txt"
#endif

using std::cerr;
using std::cos;
using std::nullopt;
using std::optional;
using std::ostream;
using std::pair;
using std::sin;
using std::unique_ptr;
using std::unordered_map;
using std::unordered_set;

namespace fs     = std::filesystem;
namespace ranges = std::ranges;
namespace views  = std::views;

namespace Ortho {

void pipeline_terminate() { Exiv2::XmpParser::terminate(); }

template <typename T>
class Property {
private:

  T value;

public:

  explicit Property() : value(T()) {}

  template <typename U>
  explicit Property(U&& val) : value(std::forward<U>(val)) {}

  template <typename U>
  Property& operator=(U&& val) {
    set(std::forward<U>(val));
    return *this;
  }

  const T& get() const { return value; }

  T& get_mut() { return value; }

  template <typename U>
  void set(U&& val) {
    value = std::forward<U>(val);
  }

  friend ostream& operator<<(ostream& os, const Property& prop) {
    os << prop.get();
    return os;
  }
};

class Image : public Property<cv::Mat> {
public:

  template <typename U>
    requires std::same_as<std::decay_t<U>, cv::Mat>
  explicit Image(U&& img) : Property(std::forward<U>(img)) {}

  explicit Image() : Property() {}

  ~Image() {
    if(!get().empty()) {
      get_mut().release();
    }
  }

  void write(const fs::path& output_path) const { cv::imwrite(output_path.string(), get()); }

  void release() {
    if(!get().empty()) {
      get_mut().release();
    }
  }
};

class Angle : public Property<float> {
private:

  static float normalize(const float& degrees) {
    return degrees;
    // return std::fmod(std::fmod(degrees, 360.0f) + 360.0f, 360.0f);
  }

public:

  static constexpr float PI = 3.1415926535897932384626433832795;

  explicit Angle(const float& degrees) : Property(normalize(degrees)) {}

  float radians() const { return get() * PI / 180.0f; }

  float degrees() const { return get(); }

  void set_degrees(const float& degrees) { set(degrees); }

  void set_radians(const float& radians) { set(radians * 180.0f / PI); }
};

struct Pose {
private:

  cv::Mat R_, t_, T_;

public:

  Angle                 yaw, pitch, roll, latitude, longitude;
  Property<float>       altitude, altitude_ref;
  Property<cv::Point2f> coord;

  void set_reference(const float& latitude_ref, const float& longitude_ref, const float& altitude_ref_) {
    altitude_ref.set(altitude_ref_);
    const auto  latitude_r = Angle(latitude_ref), longitude_r = Angle(longitude_ref);
    const float x = 6371000 * (longitude.radians() - longitude_r.radians()) * std::cos(latitude_r.radians()),
                y = 6371000 * (latitude.radians() - latitude_r.radians());
    coord         = cv::Point2f(x, y);
  }

  static cv::Mat Rx(float radians) {
    // clang-format off
    return (cv::Mat_<float>(3, 3) << 
      1,  0,             0,
      0,  cos(radians), -sin(radians),
      0,  sin(radians),  cos(radians));
    // clang-format on
  }

  static cv::Mat Ry(float radians) {
    // clang-format off
    return (cv::Mat_<float>(3, 3) << 
     cos(radians),  0, sin(radians),
     0,             1, 0,
    -sin(radians),  0, cos(radians));
    // clang-format on
  }

  static cv::Mat Rz(float radians) {
    // clang-format off
    return (cv::Mat_<float>(3, 3) << 
    cos(radians), -sin(radians), 0,
    sin(radians),  cos(radians), 0,
    0,             0,            1);
    // clang-format on
  }

  explicit Pose(
      const float& yaw_d,
      const float& pitch_d,
      const float& roll_d,
      const float& latitude_d,
      const float& longitude_d,
      const float& altitude_d) :
      yaw(yaw_d), pitch(pitch_d), roll(roll_d), latitude(latitude_d), longitude(longitude_d), altitude(altitude_d) {
    cv::Mat R_z = Rz(yaw.radians());
    cv::Mat R_y = Ry(pitch.radians());
    cv::Mat R_x = Rx(roll.radians());
    cv::Mat m1_ = R_z * R_y * R_x * Ry(Angle::PI / 2);
    m1_.copyTo(R_);
    cv::Mat m2_ = (cv::Mat_<float>(3, 1) << 0.0f, 0.0f, -altitude.get());
    m2_.copyTo(t_);
    cv::hconcat(R_, t_, T_);
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
};

struct Intrinsic {
private:

  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;

public:

  explicit Intrinsic(Intrinsic&& intrinsic) :
      camera_matrix(std::move(intrinsic.camera_matrix)), distortion_coefficients(intrinsic.distortion_coefficients) {}

  explicit Intrinsic(const float& sensor_width, const float& focal, const float& w, const float& h) {
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
};

class IntrinsicFactory {
private:

  static unordered_map<std::string, float> build_sensor_width_database() {
    fs::path sensor_width_database_path(SENSOR_WIDTH_DATABASE);

    unordered_map<std::string, float> sensor_width_database;
    if(!fs::exists(sensor_width_database_path)) {
      cerr << "Error: Sensor width database not found\n";
      return sensor_width_database;
    }
    std::ifstream ifs(sensor_width_database_path);
    if(!ifs.is_open()) {
      cerr << "Error: Failed to open sensor width database\n";
      return sensor_width_database;
    }
    std::string line;
    while(std::getline(ifs, line)) {
      auto v = line | views::split(';')
               | views::transform([](auto token_range) { return std::string(token_range.begin(), token_range.end()); });
      std::vector<std::string> tokens(v.begin(), v.end());
      if(tokens.size() != 2) {
        // cerr << "Error: Invalid sensor width entry format of the line: " << line << "\n";
        continue;
      }
      sensor_width_database.emplace(tokens[0], std::stod(tokens[1]));
    }
    return sensor_width_database;
  }

  static const inline unordered_map<std::string, float> sensor_width_database = build_sensor_width_database();

public:

  static optional<Intrinsic> build(const std::string& sensor, const float& focal, const float& w, const float& h) {
    if(sensor_width_database.count(sensor) == 0) {
      cerr << "Error: Sensor width not found in the database. Sensor name is " << sensor << "\n";
      return nullopt;
    }
    return std::make_optional<Intrinsic>(sensor_width_database.at(sensor), focal, w, h);
  }
};

struct ImgData {
public:

  Pose                      pose;
  Intrinsic                 intrinsic;
  Property<fs::path>        path;
  Property<Exiv2::ExifData> exif;
  Property<Exiv2::XmpData>  xmp;
  Image                     img, ortho;

  explicit ImgData(Pose&& pose, Intrinsic&& intrinsic, fs::path&& path, Exiv2::ExifData&& exif, Exiv2::XmpData&& xmp) :
      pose(std::move(pose)), intrinsic(std::move(intrinsic)), path(std::move(path)), exif(std::move(exif)),
      xmp(std::move(xmp)) {}

  explicit ImgData(ImgData&& val) :
      pose(std::move(val.pose)), intrinsic(std::move(val.intrinsic)), path(std::move(val.path)),
      img(std::move(val.img)), ortho(std::move(val.ortho)), exif(std::move(val.exif)), xmp(std::move(val.xmp)) {}

  friend ostream& operator<<(ostream& os, const ImgData& data) {
    os << "Path: " << data.path.get() << "\n"
       << "Camera Matrix: " << data.intrinsic << "\n"
       << "Pose: " << data.pose << "\n";
    return os;
  }

  void read() {
    cv::Mat img_ = cv::imread(path.get().string());
    if(img_.empty()) {
      return;
    }
    auto&& [max_w, max_h] = resolution;

    float factor = std::max(1.0f * img_.cols / max_w, 1.0f * img_.rows / max_h);
    if(factor > 1.0f) {
      cv::resize(img_, img.get_mut(), cv::Size(), 1.0f / factor, 1.0f / factor, cv::INTER_AREA);
    } else {
      img.set(std::move(img_));
    }
  }

  void generate_ortho() { ortho.set(orthorectify(img.get().cols, img.get().rows)); }

  void write(const fs::path& output_dir) const {
    fs::path output_path = output_dir / path.get().filename();
    img.write(output_path);
    if(!fs::exists(output_path)) {
      std::cerr << "Error: " << output_path << " could not be written\n";
      return;
    }
    write_exif_xmp(img.get().cols, img.get().rows, output_path);
  }

  void write_ortho(const fs::path& output_dir) {
    fs::path output_path = output_dir / path.get().filename();
    ortho.write(output_path);
    if(!fs::exists(output_path)) {
      std::cerr << "Error: " << output_path << " could not be written\n";
      return;
    }
    write_exif_xmp(ortho.get().cols, ortho.get().rows, output_path);
  }

private:

  static const inline pair<int, int> resolution = {4096, 4096};

  void write_exif_xmp(const int w, const int h, const fs::path& output_path) const {
    Exiv2::ExifData info           = exif.get();
    info["Exif.Image.ImageWidth"]  = w;
    info["Exif.Image.ImageLength"] = h;

    auto output_img = Exiv2::ImageFactory::open(output_path.string());
    output_img->setExifData(info);
    output_img->setXmpData(xmp.get());
    output_img->writeMetadata();
  }

  cv::Point3f backproject(const cv::Point2f& point) const {
    cv::Mat point_ = (cv::Mat_<float>(3, 1) << point.x, point.y, 1);
    cv::Mat K_inv  = intrinsic.K().inv();
    float   gamma  = -pose.t().at<float>(2, 0) / cv::Mat(pose.R().row(2) * K_inv * point_).at<float>(0, 0);
    cv::Mat xyz_w  = gamma * pose.R() * K_inv * point_ + pose.t();
    return cv::Point3f(xyz_w.at<float>(0, 0), xyz_w.at<float>(1, 0), xyz_w.at<float>(2, 0));
  }

  cv::Point2f project(const cv::Point3f& point) const {
    cv::Mat p_cam   = (cv::Mat_<float>(3, 1) << point.x, point.y, point.z + pose.altitude_ref.get());
    float   z_w     = p_cam.at<float>(2, 0);
    cv::Mat p_pixel = intrinsic.K() * p_cam / z_w;
    return cv::Point2f(p_pixel.at<float>(0, 0), p_pixel.at<float>(1, 0));
  }

  cv::Mat orthorectify(const float w, const float h) const {
    std::vector<cv::Point2f> src = {cv::Point2f(w, 0), cv::Point2f(0, 0), cv::Point2f(0, h), cv::Point2f(w, h)};

    auto v = src | views::transform([this](auto&& point) { return backproject(point); })
             | views::transform([this](auto&& point) { return project(point); });

    std::vector<cv::Point2f> dst(v.begin(), v.end());

    cv::Rect rect = cv::boundingRect(dst);

    std::for_each(dst.begin(), dst.end(), [&rect](auto&& point) {
      point.x -= rect.x;
      point.y -= rect.y;
    });
    cv::Mat M = cv::getPerspectiveTransform(src, dst);
    cv::Mat dst_img;
    cv::warpPerspective(img.get(), dst_img, M, cv::Size(rect.width, rect.height), cv::INTER_CUBIC);
    return dst_img;
  }
};

class ImgDataFactory {
private:

  struct ExifKey {
    static const inline std::string latitude = "Exif.GPSInfo.GPSLatitude", longitude = "Exif.GPSInfo.GPSLongitude",
                                    altitude = "Exif.GPSInfo.GPSAltitude", model = "Exif.Image.Model",
                                    make = "Exif.Image.Make", focal_length = "Exif.Photo.FocalLength";
    static const inline std::vector<std::string> keys = {latitude, longitude, altitude, model, make, focal_length};
  };

  struct XmpKey {
    static const inline std::string latitude = "Xmp.drone-dji.GpsLatitude", longitude = "Xmp.drone-dji.GpsLongitude",
                                    altitude = "Xmp.drone-dji.RelativeAltitude", yaw = "Xmp.drone-dji.GimbalYawDegree",
                                    pitch = "Xmp.drone-dji.GimbalPitchDegree", roll = "Xmp.drone-dji.GimbalRollDegree";
    static const inline std::vector<std::string> keys = {latitude, longitude, altitude, yaw, pitch, roll};
  };

  static const inline unordered_set<std::string> extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

  template <typename U, typename T>
  static optional<std::string> validate(const T& data, const std::vector<std::string>& keys) {
    for(const auto& key : keys) {
      if(data.findKey(U(key)) == data.end()) {
        std::stringstream ss;
        ss << "Error: Key " << key << " not found\n";
        return ss.str();
      }
    }
    return nullopt;
  }

public:

  static optional<unique_ptr<ImgData>> build(fs::path path) {
    if(!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      return nullopt;
    }
    auto image_info = Exiv2::ImageFactory::open(path.string());
    if(image_info.get() == 0) {
      return nullopt;
    }
    try {
      image_info->readMetadata();
    } catch(std::exception& e) {
      cerr << "Error: " << e.what() << "\n";
      return nullopt;
    }
    Exiv2::ExifData exif = image_info->exifData();
    Exiv2::XmpData  xmp  = image_info->xmpData();
    if(exif.empty() || xmp.empty()) {
      return nullopt;
    }
    optional<std::string> res = validate<Exiv2::ExifKey>(exif, ExifKey::keys);
    if(res.has_value()) {
      cerr << path << " " << res.value();
      return nullopt;
    }
    res = validate<Exiv2::XmpKey>(xmp, XmpKey::keys);
    if(res.has_value()) {
      cerr << path << " " << res.value();
      return nullopt;
    }
    std::stringstream sensor_name;
    sensor_name << exif[ExifKey::make].toString() << " " << exif[ExifKey::model].toString();

    const float width = exif["Exif.Image.ImageWidth"].toFloat(), height = exif["Exif.Image.ImageLength"].toFloat();

    auto intrinsic = IntrinsicFactory::build(sensor_name.str(), exif[ExifKey::focal_length].toFloat(), width, height);
    if(!intrinsic.has_value()) {
      return nullopt;
    }

    return std::make_unique<ImgData>(
        std::move(Pose(
            xmp[XmpKey::yaw].toFloat(),
            xmp[XmpKey::pitch].toFloat(),
            xmp[XmpKey::roll].toFloat(),
            xmp[XmpKey::latitude].toFloat(),
            xmp[XmpKey::longitude].toFloat(),
            xmp[XmpKey::altitude].toFloat())),
        std::move(intrinsic.value()),
        std::move(path),
        std::move(exif),
        std::move(xmp));
  }
};
} // namespace Ortho

#endif