#ifndef IMGDATA_HPP
#define IMGDATA_HPP

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
using std::cout;
using std::move;
using std::nullopt;
using std::optional;
using std::ostream;
using std::pair;
using std::sin;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;

namespace fs     = std::filesystem;
namespace ranges = std::ranges;
namespace views  = std::views;

namespace Ortho {

template <typename T>
class Property {
private:

  T value;

public:

  explicit Property() : value(T()) {}

  explicit Property(const T& value) : value(value) {}

  explicit Property(T&& value) : value(move(value)) {}

  Property& operator=(const T& value) = delete;

  Property& operator=(T&& value) {
    set(value);
    return *this;
  }

  const T& get() const { return value; }

  void set(const T& value) { this->value = value; }

  void set(T&& value) { this->value = move(value); }

  friend ostream& operator<<(ostream& os, const Property& prop) {
    os << prop.get();
    return os;
  }
};

class Angle : public Property<double> {
private:

  static constexpr double PI = 3.1415926535897932384626433832795;

  static double normalize(const double& degrees) {
    return degrees;
    // return std::fmod(std::fmod(degrees, 360.0) + 360.0, 360.0);
  }

public:

  Angle() = delete;

  explicit Angle(const double& degrees) : Property(normalize(degrees)) {}

  double to_radians() const { return get() * PI / 180.0; }

  double to_degrees() const { return get(); }

  void set_degrees(const double& degrees) { set(degrees); }

  void set_radians(const double& radians) { set(radians * 180.0 / PI); }
};

struct Coordinate {
private:

  Angle            yaw, pitch, roll;
  Property<double> latitude, longitude, altitude;
  cv::Mat          R_, t_, T_;

public:

  Coordinate() = delete;

  explicit Coordinate(Coordinate&& coord) :
      yaw(move(coord.yaw)), pitch(move(coord.pitch)), roll(move(coord.roll)), latitude(move(coord.latitude)),
      longitude(move(coord.longitude)), altitude(move(coord.altitude)), R_(move(coord.R_)), t_(move(coord.t_)),
      T_(move(coord.T_)) {}

  explicit Coordinate(
      const double& yaw_d,
      const double& pitch_d,
      const double& roll_d,
      const double& latitude_d,
      const double& longitude_d,
      const double& altitude_d) :
      yaw(yaw_d), pitch(pitch_d), roll(roll_d), latitude(latitude_d), longitude(longitude_d), altitude(altitude_d) {
    // clang-format off
    cv::Mat yaw_ = (cv::Mat_<double>(3, 3) << 
      1,  0,                       0,
      0,  cos(yaw.to_radians()),  -sin(yaw.to_radians()),
      0,  sin(yaw.to_radians()),   cos(yaw.to_radians()));
    
    cv::Mat pitch_ = (cv::Mat_<double>(3, 3) << 
      cos(pitch.to_radians()),  0,  sin(pitch.to_radians()),
      0,                        1,  0,
     -sin(pitch.to_radians()),  0,  cos(pitch.to_radians()));

    cv::Mat roll_ = (cv::Mat_<double>(3, 3) << 
      cos(roll.to_radians()),  -sin(roll.to_radians()),   0,
      sin(roll.to_radians()),   cos(roll.to_radians()),   0,
      0,                        0,                        1);
    // clang-format on
    cv::Mat m1_ = yaw_ * pitch_ * roll_;
    m1_.copyTo(R_);
    // cv::Mat m2_ = (cv::Mat_<double>(3, 1) << latitude.get(), longitude.get(), altitude.get());
    cv::Mat m2_ = (cv::Mat_<double>(3, 1) << 0.0, 0.0, altitude.get());
    m2_.copyTo(t_);
    cv::hconcat(R_, t_, T_);
  }

  const cv::Mat& R() const { return R_; }

  const cv::Mat& t() const { return t_; }

  const cv::Mat& T() const { return T_; }

  friend ostream& operator<<(ostream& os, const Coordinate& coord) {
    os << "Yaw: " << coord.yaw << "\n"
       << "Pitch: " << coord.pitch << "\n"
       << "Roll: " << coord.roll << "\n"
       << "Latitude: " << coord.latitude << "\n"
       << "Longitude: " << coord.longitude << "\n"
       << "Altitude: " << coord.altitude << "\n";
    return os;
  }
};

struct Intrinsic {
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;

  Intrinsic() = delete;

  explicit Intrinsic(Intrinsic&& intrinsic) :
      camera_matrix(move(intrinsic.camera_matrix)), distortion_coefficients(intrinsic.distortion_coefficients) {}

  Intrinsic(const Intrinsic& intrinsic) : camera_matrix(intrinsic.camera_matrix.clone()) {}

  explicit Intrinsic(const double& sensor_width, const double& focal, const double& w, const double& h) {
    double pix_f = std::max(w, h) * focal / sensor_width;
    // clang-format off
    cv::Mat m_ = (cv::Mat_<double>(3, 3) << 
      pix_f,      0,  w / 2,
          0,  pix_f,  h / 2,
          0,      0,      1
    );
    // clang-format on
    m_.copyTo(camera_matrix);
  }

  friend ostream& operator<<(ostream& os, const Intrinsic& intrinsic) {
    os << "Camera Matrix: " << intrinsic.camera_matrix << "\n"
       << "Distortion Coefficients: " << intrinsic.distortion_coefficients << "\n";
    return os;
  }
};

class IntrinsicFactory {
private:

  static unordered_map<string, double> build_sensor_width_database() {
    fs::path sensor_width_database_path(SENSOR_WIDTH_DATABASE);

    unordered_map<string, double> sensor_width_database;
    if(!fs::exists(sensor_width_database_path)) {
      cerr << "Error: Sensor width database not found\n";
      return sensor_width_database;
    }
    std::ifstream ifs(sensor_width_database_path);
    if(!ifs.is_open()) {
      cerr << "Error: Failed to open sensor width database\n";
      return sensor_width_database;
    }
    string line;
    while(std::getline(ifs, line)) {
      auto v = line | views::split(';')
               | views::transform([](auto token_range) { return string(token_range.begin(), token_range.end()); });
      vector<string> tokens(v.begin(), v.end());
      if(tokens.size() != 2) {
        // cerr << "Error: Invalid sensor width entry format of the line: " << line << "\n";
        continue;
      }
      sensor_width_database[tokens[0]] = std::stod(tokens[1]);
    }
    return sensor_width_database;
  }

  static const inline unordered_map<string, double> sensor_width_database = build_sensor_width_database();

public:

  static optional<Intrinsic> build(const string& sensor, const double& focal, const double& w, const double& h) {
    if(sensor_width_database.count(sensor) == 0) {
      cerr << "Error: Sensor width not found in the database. Sensor name is " << sensor << "\n";
      return nullopt;
    }
    return Intrinsic(sensor_width_database.at(sensor), focal, w, h);
  }
};

struct ImgData {
public:

  Coordinate                coord;
  Intrinsic                 intrinsic;
  Property<fs::path>        path;
  Property<Exiv2::ExifData> exif;
  Property<Exiv2::XmpData>  xmp;
  Property<cv::Mat>         img, ortho;

  ImgData() = delete;

  explicit ImgData(
      Coordinate&&      coord,
      Intrinsic&&       intrinsic,
      fs::path&&        path,
      Exiv2::ExifData&& exif,
      Exiv2::XmpData&&  xmp,
      cv::Mat&&         img) : coord(move(coord)), intrinsic(move(intrinsic)), path(move(path)), img(move(img)) {
    this->exif.set(exif);
    this->xmp.set(xmp);
    ortho = Property<cv::Mat>(orthorectify(img.cols, img.rows));
  }

  friend ostream& operator<<(ostream& os, const ImgData& data) {
    os << "Path: " << data.path.get() << "\n"
       << "Camera Matrix: " << data.intrinsic << "\n"
       << "Coordinate: " << data.coord << "\n";
    return os;
  }

  void write(const fs::path& output_path) const {
    cv::imwrite(output_path, img.get());
    Exiv2::ExifData info           = exif.get();
    info["Exif.Image.ImageWidth"]  = img.get().cols;
    info["Exif.Image.ImageLength"] = img.get().rows;
    auto output_img                = std::unique_ptr<Exiv2::Image>(Exiv2::ImageFactory::open(output_path));
    output_img->setExifData(info);
    output_img->writeMetadata();
  }

  void write_ortho(const fs::path& output_path) const {
    cv::imwrite(output_path, ortho.get());
    Exiv2::ExifData info           = exif.get();
    info["Exif.Image.ImageWidth"]  = ortho.get().cols;
    info["Exif.Image.ImageLength"] = ortho.get().rows;
    auto output_img                = std::unique_ptr<Exiv2::Image>(Exiv2::ImageFactory::open(output_path));
    output_img->setExifData(info);
    output_img->writeMetadata();
  }

private:

  cv::Point2f project(const cv::Point2f& point) {
    cv::Mat point_ = (cv::Mat_<double>(3, 1) << point.x, point.y, 1);
    double  gamma  = -cv::Mat(coord.t().at<double>(2, 0) / (coord.R().row(2) * intrinsic.camera_matrix.inv() * point_))
                        .at<double>(0, 0);
    cv::Mat xyz_c = gamma * intrinsic.camera_matrix.inv() * point_;
    cv::Mat xyz_c_homo =
        (cv::Mat_<double>(4, 1) << xyz_c.at<double>(0, 0), xyz_c.at<double>(1, 0), xyz_c.at<double>(2, 0), 1);
    cv::Mat xy_w = coord.T().rowRange(0, 1) * xyz_c_homo;
    return cv::Point2f(xy_w.at<double>(0, 0), xy_w.at<double>(0, 1));
  }

  cv::Mat orthorectify(const double w, const double h) {
    vector<cv::Point2f> src = {cv::Point2f(0, 0), cv::Point2f(w, 0), cv::Point2f(w, h), cv::Point2f(0, h)}, dst;
    std::transform(src.begin(), src.end(), std::back_inserter(dst), [this](auto&& point) { return project(point); });
    cv::Point2f min_ =
        std::accumulate(dst.begin(), dst.end(), cv::Point2f(0.0, 0.0), [](cv::Point2f min_, auto&& point) {
          return cv::Point2f(std::min(point.x, min_.x), std::min(point.y, min_.y));
        });
    std::for_each(dst.begin(), dst.end(), [&min_](auto&& point) {
      point.x -= min_.x;
      point.y -= min_.y;
    });
    cv::Size dst_size = std::accumulate(dst.begin(), dst.end(), cv::Size(0, 0), [](cv::Size cur, auto&& point) {
      return cv::Size(
          std::max(cur.width, static_cast<int>(std::ceil(point.x))),
          std::max(cur.height, static_cast<int>(std::ceil(point.y))));
    });
    cv::Mat  M        = cv::getPerspectiveTransform(src, dst);
    cv::Mat  dst_img;
    cv::warpPerspective(img.get(), dst_img, M, dst_size, cv::INTER_CUBIC);
    return dst_img;
  }
};

class ImgDataFactory {
private:

  static const inline pair<int, int> resolution = {2048, 2048};

  struct ExifKey {
    static const inline string latitude = "Exif.GPSInfo.GPSLatitude", longitude = "Exif.GPSInfo.GPSLongitude",
                               altitude = "Exif.GPSInfo.GPSAltitude", model = "Exif.Image.Model",
                               make = "Exif.Image.Make", focal_length = "Exif.Photo.FocalLength";
    static const inline vector<string> keys = {latitude, longitude, altitude, model, make, focal_length};
  };

  struct XmpKey {
    static const inline string latitude = "Xmp.drone-dji.GpsLatitude", longitude = "Xmp.drone-dji.GpsLongitude",
                               altitude = "Xmp.drone-dji.RelativeAltitude", yaw = "Xmp.drone-dji.GimbalYawDegree",
                               pitch = "Xmp.drone-dji.GimbalPitchDegree", roll = "Xmp.drone-dji.GimbalRollDegree";
    static const inline vector<string> keys = {latitude, longitude, altitude, yaw, pitch, roll};
  };

  static const inline unordered_set<string> extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

  template <typename U, typename T>
  static optional<string> validate(const T& data, const vector<string>& keys) {
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

  static optional<ImgData> build(fs::path path) {
    if(!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      return nullopt;
    }
    auto image_info = std::unique_ptr<Exiv2::Image>(Exiv2::ImageFactory::open(path));
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
    optional<string> res = validate<Exiv2::ExifKey>(exif, ExifKey::keys);
    if(res.has_value()) {
      cerr << path << " " << res.value();
      return nullopt;
    }
    res = validate<Exiv2::XmpKey>(xmp, XmpKey::keys);
    if(res.has_value()) {
      cerr << path << " " << res.value();
      return nullopt;
    }
    cv::Mat img = cv::imread(path);
    if(img.empty()) {
      return nullopt;
    }
    auto&& [max_w, max_h] = resolution;

    double factor = std::max(1.0 * img.cols / max_w, 1.0 * img.rows / max_h);
    if(factor > 1.0) {
      cv::resize(img, img, cv::Size(), 1.0 / factor, 1.0 / factor, cv::INTER_AREA);
    }
    std::stringstream sensor_name;
    sensor_name << exif[ExifKey::make].toString() << " " << exif[ExifKey::model].toString();
    auto intrinsic =
        IntrinsicFactory::build(sensor_name.str(), exif[ExifKey::focal_length].toFloat(), img.cols, img.rows);
    if(!intrinsic.has_value()) {
      return nullopt;
    }
    return ImgData(
        move(Coordinate(
            xmp[XmpKey::yaw].toFloat(),
            xmp[XmpKey::pitch].toFloat(),
            xmp[XmpKey::roll].toFloat(),
            xmp[XmpKey::latitude].toFloat(),
            xmp[XmpKey::longitude].toFloat(),
            xmp[XmpKey::altitude].toFloat())),
        move(intrinsic.value()),
        move(path),
        move(exif),
        move(xmp),
        move(img));
  }
};
} // namespace Ortho

#endif