#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <ranges>
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

  Property() = delete;

  explicit Property(const T& value) : value(value) {}

  explicit Property(T&& value) : value(move(value)) {}

  const T& get() const { return value; }

  void set(const T& value) { this->value = value; }

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
      cos(yaw.to_radians()),  -sin(yaw.to_radians()),   0,
      sin(yaw.to_radians()),   cos(yaw.to_radians()),   0,
      0,                             0,                 1);

    cv::Mat pitch_ = (cv::Mat_<double>(3, 3) << 
      cos(pitch.to_radians()),  0,  sin(pitch.to_radians()),
      0,                        1,  0,
     -sin(pitch.to_radians()),  0,  cos(pitch.to_radians()));
    
     cv::Mat roll_ = (cv::Mat_<double>(3, 3) << 
      1,  0,                        0,
      0,  cos(roll.to_radians()),  -sin(roll.to_radians()),
      0,  sin(roll.to_radians()),   cos(roll.to_radians()));
    // clang-format on
    cv::Mat m1_ = yaw_ * pitch_ * roll_;
    m1_.copyTo(R_);
    cv::Mat m2_ = (cv::Mat_<double>(3, 1) << latitude.get(), longitude.get(), altitude.get());
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

  // friend
};

struct Intrinsic {
  cv::Mat camera_matrix;
  cv::Mat distortion_coefficients;
  Intrinsic() = delete;

  explicit Intrinsic(const double& focal, const double& w, const double& h) {
    cv::Mat m_ = (cv::Mat_<double>(3, 3) << focal, 0, w / 2, 0, focal, h / 2, 0, 0, 1);
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

  static const unordered_map<string, double> sensor_width_database;

  static unordered_map<string, double> build_sensor_width_database() {
    fs::path                      sensor_width_database_path(SENSOR_WIDTH_DATABASE);
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
      vector<string> tokens;
      for(auto&& token_range : views::split(line, ';')) {
        tokens.emplace_back(token_range.begin(), token_range.end());
      }
      if(tokens.size() != 2) {
        cerr << "Error: Invalid sensor width entry format of the line: " << line << "\n";
        continue;
      }
      sensor_width_database[tokens[0]] = std::stod(tokens[1]);
    }
    return sensor_width_database;
  }

public:

  static optional<Intrinsic> build(const string& sensor, const double& focal, const double& w, const double& h) {
    if(sensor_width_database.count(sensor) == 0) {
      cerr << "Error: Sensor width not found in the database\n";
      return nullopt;
    }
    double sensor_width = sensor_width_database.at(sensor);
    double pixel_focal  = std::max(w, h) * focal / sensor_width;
    return Intrinsic(pixel_focal, w, h);
  }
};

const unordered_map<string, double> IntrinsicFactory::sensor_width_database =
    IntrinsicFactory::build_sensor_width_database();

struct ImgData {
public:

  Coordinate                coord;
  Intrinsic                 intrinsic;
  Property<fs::path>        path;
  Property<Exiv2::ExifData> exif;
  Property<Exiv2::XmpData>  xmp;
  Property<cv::Mat>         img;
  ImgData() = delete;

  explicit ImgData(
      const Coordinate&      coord,
      const Intrinsic&       intrinsic,
      const fs::path&        path,
      const Exiv2::ExifData& exif,
      const Exiv2::XmpData&  xmp,
      const cv::Mat&         img) :
      coord(move(coord)), intrinsic(move(intrinsic)), path(move(path)), exif(move(exif)), xmp(move(xmp)),
      img(move(img)) {}

  friend ostream& operator<<(ostream& os, const ImgData& data) {
    os << "Path: " << data.path.get() << "\n"
       << "Camera Matrix: " << data.intrinsic << "\n"
       << "Coordinate: " << data.coord << "\n";
    return os;
  }
};

class ImgDataFactory {
private:

  static const struct {
    static const inline string latitude = "Exif.GPSInfo.GPSLatitude", longitude = "Exif.GPSInfo.GPSLongitude",
                               altitude = "Exif.GPSInfo.GPSAltitude", model = "Exif.Image.Model",
                               focal_length = "Exif.Photo.FocalLength";
  } exif_keys;

  static const struct {
    static const inline string latitude = "Xmp.drone-dji.GpsLatitude", longitude = "Xmp.drone-dji.GpsLongitude",
                               altitude = "Xmp.drone-dji.RelativeAltitude", yaw = "Xmp.drone-dji.GimbalYawDegree",
                               pitch = "Xmp.drone-dji.GimbalPitchDegree", roll = "Xmp.drone-dji.GimbalRollDegree";
  } xmp_keys;

  template <typename T>
    requires(std::tuple_size_v<std::decay_t<T>> >= 0 && std::is_lvalue_reference_v<T>)
  auto each_member(T&& t) {
    constexpr auto size = std::tuple_size_v<std::remove_cvref_t<T>>;
    return each_member_impl(std::forward<T>(t), std::make_index_sequence<size>());
  }

  template <typename T, size_t... Is>
  auto each_member_impl(T&& t, std::index_sequence<Is...>) {
    using ElemT = std::tuple_element_t<Is, std::remove_cvref_t<T>>;
    return std::vector<std::reference_wrapper<ElemT>>{std::ref(std::get<Is>(t))...};
  }

  static const inline unordered_set<string> extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};
  static const inline pair<int, int> resolution = {1920, 1080};

  template <typename U, typename E, typename T>
  static bool validate(const T& data, const E& keys) {
    for(const auto& key : each_member(keys)) {
      if(data.findKey(U(key)) == data.end()) {
        return false;
      }
    }
    return true;
  }

public:

  static optional<ImgData> build(fs::path& path) {
    if(!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      return nullopt;
    }
    auto image_info = Exiv2::ImageFactory::open(path);
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
    if(!validate<Exiv2::ExifKey>(exif, exif_keys) || !validate<Exiv2::XmpKey>(xmp, xmp_keys)) {
      return nullopt;
    }
    cv::Mat img = cv::imread(path.string());
    if(img.empty()) {
      return nullopt;
    }
    auto&& [max_w, max_h] = resolution;

    double factor = std::max(1.0 * img.cols / max_w, 1.0 * img.rows / max_h);
    if(factor > 1.0) {
      cv::resize(img, img, cv::Size(), 1.0 / factor, 1.0 / factor, cv::INTER_AREA);
    }
    auto intrinsic = IntrinsicFactory::build(
        exif[exif_keys.model].toString(), exif[exif_keys.focal_length].toFloat(), img.cols, img.rows);
    if(!intrinsic.has_value()) {
      return nullopt;
    }
    return ImgData(
        Coordinate(
            xmp[xmp_keys.yaw].toFloat(),
            xmp[xmp_keys.pitch].toFloat(),
            xmp[xmp_keys.roll].toFloat(),
            xmp[xmp_keys.latitude].toFloat(),
            xmp[xmp_keys.longitude].toFloat(),
            xmp[xmp_keys.altitude].toFloat()),
        intrinsic.value(),
        path,
        exif,
        xmp,
        img);
  }
};
} // namespace Ortho
