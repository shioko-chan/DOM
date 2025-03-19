#include <iostream>
#include <iomanip>
#include <cassert>
#include <filesystem>
#include <algorithm>
#include <ranges>
#include <map>
#include <unordered_set>
#include <string>
#include <cmath>
#include <tuple>
#include <vector>
#include <thread>
#include <mutex>
#include <numeric>
#include <unordered_map>
#include <optional>
#include <functional>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "utility/progress.hpp"

namespace fs = std::filesystem;
namespace views = std::views;
namespace ranges = std::ranges;

using std::vector;
using std::map;
using std::string;
using std::unordered_set;
using std::cout;
using std::cerr;

const std::pair<int, int> RESOLUTION = { 1920, 1080 };

template <typename T>
class Property {
private:
  T value;

public:
  Property() = delete;
  explicit Property(const T& value) : value(value) {}
  explicit Property(T&& value) : value(std::move(value)) {}
  const T& get() const {
    return value;
  }
  void set(const T& value) {
    this->value = value;
  }
  friend std::ostream& operator<<(std::ostream& os, const Property& prop) {
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
  double to_radians() const {
    return get() * PI / 180.0;
  }
  double to_degrees() const {
    return get();
  }
  void set_degrees(const double& degrees) {
    set(degrees);
  }
  void set_radians(const double& radians) {
    set(radians * 180.0 / PI);
  }
};

class ImgDataFactory {
private:
  static const vector<string> exif_keys, xmp_keys;
  template<typename U, typename T>
  static bool validate(const T& data, const vector<string>& keys) {
    for (const auto& key : keys) {
      if (data.findKey(U(key)) == data.end()) {
        return false;
      }
    }
    return true;
  }
public:
  static std::optional<ImgData> build(fs::path& path) {
    auto image_info = Exiv2::ImageFactory::open(path);
    if (image_info.get() == 0) {
      return std::nullopt;
    }
    try {
      image_info->readMetadata();
    } catch (std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return std::nullopt;
    }
    Exiv2::ExifData exif = image_info->exifData();
    Exiv2::XmpData xmp = image_info->xmpData();
    if (exif.empty() || xmp.empty()) {
      return std::nullopt;
    }
    if (!validate<Exiv2::ExifKey>(exif, exif_keys) || !validate<Exiv2::XmpKey>(xmp, xmp_keys)) {
      return std::nullopt;
    }
    cv::Mat img = cv::imread(path.string());
    if (img.empty()) {
      return std::nullopt;
    }
    return ImgData(
      xmp[xmp_keys[0]].toFloat(), xmp[xmp_keys[1]].toFloat(), xmp[xmp_keys[2]].toFloat(),
      xmp[xmp_keys[3]].toFloat(), xmp[xmp_keys[4]].toFloat() + 90.0, xmp[xmp_keys[5]].toFloat(),
      path, exif, xmp, img
    );
  }
};
const vector<string> ImgDataFactory::xmp_keys = {
  "Xmp.drone-dji.GpsLatitude",
  "Xmp.drone-dji.GpsLongitude",
  "Xmp.drone-dji.RelativeAltitude",
  "Xmp.drone-dji.GimbalYawDegree",
  "Xmp.drone-dji.GimbalPitchDegree",
  "Xmp.drone-dji.GimbalRollDegree",
};
const vector<string> ImgDataFactory::exif_keys = {
  "Exif.GPSInfo.GPSLatitude",
  "Exif.GPSInfo.GPSLongitude",
  "Exif.GPSInfo.GPSAltitude",
  "Exif.Image.Model",
};
struct ImgData
{
public:
  Angle yaw, pitch, roll;
  Property<double> latitude, longitude, altitude;
  Property<fs::path> path;
  Property<Exiv2::ExifData> exif;
  Property<Exiv2::XmpData> xmp;
  Property<cv::Mat> img;
  cv::Mat camera_matrix() {
    double focal = std::max(img.get().cols, img.get().rows) * (10260.0 / 1000.0) / 13.2;
    cv::Mat camera_matrix = (
      cv::Mat_<double>(3, 3) <<
      focal, 0, img.get().cols / 2,
      0, focal, img.get().rows / 2,
      0, 0, 1
    );
  }
  cv::Mat R() {
    cv::Mat yaw = (cv::Mat_<double>(3, 3) << std::cos(this->yaw.to_radians()), -std::sin(this->yaw.to_radians()), 0,
                                              std::sin(this->yaw.to_radians()), std::cos(this->yaw.to_radians()), 0,
                                              0, 0, 1);
    cv::Mat pitch = (cv::Mat_<double>(3, 3) << std::cos(this->pitch.to_radians()), 0, std::sin(this->pitch.to_radians()),
                                                0, 1, 0,
                                                -std::sin(this->pitch.to_radians()), 0, std::cos(this->pitch.to_radians()));
    cv::Mat roll = (cv::Mat_<double>(3, 3) << 1, 0, 0,
                                              0, std::cos(this->roll.to_radians()), -std::sin(this->roll.to_radians()),
                                              0, std::sin(this->roll.to_radians()), std::cos(this->roll.to_radians()));
    return yaw * pitch * roll;
  }
  cv::Mat t() {
    return (cv::Mat_<double>(3, 1) << 0, 0, altitude.get());
  }
  ImgData() = delete;
  explicit ImgData(double latitude, double longitude, double altitude, double yaw, double pitch, double roll, fs::path& path, Exiv2::ExifData& exif, Exiv2::XmpData& xmp, cv::Mat& img) :
    latitude(latitude), longitude(longitude), altitude(altitude), yaw(yaw), pitch(pitch), roll(roll), path(std::move(path)), exif(std::move(exif)), xmp(std::move(xmp)), img(std::move(img)) {}
  friend std::ostream& operator<<(std::ostream& os, const ImgData& data) {
    os << "Path: " << data.path.get() << "\n"
      << "Latitude: " << data.latitude << "\n"
      << "Longitude: " << data.longitude << "\n"
      << "Altitude: " << data.altitude << "\n"
      << "Yaw: " << data.yaw << "\n"
      << "Pitch: " << data.pitch << "\n"
      << "Roll: " << data.roll << "\n";
    return os;
  }
};

static unordered_set<string> extensions = { ".jpg", ".jpeg", ".png", ".tiff", ".bmp" , ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP" };

cv::Mat computeHomography(const cv::Mat& cameraMatrix, const cv::Mat& R, const cv::Mat& t) {
  cv::Mat r1 = R.col(0);
  cv::Mat r2 = R.col(1);
  cv::Mat extrinsic = cv::Mat::zeros(3, 3, CV_64F);
  r1.copyTo(extrinsic.col(0));
  r2.copyTo(extrinsic.col(1));
  t.copyTo(extrinsic.col(2));
  cv::Mat H = cameraMatrix * extrinsic;
  return H;
}

struct OrthoParams {
  cv::Rect_<double> bounds;  // 正射影像的地理范围 (min_x, min_y, width, height)
  double resolution;         // 分辨率（米/像素）
};

OrthoParams calculateOrthoBounds(const cv::Mat& H, const cv::Size& imageSize) {
  vector<cv::Point2d> corners = {
      cv::Point2d(0, 0),
      cv::Point2d(imageSize.width, 0),
      cv::Point2d(imageSize.width, imageSize.height),
      cv::Point2d(0, imageSize.height)
  };

  vector<cv::Point2d> worldCorners;
  cv::perspectiveTransform(corners, worldCorners, H.inv());

  double minX = INFINITY, minY = INFINITY, maxX = -INFINITY, maxY = -INFINITY;
  for (const auto& p : worldCorners) {
    minX = std::min(minX, p.x);
    minY = std::min(minY, p.y);
    maxX = std::max(maxX, p.x);
    maxY = std::max(maxY, p.y);
  }

  return {
      cv::Rect_<double>(minX, minY, maxX - minX, maxY - minY),
      0.05  // 分辨率：0.05 米/像素
  };
}

cv::Mat generateOrthoImage(const cv::Mat& undistorted, const cv::Mat& H, const OrthoParams& params) {
  int cols = static_cast<int>(params.bounds.width / params.resolution);
  int rows = static_cast<int>(params.bounds.height / params.resolution);

  cv::Mat ortho(rows, cols, CV_8UC3);
  cv::Mat mapX(rows, cols, CV_32F);
  cv::Mat mapY(rows, cols, CV_32F);

  // 填充映射关系：正射影像像素 → 世界坐标 → 原始图像像素
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      double worldX = params.bounds.x + x * params.resolution;
      double worldY = params.bounds.y + y * params.resolution;
      cv::Mat pt = (cv::Mat_<double>(3, 1) << worldX, worldY, 1);
      cv::Mat pixel = H * pt;
      pixel /= pixel.at<double>(2);  // 齐次坐标归一化
      mapX.at<float>(y, x) = pixel.at<double>(0);
      mapY.at<float>(y, x) = pixel.at<double>(1);
    }
  }
  cv::remap(undistorted, ortho, mapX, mapY, cv::INTER_LINEAR);
  return ortho;
}

int findMode(auto data) {
  std::unordered_map<int, int> freqMap;
  int maxCount = 0;
  int mode = static_cast<int>(std::round(data[0]));

  for (int num : data) {
    freqMap[num]++;
    if (freqMap[num] > maxCount) {
      maxCount = freqMap[num];
      mode = num;
    }
  }
  return mode;
}

auto generate_start_end(int total, int dividor) {
  int base = total / dividor;
  int remainder = total % dividor;
  auto sequence = views::iota(0, dividor) | views::transform([=](int i) { return i < remainder ? base + 1 : base; });
  std::vector<int> cumulative { 0 };
  std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
  return views::iota(0, dividor) | views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
}

struct ThreadSharingContext {
private:
  Ortho::Progress& progress;
  vector<ImgData>& imgs_data;
  fs::path output_dir;
  double avg_yaw;
public:
  ThreadSharingContext() = delete;
  ThreadSharingContext(Ortho::Progress& progress, vector<ImgData>& imgs_data, fs::path output_dir, double avg_yaw) :
    progress(progress), imgs_data(imgs_data), output_dir(output_dir), avg_yaw(avg_yaw) {}
  std::thread launch(int start, int end) {
    return std::thread(
      [this, start = start, end = end]() {
        for (int i = start; i < end; i++) {
          auto& img_data = imgs_data[i];
          const auto& img = img_data.img.get();
          fs::path output_path = output_dir / img_data.path.get().filename();
          cv::Mat dst;
          auto&& [max_w, max_h] = RESOLUTION;
          double factor = std::max(1.0 * img.cols / max_w, 1.0 * img.rows / max_h);
          if (factor > 1.0) {
            cv::resize(img, dst, cv::Size(), 1.0 / factor, 1.0 / factor, cv::INTER_AREA);
          }
          // double focal = std::max(img.cols, img.rows) * (10260.0 / 1000.0) / 13.2;
          // cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << focal, 0, img.cols / 2,
          //                                                    0, focal, img.rows / 2,
          //                                                    0, 0, 1);
          // cv::Mat yaw = (cv::Mat_<double>(3, 3) << std::cos(img_data.yaw), -std::sin(img_data.yaw), 0,
          //                                           std::sin(img_data.yaw), std::cos(img_data.yaw), 0,
          //                                           0, 0, 1);
          // cv::Mat pitch = (cv::Mat_<double>(3, 3) << std::cos(img_data.pitch), 0, std::sin(img_data.pitch),
          //                                             0, 1, 0,
          //                                             -std::sin(img_data.pitch), 0, std::cos(img_data.pitch));
          // cv::Mat roll = (cv::Mat_<double>(3, 3) << 1, 0, 0,
          //                                             0, std::cos(img_data.roll), -std::sin(img_data.roll),
          //                                             0, std::sin(img_data.roll), std::cos(img_data.roll));
          // cv::Mat R = yaw * pitch * roll;

          // cv::Mat t = (cv::Mat_<double>(3, 1) << 0, 0, img_data.altitude);
          // cv::Mat H = computeHomography(camera_matrix, R, t);
          // OrthoParams params = calculateOrthoBounds(H, img.size());
          // cv::Mat ortho = generateOrthoImage(img, H, params);
          // fs::path output_path = output_dir / img_data.path.filename();
          // cv::imwrite(output_path.string(), ortho);
          auto yaw = img_data.yaw.to_degrees();
          double diff = avg_yaw - yaw;;
          if (std::abs(std::round(diff)) > 5.0) {
            cv::Point2f center(dst.cols / 2.0f, dst.rows / 2.0f);
            cv::Mat rot_mat = cv::getRotationMatrix2D(center, diff, 1.0);
            cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), dst.size(), diff).boundingRect2f();
            rot_mat.at<double>(0, 2) += bbox.width / 2.0 - dst.cols / 2.0;
            rot_mat.at<double>(1, 2) += bbox.height / 2.0 - dst.rows / 2.0;
            cv::warpAffine(dst, dst, rot_mat, bbox.size(), cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
          }
          cv::imwrite(output_path.string(), dst);
          Exiv2::ExifData info = img_data.exif.get();
          info["Exif.Image.ImageWidth"] = dst.cols;
          info["Exif.Image.ImageLength"] = dst.rows;
          auto output_img = Exiv2::ImageFactory::open(output_path.string());
          output_img->setExifData(info);
          output_img->writeMetadata();
          progress.update();
        }
      }
    );
  }
};

int main(int argc, char* const argv[]) {
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " input_dir output_dir\n";
    return 1;
  }
  auto img_data_views =
    ranges::subrange(fs::directory_iterator(argv[1]), fs::directory_iterator())
    | views::transform([](const auto& entry) {return entry.path(); })
    | views::transform(
      [](const auto& path) -> std::optional<ImgData> {
        return ImgDataFactory::build(path);
      })
    | views::filter(
      [](const auto& opt) {
        return opt.has_value();
      })
    | views::transform(
      [](const auto& opt) {
        return opt.value();
      });
  vector<ImgData> imgs_data;
  ranges::move(img_data_views, std::back_inserter(imgs_data));
  fs::path output_dir(argv[2]);
  if (!fs::exists(output_dir)) {
    fs::create_directory(output_dir);
  }
  Ortho::Progress progress(imgs_data.size());
  int avg_yaw = findMode(imgs_data | views::transform([](const ImgData& data) {return data.yaw.to_degrees(); }));
  ThreadSharingContext context(progress, imgs_data, output_dir, avg_yaw);
  for (auto&& img_data : imgs_data) {
    std::cout << img_data << "\n";
  }
  vector<std::thread> threads;
  auto thread_views =
    generate_start_end(imgs_data.size(), std::thread::hardware_concurrency())
    | views::transform(
      [&](auto&& start_end) {
        auto&& [start, end] = start_end;
        return context.launch(start, end);
      });
  ranges::copy(thread_views, std::back_inserter(threads));
  for (auto&& thread : threads) {
    thread.join();
  }

  return 0;
}
