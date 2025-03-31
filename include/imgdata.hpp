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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "image.hpp"
#include "pose_intrinsic.hpp"

using std::ostream;
using std::pair;
using std::unordered_map;
using std::unordered_set;

namespace fs     = std::filesystem;
namespace ranges = std::ranges;
namespace views  = std::views;

namespace Ortho {

struct ImgData {
public:

  Pose      pose;
  Intrinsic intrinsic;
  Image     img;

  explicit ImgData() = default;

  ImgData(const ImgData&) = delete;

  ImgData& operator=(const ImgData&) = delete;

  void rotate_rectify() { img.rotate_rectify(pose, intrinsic); }

  explicit ImgData(Pose&& pose, Intrinsic&& intrinsic, const fs::path& path, const fs::path& mid_save_path) :
      pose(std::move(pose)), intrinsic(std::move(intrinsic)), img(path, mid_save_path) {}

  explicit ImgData(ImgData&& val) :
      pose(std::move(val.pose)), intrinsic(std::move(val.intrinsic)), img(std::move(val.img)) {}

  ImgData& operator=(Ortho::ImgData&& val) {
    pose      = std::move(val.pose);
    intrinsic = std::move(val.intrinsic);
    img       = std::move(val.img);
    return *this;
  }

  friend ostream& operator<<(ostream& os, const ImgData& data) {
    os << "Camera Matrix: " << data.intrinsic << "\n"
       << "Pose: " << data.pose << "\n";
    return os;
  }
};

using ImgsData = std::vector<ImgData>;

class ImgDataFactory {
private:

  static inline const unordered_set<std::string> extensions =
      {".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".JPG", ".JPEG", ".PNG", ".TIFF", ".BMP"};

public:

  static std::optional<ImgData> build(fs::path path, fs::path temp_save_path) {
    if(!fs::is_regular_file(path) || extensions.count(path.extension().string()) == 0) {
      std::cerr << "Error: " << path << " is not a valid image file\n";
      return std::nullopt;
    }
    auto image_info = Exiv2::ImageFactory::open(path.string());
    if(image_info.get() == 0) {
      std::cerr << "Error: " << path << " could not be opened by Exiv2\n";
      return std::nullopt;
    }
    try {
      image_info->readMetadata();
    } catch(std::exception& e) {
      std::cerr << "Error: " << e.what() << "\n";
      return std::nullopt;
    }
    Exiv2::ExifData exif = image_info->exifData();
    Exiv2::XmpData  xmp  = image_info->xmpData();
    if(exif.empty() || xmp.empty()) {
      std::cerr << path << " Error: Exif or Xmp data is empty\n";
      return std::nullopt;
    }

    std::optional<std::string> res;
    res = IntrinsicFactory::validate_exif(exif);
    if(res.has_value()) {
      std::cerr << res.value();
      return std::nullopt;
    }
    res = PoseFactory::validate_xmp(xmp);
    if(res.has_value()) {
      std::cerr << res.value();
      return std::nullopt;
    }

    Pose      pose      = PoseFactory::build(xmp);
    Intrinsic intrinsic = IntrinsicFactory::build(exif);

    return std::make_optional<ImgData>(std::move(pose), std::move(intrinsic), path, temp_save_path);
  }
};
} // namespace Ortho

#endif