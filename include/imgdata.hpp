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
  Pose      pose;
  Intrinsic intrinsic;
  Image     img;

  void rotate_rectify() { img.rotate_rectify(pose, intrinsic); }

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
      ERROR("Error: {} is not a valid image file", path.string());
      return std::nullopt;
    }

    Image img(path, temp_save_path);

    auto&& res = IntrinsicFactory::validate_exif(img.exif_data());
    if(res.has_value()) {
      std::cerr << res.value();
      return std::nullopt;
    }
    res = PoseFactory::validate_xmp(img.xmp_data());
    if(res.has_value()) {
      std::cerr << res.value();
      return std::nullopt;
    }

    Intrinsic intrinsic = IntrinsicFactory::build(img.exif_data());
    Pose      pose      = PoseFactory::build(img.xmp_data());

    return {{pose, intrinsic, img}};
  }
};
} // namespace Ortho

#endif