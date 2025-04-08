#ifndef ORTHO_IMAGE_HPP
#define ORTHO_IMAGE_HPP

#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "log.hpp"
#include "mem.hpp"
#include "utility.hpp"

namespace Ortho {

class ImageMem : public ManageAble {
public:

  template <typename T>
    requires std::same_as<std::decay_t<T>, cv::Mat>
  ImageMem(T&& img) : img(std::forward<T>(img)) {}

  size_t size() const noexcept override {
    if(img.empty()) {
      return 0;
    }
    return img.cols * img.rows * img.channels() * img.elemSize1();
  }

  cv::Mat& get() { return img; }

private:

  cv::Mat img;
};

struct ImgRefGuard {
  ImgRefGuard(RefGuard&& refguard) : refguard(std::move(refguard)) {}

  cv::Mat& get() { return refguard.get<ImageMem>().get(); }

  void unlock() { refguard.unlock(); }

private:

  RefGuard refguard;
};

class Image {
public:

  Image() = default;

  Image(fs::path img_read_path, cv::ImreadModes mode = cv::IMREAD_COLOR) : path(img_read_path), initialized(true) {
    if(!fs::exists(path)) {
      ERROR("Path {} does not exist.", path.string());
      return;
    }
    mem.register_node(
        path.string(),
        nullptr,
        SwapInFunc([this, mode] {
          cv::Mat img = read(path, mode);
          decimate_keep_aspect_ratio(&img, resolution);
          return new ImageMem(std::move(img));
        }),
        SwapOutFunc([](ManageAblePtr ptr) {}));
  }

  Image(fs::path temporary_save_path, cv::Mat&& img) : path(temporary_save_path), initialized(true) {
    fs::path parent_path = path.parent_path();
    if(!fs::exists(parent_path) && !fs::create_directories(parent_path)) {
      ERROR("Error: {} could not be created", parent_path.string());
      return;
    }
    mem.register_node(
        path.string(),
        std::make_unique<ImageMem>(std::move(img)),
        SwapInFunc([this] { return new ImageMem(std::move(read(path, cv::IMREAD_UNCHANGED))); }),
        SwapOutFunc([this](ManageAblePtr ptr) {
          if(ptr) {
            cv::imwrite(path.string(), dynamic_cast<ImageMem*>(ptr.get())->get());
          }
        }));
  }

  void delay_initialize(fs::path temporary_save_path, cv::Mat&& img) {
    if(initialized) {
      return;
    }
    path        = temporary_save_path;
    initialized = true;
    mem.register_node(
        path.string(),
        std::make_unique<ImageMem>(std::move(img)),
        SwapInFunc([this] { return new ImageMem(std::move(read(path, cv::IMREAD_UNCHANGED))); }),
        SwapOutFunc([this](ManageAblePtr ptr) {
          if(ptr) {
            cv::imwrite(path.string(), dynamic_cast<ImageMem*>(ptr.get())->get());
          }
        }));
  }

  ImgRefGuard get() const {
    check_init();
    return mem.get_node(path.string()).value();
  }

  const fs::path& get_img_path() const {
    check_init();
    return path;
  }

  fs::path get_img_name() const {
    check_init();
    return path.filename();
  }

  fs::path get_img_stem() const {
    check_init();
    return path.stem();
  }

  fs::path get_img_extension() const {
    check_init();
    return path.extension();
  }

  bool is_initialized() const { return initialized; }

private:

  void check_init() const {
    if(!initialized) {
      throw std::runtime_error("Error: Image not initialized");
    }
  }

  static cv::Mat read(const fs::path& path, cv::ImreadModes mode) {
    cv::Mat img = cv::imread(path.string(), mode);
    if(img.empty()) {
      throw std::runtime_error("Error: " + path.string() + " could not be read");
    }
    return img;
  }

  static inline cv::Size resolution{1024, 1024};

  fs::path path;

  bool initialized{false};
};

class ExifXmp {
public:

  ExifXmp() = default;

  ExifXmp(const fs::path& img_read_path) : path(img_read_path) {}

  Exiv2::ExifData& exif_data() {
    check_and_load_exif_xmp();
    return exif_;
  }

  Exiv2::XmpData& xmp_data() {
    check_and_load_exif_xmp();
    return xmp_;
  }

  const fs::path& get_img_path() const { return path; }

private:

  fs::path        path;
  Exiv2::ExifData exif_;
  Exiv2::XmpData  xmp_;

  static inline std::mutex xmp_lock;

  void check_and_load_exif_xmp() {
    if(!exif_.empty() && !xmp_.empty()) {
      return;
    }
    auto image_info = Exiv2::ImageFactory::open(path.string());
    if(!image_info) {
      ERROR("Error: {} could not be opened by Exiv2", path.string());
      return;
    }
    std::lock_guard<std::mutex> lock(xmp_lock);
    try {
      image_info->readMetadata();
    } catch(std::exception& e) {
      ERROR("An error occur while reading Metadata: {}", e.what());
      return;
    }
    exif_ = image_info->exifData();
    xmp_  = image_info->xmpData();
  }
};
} // namespace Ortho
#endif