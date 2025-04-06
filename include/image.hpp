#ifndef ORTHO_IMAGE_HPP
#define ORTHO_IMAGE_HPP

#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "log.hpp"
#include "lru.hpp"
#include "utility.hpp"

namespace Ortho {

struct Image_ : public ManagementUnit<fs::path> {
private:

  static inline cv::Size resolution{4096, 4096};

  cv::Mat  img_;
  key_type path;

  inline void check_create_parent_directory() {
    if(!fs::exists(path.parent_path())) {
      if(!fs::create_directories(path.parent_path())) {
        throw std::runtime_error("Error: " + path.parent_path().string() + " could not be created");
      }
    }
  }

public:

  Image_(const key_type& path) : ManagementUnit(false), path(path) { check_create_parent_directory(); }

  Image_(const key_type& path, cv::Mat&& img) : ManagementUnit(true), path(path), img_(std::move(img)) {
    check_create_parent_directory();
  }

  const cv::Mat& get() const { return img_; }

  void swap_in() override {
    if(!img_.empty()) {
      throw std::runtime_error("Error: Image is already in memory");
    }
    if(!fs::exists(path)) {
      throw std::runtime_error("Error: " + path.string() + " does not exist");
    }
    img_ = cv::imread(path.string());
    decimate_keep_aspect_ratio(&img_, resolution);
    if(img_.empty()) {
      throw std::runtime_error("Error: " + path.string() + " could not be read");
    }
  }

  void swap_out() override {
    if(img_.empty()) {
      throw std::runtime_error("Error: Image is not in memory");
    }
    check_create_parent_directory();
    if(fs::exists(path)) {
      img_.release();
      return;
    }
    if(!cv::imwrite(path.string(), img_)) {
      throw std::runtime_error("Error: " + path.string() + " could not be written");
    }
    img_.release();
  }

  const inline key_type& get_key() const override { return path; }

  inline size_t size() const override {
    if(img_.empty()) {
      return 0;
    }
    return img_.cols * img_.rows * img_.channels() * img_.elemSize1();
  }
};

class Image {
public:

  Image() = default;

  explicit Image(const fs::path& img_read_path, const fs::path& temporary_save_path) : path(img_read_path) {
    cache.put(std::move(Image_(img_read_path)));
    rotate_path      = temporary_save_path / (path.stem().string() + "_rotate" + path.extension().string());
    rotate_mask_path = temporary_save_path / (path.stem().string() + "_rotate_mask" + path.extension().string());
  }

  const fs::path& get_img_path() const { return path; }

  fs::path get_img_name() const { return path.filename(); }

  fs::path get_img_stem() const { return path.stem(); }

  fs::path get_img_extension() const { return path.extension(); }

  std::optional<TRefLockPair<Image_>> img() const { return cache.get(path); }

  std::optional<TRefLockPair<Image_>> rotate_rectified() const { return cache.get(rotate_path); }

  std::optional<TRefLockPair<Image_>> rotate_rectified_mask() const { return cache.get(rotate_mask_path); }

  Exiv2::ExifData& exif_data() {
    load_exif_xmp();
    return exif_;
  }

  Exiv2::XmpData& xmp_data() {
    load_exif_xmp();
    return xmp_;
  }

  void set_rotate_rectified(cv::Mat& img) { cache.put(std::move(Image_(rotate_path, std::move(img)))); }

  void set_rotate_rectified_mask(cv::Mat& mask) { cache.put(std::move(Image_(rotate_mask_path, std::move(mask)))); }

private:

  static inline LRU<Image_> cache{8ul * (1ul << 30)};

  static inline std::mutex mtx;

  fs::path        path, rotate_path, rotate_mask_path;
  Exiv2::ExifData exif_;
  Exiv2::XmpData  xmp_;

  void load_exif_xmp() {
    if(!exif_.empty() && !xmp_.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mtx);

    auto image_info = Exiv2::ImageFactory::open(path.string());
    if(!image_info) {
      ERROR("Error: {} could not be opened by Exiv2", path.string());
      return;
    }
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