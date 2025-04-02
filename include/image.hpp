#ifndef ORTHO_IMAGE_HPP
#define ORTHO_IMAGE_HPP

#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "log.hpp"
#include "lru.hpp"
#include "rotate_rectify.hpp"
#include "utility.hpp"

namespace Ortho {

using MatRefLockPair = TRefLockPair<cv::Mat>;

class Image {
public:

  Image() = default;

  explicit Image(const fs::path& img_read_path, const fs::path& temporary_save_path) : path(img_read_path) {
    cache.put(std::move(Image_(img_read_path)));
    rotate_path      = temporary_save_path / (path.stem().string() + "_rotate" + path.extension().string());
    rotate_mask_path = temporary_save_path / (path.stem().string() + "_rotate_mask" + path.extension().string());
  }

  void rotate_rectify(const Pose& pose, const Intrinsic& intrinsic) {
    auto [img_, lock]         = img();
    auto&& [rotate_img, mask] = Ortho::rotate_rectify(img_.size(), pose, intrinsic, img_);
    lock.unlock();
    if(rotate_img.empty()) {
      ERROR("Error: {} could not be rotate-rectified", path.string());
      return;
    }
    cache.put(std::move(Image_(rotate_path, std::move(rotate_img))));
    cache.put(std::move(Image_(rotate_mask_path, std::move(mask))));
  }

  const fs::path& get_img_path() const { return path; }

  fs::path get_img_name() const { return path.filename(); }

  fs::path get_img_stem() const { return path.stem(); }

  fs::path get_img_extension() const { return path.extension(); }

  MatRefLockPair img() const {
    auto [img_, lock] = cache.get(path).value();
    return {img_.get(), std::move(lock)};
  }

  MatRefLockPair rotate() const {
    auto [img_, lock] = cache.get(rotate_path).value();
    return {img_.get(), std::move(lock)};
  }

  MatRefLockPair rotate_mask() const {
    auto [img_, lock] = cache.get(rotate_mask_path).value();
    return {img_.get(), std::move(lock)};
  }

  Exiv2::ExifData& exif_data() {
    load_exif_xmp();
    return exif_;
  }

  Exiv2::XmpData& xmp_data() {
    load_exif_xmp();
    return xmp_;
  }

private:

  struct Image_ : public CacheElem<fs::path> {
  private:

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

    Image_(const key_type& path) : CacheElem(false), path(path) { check_create_parent_directory(); }

    Image_(const key_type& path, cv::Mat&& img) : CacheElem(true), path(path), img_(std::move(img)) {
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
      decimate_keep_aspect_ratio(&img_, {1024, 1024});
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

    inline std::size_t size() const override {
      if(img_.empty()) {
        return 0;
      }
      return img_.cols * img_.rows * img_.channels() * img_.elemSize1();
    }
  };

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