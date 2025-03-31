#ifndef ORTHO_IMAGE_HPP
#define ORTHO_IMAGE_HPP

#include <filesystem>
#include <iostream>
#include <mutex>
#include <optional>
#include <thread>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "lru.hpp"
#include "rotate_rectify.hpp"

namespace fs = std::filesystem;

namespace Ortho {
class Image {
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

  fs::path        path, ortho_path;
  Exiv2::ExifData exif_;
  Exiv2::XmpData  xmp_;

  void load_exif_xmp() {
    if(!exif_.empty() && !xmp_.empty()) {
      return;
    }
    std::lock_guard<std::mutex> lock(mtx);

    auto image_info = Exiv2::ImageFactory::open(path.string());
    if(!image_info) {
      std::cerr << "Error: " << path.string() << " could not be opened by Exiv2\n";
      return;
    }
    try {
      image_info->readMetadata();
    } catch(std::exception& e) {
      std::cerr << "Error: readMetadata " << e.what() << "\n";
      return;
    }
    exif_ = image_info->exifData();
    xmp_  = image_info->xmpData();
  }

public:

  using MatRefLockPair = TRefLockPair<cv::Mat>;

  explicit Image() = default;

  explicit Image(const fs::path& img_read_path, const fs::path& temporary_save_path) : path(img_read_path) {
    cache.put(std::move(Image_(img_read_path)));
    ortho_path = temporary_save_path / (path.stem().string() + "_ortho" + path.extension().string());
  }

  void rotate_rectify(const Pose& pose, const Intrinsic& intrinsic) {
    auto [img_, lock] = img();
    cv::Mat ortho_img = Ortho::rotate_rectify(img_.size(), pose, intrinsic, img_);
    lock.unlock();
    if(ortho_img.empty()) {
      std::cerr << "Error: " << path.string() << " could not be rotate-rectified\n";
      return;
    }
    cache.put(std::move(Image_(ortho_path, std::move(ortho_img))));
  }

  MatRefLockPair img() {
    auto [img_, lock] = cache.get(path).value();
    return std::make_pair(std::ref(img_.get()), std::move(lock));
  }

  MatRefLockPair ortho() {
    auto [img_, lock] = cache.get(ortho_path).value();
    return std::make_pair(std::ref(img_.get()), std::move(lock));
  }

  const Exiv2::ExifData& exif_data() {
    load_exif_xmp();
    return exif_;
  }

  const Exiv2::XmpData& xmp_data() {
    load_exif_xmp();
    return xmp_;
  }
};
} // namespace Ortho
#endif