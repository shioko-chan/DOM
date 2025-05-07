#ifndef SKYMERGE_IMAGE_HPP
#define SKYMERGE_IMAGE_HPP

#include <exception>
#include <filesystem>
#include <memory>
#include <opencv2/core.hpp>
#include <optional>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>
#include <utility>

#include "config.hpp"
#include "tools/mem.hpp"
#include "tools/report_error.hpp"
#include "tools/utility.hpp"
#include "types/cv_alias.hpp"

namespace SkyMerge {

namespace fs = std::filesystem;

class ImageMem : public ManageAble {
public:

  template <typename T>
    requires std::same_as<std::decay_t<T>, cv::Mat>
  explicit ImageMem(T&& img) noexcept : img(std::forward<T>(img)) {}

  [[nodiscard]] auto size() const noexcept -> size_t override {
    if(img.empty()) {
      return 0;
    }
    return img.elemSize1() * img.cols * img.rows * img.channels();
  }

  auto get() noexcept -> cv::Mat& { return img; }

private:

  cv::Mat img;
};

struct ImgRefGuard {
  ImgRefGuard(const ImgRefGuard&)                    = delete;
  ImgRefGuard(ImgRefGuard&&) noexcept                = default;
  auto operator=(const ImgRefGuard&) -> ImgRefGuard& = delete;
  auto operator=(ImgRefGuard&&) -> ImgRefGuard&      = delete;
  ~ImgRefGuard() noexcept                            = default;

  explicit ImgRefGuard(RefGuard&& refguard) noexcept : refguard(std::move(refguard)) {}

  auto get() noexcept -> cv::Mat& { return refguard.get<ImageMem>().get(); }

  void unlock() noexcept { refguard.unlock(); }

private:

  RefGuard refguard;
};

class OriginImage {
public:

  OriginImage() noexcept = default;

  OriginImage(const OriginImage&) noexcept                    = default;
  OriginImage(OriginImage&&) noexcept                         = default;
  auto operator=(const OriginImage&) noexcept -> OriginImage& = default;
  auto operator=(OriginImage&&) noexcept -> OriginImage&      = default;
  ~OriginImage() noexcept                                     = default;

  explicit OriginImage(fs::path img_read_path, cv::ImreadModes mode = cv::IMREAD_COLOR) noexcept :
      path(std::move(img_read_path)), mode(mode) {
    if(!fs::exists(path)) {
      report_error("Image path \"{}\" is not exist.", img_read_path.string());
    }
  }

  [[nodiscard]] auto get() noexcept -> cv::Mat {
    cv::Mat img = read(path, mode);
    decimate_keep_aspect_ratio(&img, ORIGIN_RESOLUTION_LIM);
    img_size = img.size();
    return img;
  }

  [[nodiscard]] auto get_img_path() const noexcept -> const fs::path& { return path; }

  [[nodiscard]] auto get_img_name() const noexcept -> fs::path { return path.filename(); }

  [[nodiscard]] auto get_img_stem() const noexcept -> fs::path { return path.stem(); }

  [[nodiscard]] auto get_img_extension() const noexcept -> fs::path { return path.extension(); }

  [[nodiscard]] auto get_size() noexcept -> cv::Size {
    if(img_size.empty()) {
      img_size = get().size();
    }
    return img_size;
  }

private:

  [[nodiscard]] static auto read(const fs::path& path, cv::ImreadModes mode) noexcept -> cv::Mat {
    cv::Mat img;
    try {
      img = cv::imread(path.string(), mode);
    } catch(const cv::Exception& cv_exception) {
      report_error(cv_exception, "{} could not be read.", path.string());
    } catch(const std::exception& exception) {
      report_error(exception, "{} could not be read.", path.string());
    }
    if(img.empty()) {
      report_error("{} could not be read. Image is empty after cv::imread().", path.string());
    }
    return img;
  }

  cv::Size        img_size;
  fs::path        path;
  cv::ImreadModes mode{cv::IMREAD_COLOR};
};

class Image {
public:

  Image() noexcept = default;

  Image(const Image&) noexcept                    = default;
  Image(Image&&) noexcept                         = default;
  auto operator=(const Image&) noexcept -> Image& = default;
  auto operator=(Image&&) noexcept -> Image&      = default;
  ~Image() noexcept                               = default;

  explicit Image(fs::path temporary_save_path, cv::Mat&& img, const Points<double>& pixel_span = Points<double>{}) noexcept
      : path(std::move(temporary_save_path)), initialized(true) {
    img_size = img.size();
    if(!pixel_span.empty()) {
      auto view = convert_arithmetic_type<float>(pixel_span);
      this->pixel_span.assign(view.begin(), view.end());
    }
    Mem::register_node(
        path.string(),
        std::make_unique<ImageMem>(std::move(img)),
        [path = this->path] noexcept { return std::make_unique<ImageMem>(std::move(read(path, cv::IMREAD_UNCHANGED))); },
        [path = this->path](ManageAblePtr ptr) noexcept {
          if(ptr) {
            ImageMem* img_ptr{};
            try {
              img_ptr = dynamic_cast<ImageMem*>(ptr.get());
            } catch(const std::exception& exception) {
              report_error(exception, "dynamic_cast failed while writing back image {}", path.string());
            }
            write(path, img_ptr->get());
          }
        });
  }

  void delay_initialize(
      fs::path              temporary_save_path,
      cv::Mat&&             img,
      const Points<double>& pixel_span = Points<double>{}) noexcept {
    if(initialized) {
      return;
    }
    *this = Image{std::move(temporary_save_path), std::move(img), pixel_span};
  }

  [[nodiscard]] auto get() noexcept -> ImgRefGuard {
    check_init();
    auto guard = ImgRefGuard{*Mem::get_node(path.string())};
    img_size   = guard.get().size();
    return guard;
  }

  [[nodiscard]] auto get_img_path() const noexcept -> const fs::path& {
    check_init();
    return path;
  }

  [[nodiscard]] auto get_img_name() const noexcept -> fs::path {
    check_init();
    return path.filename();
  }

  [[nodiscard]] auto get_img_stem() const noexcept -> fs::path {
    check_init();
    return path.stem();
  }

  [[nodiscard]] auto get_img_extension() const noexcept -> fs::path {
    check_init();
    return path.extension();
  }

  [[nodiscard]] auto is_initialized() const noexcept -> bool { return initialized; }

  [[nodiscard]] auto get_size() noexcept -> cv::Size {
    check_init();
    if(img_size.empty()) {
      auto guard = get();
      img_size   = guard.get().size();
    }
    return img_size;
  }

  [[nodiscard]] auto check_valid_pixel(HasXY auto point) const noexcept -> bool {
    if(pixel_span.empty()) {
      std::cout << 12382194238467;
      return true;
    }
    return cv::pointPolygonTest(pixel_span, Point<double>{static_cast<double>(point.x), static_cast<double>(point.y)}, false)
           >= 0;
  }

private:

  void check_init() const noexcept {
    if(!initialized) {
      report_error("Image not initialized but an operation was called on it.");
    }
  }

  [[nodiscard]] static auto read(const fs::path& path, cv::ImreadModes mode) noexcept -> cv::Mat {
    cv::Mat img;
    try {
      img = cv::imread(path.string(), mode);
    } catch(const cv::Exception& cv_exception) {
      report_error(cv_exception, "{} could not be read.", path.string());
    } catch(const std::exception& exception) {
      report_error(exception, "{} could not be read.", path.string());
    }
    if(img.empty()) {
      report_error("{} could not be read. Image is empty after cv::imread().", path.string());
    }
    return img;
  }

  static void write(const fs::path& path, const cv::Mat& img) noexcept {
    try {
      cv::imwrite(path.string(), img);
    } catch(const cv::Exception& cv_exception) {
      report_error(cv_exception, "{} could not be written.", path.string());
    } catch(const std::exception& exception) {
      report_error(exception, "{} could not be written.", path.string());
    }
    if(!fs::exists(path)) {
      report_error("{} could not be written. Path still does not exist after cv::imwrite().", path.string());
    }
  }

  cv::Size      img_size;
  Points<float> pixel_span;
  fs::path      path;
  bool          initialized{false};
};

class ExifXmp {
public:

  ExifXmp() noexcept = default;

  explicit ExifXmp(fs::path img_read_path) noexcept : path(std::move(img_read_path)) {}

  [[nodiscard]] auto exif_data() noexcept -> Exiv2::ExifData& {
    check_and_load_exif_xmp();
    return exif_;
  }

  [[nodiscard]] auto xmp_data() noexcept -> Exiv2::XmpData& {
    check_and_load_exif_xmp();
    return xmp_;
  }

  [[nodiscard]] auto get_img_path() const noexcept -> const fs::path& { return path; }

private:

  fs::path        path;
  Exiv2::ExifData exif_;
  Exiv2::XmpData  xmp_;

  void check_and_load_exif_xmp() {
    if(!exif_.empty() && !xmp_.empty()) {
      return;
    }
    try {
      auto image_info = Exiv2::ImageFactory::open(path.string());
      if(!image_info) {
        report_error("Image with path \"{}\" could not be opened by Exiv2", path.string());
      }
      image_info->readMetadata();
      exif_ = image_info->exifData();
      xmp_  = image_info->xmpData();
    } catch(const std::exception& e) {
      report_error(e, "An error occur while reading Metadata of image with path \"{}\"", path.string());
    }
  }
};
} // namespace SkyMerge
#endif
