#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <ranges>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <exiv2/exiv2.hpp>
#include <opencv2/opencv.hpp>

#include "utility/imgdata.hpp"
#include "utility/progress.hpp"

namespace fs = std::filesystem;
namespace views = std::views;
namespace ranges = std::ranges;

using std::cerr;
using std::cout;
using std::map;
using std::string;
using std::vector;

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
  cv::Rect_<double> bounds;     // 正射影像的地理范围 (min_x, min_y, width, height)
  double            resolution; // 分辨率（米/像素）
};

OrthoParams calculateOrthoBounds(const cv::Mat& H, const cv::Size& imageSize) {
  vector<cv::Point2d> corners =
      {cv::Point2d(0, 0),
       cv::Point2d(imageSize.width, 0),
       cv::Point2d(imageSize.width, imageSize.height),
       cv::Point2d(0, imageSize.height)};

  vector<cv::Point2d> worldCorners;
  cv::perspectiveTransform(corners, worldCorners, H.inv());

  double minX = INFINITY, minY = INFINITY, maxX = -INFINITY, maxY = -INFINITY;
  for(const auto& p : worldCorners) {
    minX = std::min(minX, p.x);
    minY = std::min(minY, p.y);
    maxX = std::max(maxX, p.x);
    maxY = std::max(maxY, p.y);
  }

  return {
      cv::Rect_<double>(minX, minY, maxX - minX, maxY - minY),
      0.05 // 分辨率：0.05 米/像素
  };
}

cv::Mat generateOrthoImage(const cv::Mat& undistorted, const cv::Mat& H, const OrthoParams& params) {
  int cols = static_cast<int>(params.bounds.width / params.resolution);
  int rows = static_cast<int>(params.bounds.height / params.resolution);

  cv::Mat ortho(rows, cols, CV_8UC3);
  cv::Mat mapX(rows, cols, CV_32F);
  cv::Mat mapY(rows, cols, CV_32F);

  // 填充映射关系：正射影像像素 → 世界坐标 → 原始图像像素
  for(int y = 0; y < rows; ++y) {
    for(int x = 0; x < cols; ++x) {
      double  worldX = params.bounds.x + x * params.resolution;
      double  worldY = params.bounds.y + y * params.resolution;
      cv::Mat pt = (cv::Mat_<double>(3, 1) << worldX, worldY, 1);
      cv::Mat pixel = H * pt;
      pixel /= pixel.at<double>(2); // 齐次坐标归一化
      mapX.at<float>(y, x) = pixel.at<double>(0);
      mapY.at<float>(y, x) = pixel.at<double>(1);
    }
  }
  cv::remap(undistorted, ortho, mapX, mapY, cv::INTER_LINEAR);
  return ortho;
}

int findMode(auto data) {
  std::unordered_map<int, int> freqMap;
  int                          maxCount = 0;
  int                          mode = static_cast<int>(std::round(data[0]));

  for(int num : data) {
    freqMap[num]++;
    if(freqMap[num] > maxCount) {
      maxCount = freqMap[num];
      mode = num;
    }
  }
  return mode;
}

auto generate_start_end(int total, int dividor) {
  int  base = total / dividor;
  int  remainder = total % dividor;
  auto sequence = views::iota(0, dividor) | views::transform([=](int i) { return i < remainder ? base + 1 : base; });
  std::vector<int> cumulative {0};
  std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
  return views::iota(0, dividor)
         | views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
}

struct ThreadSharingContext {
private:

  Ortho::Progress&        progress;
  vector<Ortho::ImgData>& imgs_data;
  fs::path                output_dir;
  double                  avg_yaw;

public:

  ThreadSharingContext() = delete;

  ThreadSharingContext(Ortho::Progress& progress, vector<Ortho::ImgData>& imgs_data, fs::path output_dir, double avg_yaw) :
      progress(progress), imgs_data(imgs_data), output_dir(output_dir), avg_yaw(avg_yaw) {}

  std::thread launch(int start, int end) {
    return std::thread([this, start = start, end = end]() {
      for(int i = start; i < end; i++) {
        auto&       img_data = imgs_data[i];
        const auto& img = img_data.img.get();
        fs::path    output_path = output_dir / img_data.path.get().filename();
        cv::Mat     dst;

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
        auto   yaw = img_data.yaw.to_degrees();
        double diff = avg_yaw - yaw;
        ;
        if(std::abs(std::round(diff)) > 5.0) {
          cv::Point2f center(dst.cols / 2.0f, dst.rows / 2.0f);
          cv::Mat     rot_mat = cv::getRotationMatrix2D(center, diff, 1.0);
          cv::Rect2f  bbox = cv::RotatedRect(cv::Point2f(), dst.size(), diff).boundingRect2f();
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
    });
  }
};

int main(int argc, char* const argv[]) {
  if(argc != 3) {
    std::cout << "Usage: " << argv[0] << " input_dir output_dir\n";
    return 1;
  }
  auto img_data_views =
      ranges::subrange(fs::directory_iterator(argv[1]), fs::directory_iterator())
      | views::transform([](const auto& entry) { return entry.path(); })
      | views::transform([](const auto& path) -> std::optional<ImgData> { return ImgDataFactory::build(path); })
      | views::filter([](const auto& opt) { return opt.has_value(); })
      | views::transform([](const auto& opt) { return opt.value(); });
  vector<ImgData> imgs_data;
  ranges::move(img_data_views, std::back_inserter(imgs_data));
  fs::path output_dir(argv[2]);
  if(!fs::exists(output_dir)) {
    fs::create_directory(output_dir);
  }
  Ortho::Progress progress(imgs_data.size());
  int avg_yaw = findMode(imgs_data | views::transform([](const ImgData& data) { return data.yaw.to_degrees(); }));
  ThreadSharingContext context(progress, imgs_data, output_dir, avg_yaw);
  for(auto&& img_data : imgs_data) {
    std::cout << img_data << "\n";
  }
  vector<std::thread> threads;
  auto                thread_views = generate_start_end(imgs_data.size(), std::thread::hardware_concurrency())
                      | views::transform([&](auto&& start_end) {
                          auto&& [start, end] = start_end;
                          return context.launch(start, end);
                        });
  ranges::copy(thread_views, std::back_inserter(threads));
  for(auto&& thread : threads) {
    thread.join();
  }

  return 0;
}
