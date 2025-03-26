#include <algorithm>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <ranges>
#include <set>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "knn.hpp"
#include "matchpair.hpp"
#include "progress.hpp"

namespace fs     = std::filesystem;
namespace views  = std::views;
namespace ranges = std::ranges;

using namespace Ortho;

struct MultiThreadProcess {
private:

  using UPtrImgData  = std::unique_ptr<ImgData>;
  using UPtrProgress = std::unique_ptr<Progress>;

  UPtrProgress             progress;
  std::vector<fs::path>    img_paths;
  fs::path                 output_dir;
  std::vector<UPtrImgData> img_data;
  std::vector<MatchPair>   match_pairs;

  static auto generate_start_end(unsigned int total, unsigned int dividor) {
    int  base      = total / dividor;
    int  remainder = total % dividor;
    auto sequence = views::iota(0u, dividor) | views::transform([=](int i) { return i < remainder ? base + 1 : base; });
    std::vector<int> cumulative{0};
    std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
    return views::iota(0u, dividor)
           | views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
  }

  auto run(std::function<void(int)>&& process) {
    std::vector<std::thread> threads;
    progress->rerun();
    auto v = generate_start_end(img_paths.size(), std::thread::hardware_concurrency())
             | views::transform([this, &process](auto&& start_end) {
                 auto&& [start, end] = start_end;
                 return std::thread([this, start, end, &process]() {
                   for(int i = start; i < end; i++, progress->update()) {
                     process(i);
                   }
                 });
               });
    threads.assign(v.begin(), v.end());

    for(auto&& thread : threads) {
      if(thread.joinable()) {
        thread.join();
      }
    }
  }

  void find_and_set_reference_coord() {
    std::vector<float> latitude, longitude, altitude;

    for(auto&& data : img_data) {
      latitude.push_back(data->pose.latitude.get());
      longitude.push_back(data->pose.longitude.get());
      altitude.push_back(data->pose.altitude.get());
    }

    auto latitude_ref  = std::accumulate(latitude.begin(), latitude.end(), 0.0) / latitude.size();
    auto longitude_ref = std::accumulate(longitude.begin(), longitude.end(), 0.0) / longitude.size();
    auto altitude_ref  = std::accumulate(
        altitude.begin(), altitude.end(), std::numeric_limits<float>::min(), [](const float& max, const float& x) {
          return std::max(max, x);
        });

    for(auto&& data : img_data) {
      data->pose.set_reference(latitude_ref, longitude_ref, altitude_ref);
    }
  }

public:

  MultiThreadProcess(fs::path input_dir, fs::path output_dir) : output_dir(output_dir) {
    std::transform(
        fs::directory_iterator(input_dir),
        fs::directory_iterator(),
        std::back_inserter(img_paths),
        [](const auto& entry) { return entry.path(); });
    progress = std::make_unique<Progress>(img_paths.size());
    std::generate_n(std::back_inserter(img_data), img_paths.size(), []() { return UPtrImgData{nullptr}; });
  }

  void get_image_info() {
    run([this](int i) {
      fs::path& img_path = img_paths[i];

      auto res = ImgDataFactory::build(img_path);
      if(!res.has_value()) {
        std::cerr << "Error: " << img_path << " could not be processed\n";
        return;
      }
      img_data[i] = std::move(res.value());
    });
    find_and_set_reference_coord();
  }

  void orthorectify() {
    run([this](int i) {
      img_data[i]->read();
      img_data[i]->generate_ortho();
      img_data[i]->img.release();
    });
  }

  void write_ortho() {
    run([this](int i) { img_data[i]->write_ortho(output_dir); });
  }

  void find_neighbours() {
    auto knn = KNN(15, img_data | views::transform([](auto&& data) { return data->pose.coord.get(); }) | views::common);

    std::vector<std::vector<MatchPair>> matches(img_data.size());
    run([this, &knn, &matches](int i) {
      auto neighbours = knn.find_nearest_neighbour(i);
      for(auto&& neighbour : neighbours) {
        if(i < neighbour) {
          matches[i].emplace_back(i, neighbour);
        } else {
          matches[i].emplace_back(neighbour, i);
        }
      }
    });

    auto v = matches | views::join | views::common;

    std::set<MatchPair> match_set(v.begin(), v.end());
    match_pairs.assign(match_set.begin(), match_set.end());
  }

  void panorama() {
    std::vector<cv::Mat> orthos;
    for(auto&& data : img_data) {
      orthos.push_back(std::move(data->ortho.get_mut()));
      data.reset();
    }

    auto stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
    // stitcher->setWaveCorrection(false);
    cv::Mat panorama;
    auto    status = cv::Stitcher::Status::OK;
    try {
      status = stitcher->stitch(orthos, panorama);
    } catch(cv::Exception& e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }
    if(status != cv::Stitcher::OK) {
      std::cerr << "拼接失败，错误代码: " << static_cast<int>(status) << std::endl;
      switch(status) {
        case cv::Stitcher::ERR_NEED_MORE_IMGS:
          std::cerr << "错误原因: 图像数量不足或重叠区域不够。" << std::endl;
          break;
        case cv::Stitcher::ERR_HOMOGRAPHY_EST_FAIL:
          std::cerr << "错误原因: 单应性矩阵估计失败。" << std::endl;
          break;
        case cv::Stitcher::ERR_CAMERA_PARAMS_ADJUST_FAIL:
          std::cerr << "错误原因: 相机参数调整失败。" << std::endl;
          break;
        default:
          std::cerr << "未知错误。" << std::endl;
      }
    } else {
      cv::imwrite("panorama.jpg", panorama);
    }
  }
};

int main(int argc, char* const argv[]) {
  if(argc != 3) {
    std::cout << "Usage: " << argv[0] << " input_dir output_dir\n";
    return 1;
  }

  fs::path input_dir(argv[1]);
  if(!fs::exists(input_dir)) {
    std::cerr << "Error: " << input_dir << " does not exist\n";
    return 1;
  }
  fs::path output_dir(argv[2]);
  if(!fs::exists(output_dir)) {
    fs::create_directory(output_dir);
  }

  auto process = MultiThreadProcess(input_dir, output_dir);

  std::cout << "[1/3] Getting image information\n";
  process.get_image_info();
  std::cout << "[2/3] Orthorectifying images\n";
  process.orthorectify();
  std::cout << "[3/3] Writing orthorectified images\n";
  process.find_neighbours();
  // process.write_ortho();
  // process.panorama();

  pipeline_terminate();
  return 0;
}
