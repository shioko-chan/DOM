#ifndef ORTHO_PIPELINE_HPP
#define ORTHO_PIPELINE_HPP

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <ranges>
#include <thread>
#include <unordered_map>
#include <vector>

#include <exiv2/exiv2.hpp>

#include "imgdata.hpp"
#include "knn.hpp"
#include "log.hpp"
#include "matcher.hpp"
#include "matchpair.hpp"
#include "progress.hpp"

namespace fs = std::filesystem;

namespace Ortho {

void pipeline_initialize() { Exiv2::XmpParser::initialize(); }

void pipeline_terminate() { Exiv2::XmpParser::terminate(); }

struct MultiThreadProcess {
private:

  Progress              progress;
  std::vector<fs::path> img_paths;
  fs::path              output_dir, temporary_save_dir;
  ImgsData              imgs_data;
  MatchPairs            match_pairs;

  static auto generate_start_end(unsigned int total, unsigned int divisor) {
    int  base      = total / divisor;
    int  remainder = total % divisor;
    auto sequence =
        std::views::iota(0u, divisor) | std::views::transform([=](int i) { return i < remainder ? base + 1 : base; });
    std::vector<int> cumulative{0};
    std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
    return std::views::iota(0u, divisor)
           | std::views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
  }

  auto run(std::function<void(int)>&& process) {
    std::vector<std::thread> threads;
    progress.rerun();
    auto v = generate_start_end(img_paths.size(), std::thread::hardware_concurrency())
             | std::views::transform([this, &process](auto&& start_end) {
                 auto&& [start, end] = start_end;
                 return std::thread([this, start, end, &process]() {
                   for(int i = start; i < end; i++, progress.update()) {
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

    for(auto&& data : imgs_data) {
      latitude.push_back(data.pose.latitude.degrees());
      longitude.push_back(data.pose.longitude.degrees());
      altitude.push_back(data.pose.altitude);
    }

    auto latitude_ref  = std::accumulate(latitude.begin(), latitude.end(), 0.0f) / latitude.size();
    auto longitude_ref = std::accumulate(longitude.begin(), longitude.end(), 0.0f) / longitude.size();
    auto altitude_ref  = std::accumulate(
        altitude.begin(), altitude.end(), std::numeric_limits<float>::min(), [](const float& max, const float& x) {
          return std::max(max, x);
        });
    for(auto&& data : imgs_data) {
      data.pose.set_reference(latitude_ref, longitude_ref, altitude_ref);
    }
  }

public:

  MultiThreadProcess(fs::path input_dir, fs::path output_dir, fs::path temporary_save_dir) :
      output_dir(output_dir), temporary_save_dir(temporary_save_dir) {
    std::transform(
        fs::directory_iterator(input_dir),
        fs::directory_iterator(),
        std::back_inserter(img_paths),
        [](const auto& entry) { return entry.path(); });
    progress.reset(img_paths.size());
    imgs_data.resize(img_paths.size());
  }

  void get_image_info() {
    run([this](int i) {
      fs::path& img_path = img_paths[i];

      auto res = ImgDataFactory::build(img_path, temporary_save_dir);
      if(!res.has_value()) {
        ERROR("Error: {} could not be processed", img_path.string());
        return;
      }
      imgs_data[i] = std::move(res.value());

      imgs_data[i].img;
    });
    find_and_set_reference_coord();
  }

  void rotate_rectify() {
    run([this](int i) { imgs_data[i].rotate_rectify(); });
  }

  void find_neighbors(const int k = 10) {
    auto knn =
        KNN(k, imgs_data | std::views::transform([](auto&& data) { return data.pose.coord; }) | std::views::common);

    std::vector<std::vector<MatchPair>> matches(imgs_data.size());
    run([this, &knn, &matches](int i) {
      auto neighbors = knn.find_nearest_neighbour(i);
      for(auto&& neighbour : neighbors) {
        if(i < neighbour) {
          matches[i].emplace_back(i, neighbour);
        } else {
          matches[i].emplace_back(neighbour, i);
        }
      }
    });

    auto v = matches | std::views::join | std::views::common;

    std::set<MatchPair> match_set(v.begin(), v.end());
    match_pairs.assign(match_set.begin(), match_set.end());
  }

  void match() {
    find_neighbors(10);
    Matcher matcher(temporary_save_dir);
    matcher.match(match_pairs, imgs_data, progress);
  }

  void stitch() {}
};

} // namespace Ortho

#endif