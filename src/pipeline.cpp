#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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

#include "imgdata.hpp"
#include "progress.hpp"

namespace fs     = std::filesystem;
namespace views  = std::views;
namespace ranges = std::ranges;

using std::cerr;
using std::cout;
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

struct MultiThreadProcess {
private:

  unique_ptr<Ortho::Progress> progress;
  vector<fs::path>            img_paths;
  fs::path                    output_dir;

  std::thread launch_(int start, int end) {
    return std::thread([this, start = start, end = end]() {
      for(int i = start; i < end; i++, progress->update()) {
        fs::path& img_path    = img_paths[i];
        fs::path  output_path = output_dir / img_path.filename();
        auto      res         = Ortho::ImgDataFactory::build(img_path);
        if(!res.has_value()) {
          continue;
        }
        Ortho::ImgData& img_data = res.value();
        img_data.write_ortho(output_path);
      }
    });
  }

  static auto generate_start_end(int total, int dividor) {
    int  base      = total / dividor;
    int  remainder = total % dividor;
    auto sequence  = views::iota(0, dividor) | views::transform([=](int i) { return i < remainder ? base + 1 : base; });
    std::vector<int> cumulative{0};
    std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
    return views::iota(0, dividor)
           | views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
  }

public:

  MultiThreadProcess() = delete;

  MultiThreadProcess(fs::path input_dir, fs::path output_dir) : output_dir(output_dir) {
    auto img_data_views = ranges::subrange(fs::directory_iterator(input_dir), fs::directory_iterator())
                          | views::transform([](const auto& entry) { return entry.path(); });
    ranges::move(img_data_views, std::back_inserter(img_paths));
    progress = std::make_unique<Ortho::Progress>(img_paths.size());
  }

  vector<std::thread> launch() {
    auto thread_views = generate_start_end(img_paths.size(), std::thread::hardware_concurrency())
                        | views::transform([this](auto&& start_end) {
                            auto&& [start, end] = start_end;
                            return launch_(start, end);
                          });
    vector<std::thread> threads(thread_views.begin(), thread_views.end());
    return threads;
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

  MultiThreadProcess process(input_dir, output_dir);

  auto threads = process.launch();
  for(auto&& thread : threads) {
    thread.join();
  }

  return 0;
}
