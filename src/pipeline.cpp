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

#include "imgdata.hpp"
#include "progress.hpp"

namespace fs     = std::filesystem;
namespace views  = std::views;
namespace ranges = std::ranges;

using std::cerr;
using std::cout;
using std::map;
using std::string;
using std::vector;

auto generate_start_end(int total, int dividor) {
  int  base      = total / dividor;
  int  remainder = total % dividor;
  auto sequence  = views::iota(0, dividor) | views::transform([=](int i) { return i < remainder ? base + 1 : base; });
  std::vector<int> cumulative{0};
  std::partial_sum(sequence.begin(), sequence.end(), std::back_inserter(cumulative));
  return views::iota(0, dividor)
         | views::transform([cumulative](int i) { return std::make_pair(cumulative[i], cumulative[i + 1]); });
}

struct ThreadSharingContext {
private:

  Ortho::Progress&        progress;
  vector<Ortho::ImgData>& imgs_data;
  fs::path                output_dir;

public:

  ThreadSharingContext() = delete;

  ThreadSharingContext(Ortho::Progress& progress, vector<Ortho::ImgData>& imgs_data, fs::path output_dir) :
      progress(progress), imgs_data(imgs_data), output_dir(output_dir) {}

  std::thread launch(int start, int end) {
    return std::thread([this, start = start, end = end]() {
      for(int i = start; i < end; i++) {
        auto&    img_data    = imgs_data[i];
        fs::path output_path = output_dir / img_data.path.get().filename();

        img_data.write_ortho(output_path);

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
  auto img_data_views = ranges::subrange(fs::directory_iterator(argv[1]), fs::directory_iterator())
                        | views::transform([](const auto& entry) { return entry.path(); })
                        | views::transform([](const auto& path) -> std::optional<Ortho::ImgData> {
                            return Ortho::ImgDataFactory::build(path);
                          })
                        | views::filter([](const auto& opt) { return opt.has_value(); })
                        | views::transform([](auto&& opt) { return std::move(opt.value()); });

  vector<Ortho::ImgData> imgs_data;
  ranges::move(img_data_views, std::back_inserter(imgs_data));
  fs::path output_dir(argv[2]);
  if(!fs::exists(output_dir)) {
    fs::create_directory(output_dir);
  }
  Ortho::Progress      progress(imgs_data.size());
  ThreadSharingContext context(progress, imgs_data, output_dir);
  for(auto&& img_data : imgs_data) {
    std::cout << img_data << "\n";
  }

  vector<std::thread> threads;

  auto thread_views = generate_start_end(imgs_data.size(), std::thread::hardware_concurrency())
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
