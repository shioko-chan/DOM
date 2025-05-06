#include <filesystem>
#include <span>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "pipeline.hpp"
#include "tools/log.hpp"
#include "tools/utility.hpp"

namespace fs = std::filesystem;

auto main(const int argc, const char* const argv[]) -> int {
  auto start = std::chrono::high_resolution_clock::now();
  {
    const std::span<const char* const> args{argv, static_cast<size_t>(argc)};
    if(argc != 3) {
      THIS_MESSAGE("Usage: {} <input_dir> <output_dir>", args[0]);
      return 1;
    }
    fs::path input_dir(args[1]);
    if(!fs::exists(input_dir)) {
      THIS_LOG_ERROR("Input directory \"{}\" does not exist", input_dir.string());
      return 1;
    }
    fs::path output_dir(args[2]);
    if(!fs::exists(output_dir)) {
      fs::create_directory(output_dir);
    }

    auto process = Ortho::Pipeline(input_dir, output_dir, output_dir / "temp");
    THIS_MESSAGE("[1/5] Getting image information");
    auto imgs_data = process.get_image_info();
    THIS_MESSAGE("[2/5] Rotating images for matching");
    process.rotate_rectify(imgs_data);
    THIS_MESSAGE("[3/5] Matching neighbor images");
    auto match_pairs = process.match(imgs_data, Ortho::NEIGHBOR_PROPOSAL);
    THIS_MESSAGE("[4/5] Triangulate");
    auto dsm = process.triangulate(imgs_data, match_pairs);
    THIS_MESSAGE("[5/5] Stitching images");
    process.stitch(imgs_data, dsm);
  }
  Ortho::print_run_time(start);
  return 0;
}
