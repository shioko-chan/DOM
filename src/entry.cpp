#include <filesystem>
#include <span>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "pipeline.hpp"
#include "tools/log.hpp"

namespace fs = std::filesystem;

auto main(const int argc, const char* const argv[]) -> int {
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
  process.get_image_info();
  THIS_MESSAGE("[2/5] Rectifying images");
  process.rotate_rectify();
  THIS_MESSAGE("[3/5] Matching neighbor images");
  // process.match(Ortho::NEIGHBOR_PROPOSAL);
  THIS_MESSAGE("[4/5] Triangulate");
  // process.triangulate();
  THIS_MESSAGE("[5/5] Stitching images");
  process.stitch();
  return 0;
}
