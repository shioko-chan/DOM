#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "config.hpp"
#include "log.hpp"
#include "pipeline.hpp"

namespace fs = std::filesystem;

using namespace Ortho;

int main(int argc, char* const argv[]) {
  if(argc != 3) {
    MESSAGE("Usage: {} <input_dir> <output_dir>", argv[0]);
    return 1;
  }

  fs::path input_dir(argv[1]);
  if(!fs::exists(input_dir)) {
    LOG_ERROR("Input directory \"{}\" does not exist", input_dir.string());
    return 1;
  }
  fs::path output_dir(argv[2]);
  if(!fs::exists(output_dir)) {
    fs::create_directory(output_dir);
  }

  auto process = Pipeline(input_dir, output_dir, output_dir / "tmp");
  MESSAGE("[1/5] Getting image information");
  process.get_image_info();
  MESSAGE("[2/5] Rectifying images");
  process.rotate_rectify();
  MESSAGE("[3/5] Matching neighbor images");
  process.match(NEIGHBOR_PROPOSAL);
  // process.ortho_rectify();
  MESSAGE("[4/5] Stitching panorama");
  // process.stitch();
  return 0;
}
