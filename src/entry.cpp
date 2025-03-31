#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>

#include "pipeline.hpp"

namespace fs = std::filesystem;

using namespace Ortho;

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

  pipeline_initialize();

  auto process = MultiThreadProcess(input_dir, output_dir, output_dir / "tmp");
  std::cout << "[1/3] Getting image information\n";
  process.get_image_info();
  std::cout << "[2/3] Rotate rectifying images\n";
  process.rotate_rectify();
  std::cout << "[3/3] Matching neighbor images\n";
  // process.find_neighbors();
  // process.write_ortho();
  process.match();

  pipeline_terminate();
  return 0;
}
