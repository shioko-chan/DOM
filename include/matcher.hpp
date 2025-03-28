#ifndef SUPERPOINT_LIGHTGLUE_MATCHER_HPP
#define SUPERPOINT_LIGHTGLUE_MATCHER_HPP

#include <algorithm>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "matchpair.hpp"
#include "ort.hpp"
#include "progress.hpp"

#ifndef SUPERPOINT_WEIGHT
  #define SUPERPOINT_WEIGHT "superpoint.onnx"
#endif

#ifndef LIGHTGLUE_WEIGHT
  #define LIGHTGLUE_WEIGHT "superpoint_lightglue_fused_fp16.onnx"
#endif

namespace Ortho {
class Matcher {
private:

  static constexpr float superpoint_threshold = 0.15f, lightglue_threshold = 0.7f;

  InferEnv superpoint, lightglue;

public:

  Matcher() : superpoint("[superpoint]", SUPERPOINT_WEIGHT), lightglue("[lightglue]", LIGHTGLUE_WEIGHT) {}

  void match(MatchPairs& pairs, ImgsData& imgs_data, Progress& progress) {
    progress.reset(pairs.size());

    auto batches = pairs | std::views::chunk_by([](const auto& lhs, const auto& rhs) { return lhs.first == rhs.first; });

    for(auto&& batch : batches) {
      auto& img_data = imgs_data[batch.begin()->first];
      img_data.read_img();
      img_data.generate_ortho();

      // progress.update(n);
    }
  }
};
} // namespace Ortho
#endif