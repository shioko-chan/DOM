#ifndef ORTHO_STITCHER_HPP
#define ORTHO_STITCHER_HPP
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/opencv.hpp>

#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "utility.hpp"

// {
//   Points<float> corners =
//       {Point<float>(0, 0),
//        Point<float>(img2.cols - 1, 0),
//        Point<float>(img2.cols - 1, img2.rows - 1),
//        Point<float>(0, img2.rows - 1)};
//   Points<float> corners1 =
//       {Point<float>(0, 0),
//        Point<float>(img1.cols - 1, 0),
//        Point<float>(img1.cols - 1, img1.rows - 1),
//        Point<float>(0, img1.rows - 1)};
//   cv::transform(corners, corners, M);
//   corners.insert(corners.end(), corners1.begin(), corners1.end());
//   auto v =
//       corners | std::views::transform([](const Point<float>& p) { return Point<int>(abs_ceil(p.x), abs_ceil(p.y)); });
//   Points<int> corners_int(v.begin(), v.end());
//   cv::Rect    rect = cv::boundingRect(corners_int);
//   std::for_each(corners.begin(), corners.end(), [&rect](Point<float>& p) {
//     p.x -= rect.x;
//     p.y -= rect.y;
//   });
//   cv::Mat result1(rect.height, rect.width, img1.type(), cv::Scalar(0, 0, 0));
//   img1.copyTo(result1(cv::Rect(corners[4].x, corners[4].y, img1.cols, img1.rows)));
//   cv::Mat       result2(rect.height, rect.width, img1.type(), cv::Scalar(0, 0, 0));
//   Points<float> from =
//       {Point<float>(0, 0),
//        Point<float>(img2.cols - 1, 0),
//        Point<float>(img2.cols - 1, img2.rows - 1),
//        Point<float>(0, img2.rows - 1)};
//   Points<float> to(corners.begin(), corners.begin() + 4);
//   cv::Mat       M = cv::estimateAffinePartial2D(from, to);
//   cv::warpAffine(img2, result2, M, result1.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
//   cv::Mat avg;
//   cv::addWeighted(result1, 0.5, result2, 0.5, 0, avg);
//   cv::Mat mask1(rect.height, rect.width, img1_mask.type(), cv::Scalar(0));
//   img1_mask.copyTo(mask1(cv::Rect(corners[4].x, corners[4].y, img1_mask.cols, img1_mask.rows)));
//   cv::Mat mask2(rect.height, rect.width, img1_mask.type(), cv::Scalar(0));
//   cv::warpAffine(img2_mask, mask2, M, result1.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
//   cv::Mat res;
//   result1.copyTo(res, mask1);
//   result2.copyTo(res, mask2);
//   avg.copyTo(res, mask1 & mask2);
//   if(!fs::exists(temporary_save_path / "foo")) {
//     fs::create_directories(temporary_save_path / "foo");
//   }
//   cv::imwrite(
//       temporary_save_path / "foo"
//           / (lhs_img.get_img_stem().string() + "_" + rhs_img.get_img_stem().string() + "_avg.jpg"),
//       res);
// }
// auto v0 = points0 | std::views::transform([](const Point<float>& p) { return cv::KeyPoint(p.x, p.y, 1); });
//         auto v1 = points1 | std::views::transform([](const Point<float>& p) { return cv::KeyPoint(p.x, p.y, 1); });
//         std::vector<cv::KeyPoint> kpts0(v0.begin(), v0.end()), kpts1(v1.begin(), v1.end());
//         cv::Mat                   resultImg;
//         cv::drawMatches(img1, kpts0, img2, kpts1, inlier_matches, resultImg);

namespace Ortho {
class Stitcher {
private:

  std::vector<std::unordered_map<int, std::unordered_map<int, cv::Mat>>> adjacent_matrixes;

public:

  Stitcher() = default;

  Stitcher(MatchPairs& match_pairs) {
    adjacent_matrixes.resize(1);
    auto& adjacent_matrix = adjacent_matrixes[0];
    for(const auto& match_pair : match_pairs) {
      cv::invertAffineTransform(match_pair.M, adjacent_matrix[match_pair.second][match_pair.first]);
      adjacent_matrix[match_pair.first].emplace(match_pair.second, std::move(match_pair.M));
    }
  }

  cv::Mat stitch(ImgsData& imgs_data) {}
};

} // namespace Ortho
#endif