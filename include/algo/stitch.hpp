#ifndef ORTHO_ALGO_STITCH_HPP
#define ORTHO_ALGO_STITCH_HPP

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "ds/dsm.hpp"
#include "ds/imgdata.hpp"
#include "tools/log.hpp"
#include "tools/progress.hpp"

namespace Ortho {

class DSMStitcher {
public:
  static auto stitch(ImgsData& imgs_data, DSM& dsm, Progress& progress) noexcept -> void {
    if(imgs_data.empty() || dsm.empty()) {
      THIS_MESSAGE("empty imgs_data or dsm");
      return;
    }

    THIS_MESSAGE("开始将图像纹理投影到DSM上...");
    progress.reset(dsm.height_map_.rows * dsm.height_map_.cols + 1);

    auto valid_imgs = imgs_data | std::views::filter([](const auto& img_data) { return img_data.is_valid(); });

    if (valid_imgs.empty()) {
      THIS_MESSAGE("没有有效的图像数据");
      return;
    }

    for (auto p = dsm.begin(); p != dsm.end(); ++p) {
      auto& point = *p;
      
      double best_cos_angle = 1.0;
      cv::Vec3b best_color(0, 0, 0);
      bool found_match = false;

         const    auto& world_pt =                         point.point;
    
      for(auto& img_data : valid_imgs) {
        cv::Mat R_w2c = img_data.R_w2c();
        cv::Mat t_c2w = img_data.t_c2w();
        cv::Mat K = img_data.M();
        cv::Mat img = img_data.origin_img().get();
        
        cv::Mat camera_center = t_c2w.clone();
        
        cv::Mat view_vec = camera_center - world_pt;
        double view_len = cv::norm(view_vec);
        view_vec = view_vec / view_len;
        
        cv::Mat normal = (cv::Mat_<float>(3, 1) << 
                        point.normal[0], point.normal[1], point.normal[2]);
        double cos_angle = std::abs(view_vec.dot(normal));
        
        cv::Mat camera_pt = R_w2c * (world_pt - camera_center);
        
        if(camera_pt.at<double>(2, 0) <= 0) {
          continue;
        }
        
        cv::Mat img_pt = K * camera_pt;
        double u = img_pt.at<double>(0, 0) / img_pt.at<double>(2, 0);
        double v = img_pt.at<double>(1, 0) / img_pt.at<double>(2, 0);
        
        if(u < 0 || v < 0 || u >= img.cols || v >= img.rows) {
          continue;
        }
        
        int img_x = static_cast<int>(std::round(u));
        int img_y = static_cast<int>(std::round(v));
        
        if (cos_angle < best_cos_angle) {
          best_cos_angle = cos_angle;
          best_color = img.at<cv::Vec3b>(img_y, img_x);
          found_match = true;
        }
      }
      
      if (found_match) {
        point.tex_ref = best_color;
      }
      
      if ((std::distance(dsm.begin(), p) % 100) == 0) {
        progress.update(100);
      }
    }
    
    progress.update();
    THIS_MESSAGE("纹理拼接完成");
  }
};

} // namespace Ortho

#endif // ORTHO_ALGO_STITCH_HPP

