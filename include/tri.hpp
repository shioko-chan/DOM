#ifndef ORTHO_TRI_HPP
#define ORTHO_TRI_HPP

#include <vector>

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

#include "dsu.hpp"
#include "imgdata.hpp"
#include "log.hpp"
#include "matchpair.hpp"
#include "types.hpp"

namespace Ortho {

struct ReprojectionError {
public:

  ReprojectionError(Point<double> img_pnt) : pnt2d(std::move(img_pnt)) {}

  template <typename T>
  bool operator()(const T* const q, const T* const t, const T* const c, const T* const pnt3d, T* residuals) const {
    T p0[3];
    for(size_t i = 0; i < 3; ++i) {
      p0[i] = pnt3d[i] + t[i];
    }
    T p1[3];
    ceres::QuaternionRotatePoint(q, p0, p1);
    residuals[0] = c[0] * p1[0] / p1[2] + c[2] - T(pnt2d.x);
    residuals[1] = c[1] * p1[1] / p1[2] + c[3] - T(pnt2d.y);
    return true;
  }

  static ceres::CostFunction* create(const Point<float>& img_pnt) {
    return new ceres::AutoDiffCostFunction<ReprojectionError, 4, 4, 3, 4, 3>(
        new ReprojectionError(Point<double>(img_pnt)));
  }

private:

  Point<double> pnt2d;
};

struct TriRes {
  Point3<float> pnt3d;
  PointIdxs     pnt2d_idx_vec;
};

std::vector<TriRes> triangulation(const MatchPairs& match_img_pairs, ImgsData& imgs_data) {
  DisjointSetUnion dsu;
  for(const auto& match_img_pair : match_img_pairs) {
    for(const auto& match_pnt_pair : match_img_pair.matches) {
      dsu.append_match(
          PointIdx{match_img_pair.first, match_pnt_pair.first}, PointIdx{match_img_pair.second, match_pnt_pair.second});
    }
  }
  std::vector<PointIdxs> pntidx_vecs = dsu.get_pntidxs();

  auto get_v_c2w = [&imgs_data](const PointIdx& idx) {
    const auto& [img_idx, pnt_idx] = idx;
    auto&   img                    = imgs_data[img_idx];
    cv::Mat v_m                    = img.R().t() * img.K().inv() * img.kpnts.get(pnt_idx);
    v_m.convertTo(v_m, CV_64F);
    Eigen::Vector3d v;
    cv::cv2eigen(v_m, v);
    return v;
  };
  auto get_t_c2w = [&imgs_data](const PointIdx& idx) {
    const auto& [img_idx, _] = idx;
    auto&   img              = imgs_data[img_idx];
    cv::Mat t_m              = -img.t();
    t_m.convertTo(t_m, CV_64F);
    Eigen::Vector3d t;
    cv::cv2eigen(t_m, t);
    return t;
  };
  std::vector<TriRes> res;
  for(const auto& pntidx_vec : pntidx_vecs) {
    size_t n = pntidx_vec.size();
    assert(n > 1);
    size_t          rows = 3 * n, cols = n + 3;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(rows, cols);
    Eigen::VectorXd b(rows);
    for(size_t i = 0; i < n; ++i) {
      Eigen::Vector3d v = get_v_c2w(pntidx_vec[i + 1]), t = get_t_c2w(pntidx_vec[i + 1]);
      A.block<3, 3>(i * 3, 0) = -Eigen::Matrix3d::Identity();
      A.block<3, 1>(i * 3, 3) = v;
      b.segment<3>(i * 3)     = t;
    }
    Eigen::VectorXd x  = A.colPivHouseholderQr().solve(b);
    Eigen::Vector3d wp = x.segment<3>(0);

    ceres::Problem problem;

    // for(size_t i = 0; i < n; ++i) {
    //   Eigen::Vector3d v  = get_v(pntidx_vec[i]);
    //   Eigen::Vector3d wp = x(i) * v - get_t(pntidx_vec[i]);
    //   wp_s += wp;
    // }
    // wp_s /= n;
    // cv::Mat wp_m;
    // cv::eigen2cv(wp_s, wp_m);
    // res.emplace_back(mat2point3(wp_m), pntidx_vec);
  }
}
} // namespace Ortho

#endif