#ifndef ORTHO_MLS_HPP
#define ORTHO_MLS_HPP

#include <pcl/surface/mls.h>

namespace Ortho {

void fun() {
  pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> mls;
  mls.setInputCloud(cloud);    // 输入点云
  mls.setSearchRadius(0.03);   // 搜索邻域半径
  mls.setPolynomialFit(true);  // 启用多项式拟合
  mls.setPolynomialOrder(2);   // 二次曲面
  mls.setComputeNormals(true); // 是否计算法向量

  pcl::PointCloud<pcl::PointXYZ>::Ptr smoothed(new pcl::PointCloud<pcl::PointXYZ>);
  mls.process(*smoothed);

  // 步骤2：统计离群点去除
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(smoothed_cloud);
  sor.setMeanK(50);            // 邻域点数
  sor.setStddevMulThresh(1.0); // 标准差阈值
  sor.filter(*filtered_cloud);
}

} // namespace Ortho
#endif