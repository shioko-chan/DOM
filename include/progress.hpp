#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <iostream>
#include <iomanip>
#include <mutex>
#include <cmath>
#include <algorithm>

namespace Ortho {
struct Progress {
private:
  std::mutex mtx;
  double factor = 0.0;
  int cnt = 0, total = 0;
public:
  Progress(int total) :total(total) {}
  void update() {
    std::lock_guard<std::mutex> lock(mtx);
    cnt += 1;
    if (cnt >= static_cast<int>(std::round(factor * total))) {
      factor = std::max(factor, 1.0 * cnt / total);
      std::cout << "\r[" << std::fixed << std::setprecision(0) << factor * 100 << "%] (" << cnt << "/" << total << ")\n";
      factor += 0.01;
    }
  }
  void reset() {
    std::lock_guard<std::mutex> lock(mtx);
    cnt = 0;
    factor = 0.0;
  }
  void new_run(int total) {
    std::lock_guard<std::mutex> lock(mtx);
    this->total = total;
    cnt = 0;
    factor = 0.0;
  }
};
}
#endif
