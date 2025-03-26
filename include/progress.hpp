#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace Ortho {
struct Progress {
private:

  std::mutex mtx;
  double     factor = 0.0;
  int        cnt = 0, total = 0;

public:

  Progress() = default;

  Progress(int total) : total(total) {}

  void update() {
    std::lock_guard<std::mutex> lock(mtx);
    cnt += 1;
    if(cnt >= static_cast<int>(std::round(factor * total))) {
      factor = cnt * 1.0 / total;
      std::cout << "\r[" << std::fixed << std::setprecision(2) << factor * 100 << "%] (" << cnt << "/" << total << ")\n";
      factor = std::min(factor + 0.01, 1.0);
    }
  }

  void rerun() {
    std::lock_guard<std::mutex> lock(mtx);
    cnt    = 0;
    factor = 0.0;
  }

  void reset(int total) {
    std::lock_guard<std::mutex> lock(mtx);
    this->total = total;
    cnt         = 0;
    factor      = 0.0;
  }
};
} // namespace Ortho
#endif
