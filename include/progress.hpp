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
  float      factor = 0.0f;
  int        cnt = 0, total = 0;

public:

  Progress() = default;

  Progress(int total) : total(total) {}

  void update(const int inc = 1) {
    std::lock_guard<std::mutex> lock(mtx);
    cnt += inc;
    if(cnt >= static_cast<int>(std::round(factor * total))) {
      factor = cnt * 1.0f / total;
      std::cout << "\r[" << std::fixed << std::setprecision(2) << factor * 100 << "%] (" << cnt << "/" << total << ")\n";
      factor = std::min(factor + 0.01f, 1.0f);
    }
  }

  void rerun() {
    std::lock_guard<std::mutex> lock(mtx);
    cnt    = 0;
    factor = 0.0f;
  }

  void reset(const int total_) {
    std::lock_guard<std::mutex> lock(mtx);
    total  = total_;
    cnt    = 0;
    factor = 0.0f;
  }
};
} // namespace Ortho
#endif
