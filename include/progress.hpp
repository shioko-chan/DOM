#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>

#include "log.hpp"

namespace Ortho {
struct Progress {
private:

  static constexpr int bar_width{50};

  std::mutex mtx;
  int        cnt{0}, total{0};

  void print_bar() {
    std::lock_guard<std::mutex> lock(stream_mtx);
    float                       factor = 1.0f * cnt / total;
    std::cout << BOLD "\r[" << std::fixed << std::setprecision(2) << factor * 100 << "%]";
    int pos = static_cast<int>(std::round(bar_width * factor));
    for(int i = 0; i < bar_width; ++i) {
      if(i < pos)
        std::cout << "=";
      else if(i == pos)
        std::cout << ">";
      else
        std::cout << "-";
    }
    std::cout << "(" << cnt << "/" << total << ")" RESET;
    if(cnt == total) {
      std::cout << std::endl;
    } else {
      std::cout << std::flush;
    }
  }

public:

  Progress() = default;

  Progress(int total) : total(total) {}

  void update(int inc = 1, int current = -1, bool countdown = false) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      if(countdown) {
        if(current >= 0) {
          cnt = total - current;
        } else {
          cnt = cnt - inc;
        }
        cnt = std::max(0, cnt);
      } else {
        if(current >= 0) {
          cnt = current;
        } else {
          cnt = cnt + inc;
        }
        cnt = std::min(total, cnt);
      }
    }
    print_bar();
  }

  inline void rerun() {
    std::lock_guard<std::mutex> lock(mtx);
    cnt = 0;
  }

  inline void reset(int total_) {
    std::lock_guard<std::mutex> lock(mtx);
    total = total_;
    cnt   = 0;
  }
};
} // namespace Ortho
#endif
