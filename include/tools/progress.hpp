#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <iostream>

#include "tools/ansi.hpp"

namespace SkyMerge {
struct alignas(16) Progress {
private:

  static constexpr int bar_width{50};

  std::atomic<int>  cnt{0};
  std::atomic<int>  total{0};
  std::atomic<bool> is_printing{false};

  void print_bar() {
    bool expected = false;
    if(!is_printing.compare_exchange_strong(expected, true)) {
      return;
    }

    double factor = 1. * cnt.load(std::memory_order_relaxed) / total.load(std::memory_order_relaxed);
    std::cout << ansi::BOLD << "\r[" << std::fixed << std::setprecision(2) << factor * 100 << "%]";
    int pos = static_cast<int>(std::round(bar_width * factor));
    for(int i = 0; i < bar_width; ++i) {
      if(i < pos) {
        std::cout << "=";
      } else if(i == pos) {
        std::cout << ">";
      } else {
        std::cout << "-";
      }
    }
    std::cout << "(" << cnt.load(std::memory_order_relaxed) << "/" << total.load(std::memory_order_relaxed) << ")"
              << ansi::RESET;
    if(cnt.load(std::memory_order_relaxed) == total.load(std::memory_order_relaxed)) {
      std::cout << '\n';
    } else {
      std::cout << std::flush;
    }

    is_printing.store(false, std::memory_order_release);
  }

public:

  Progress() = default;

  explicit Progress(int total_) : total(total_) {}

  void update(int inc = 1, int current = -1, bool countdown = false) {
    if(countdown) {
      if(current >= 0) {
        cnt.store(total.load(std::memory_order_relaxed) - current, std::memory_order_relaxed);
      } else {
        int old_val = cnt.load(std::memory_order_relaxed);
        int new_val = std::max(0, old_val - inc);
        cnt.store(new_val, std::memory_order_relaxed);
      }
    } else {
      if(current >= 0) {
        cnt.store(current, std::memory_order_relaxed);
      } else {
        int old_val = cnt.load(std::memory_order_relaxed);
        int new_val = std::min(total.load(std::memory_order_relaxed), old_val + inc);
        cnt.store(new_val, std::memory_order_relaxed);
      }
    }
    print_bar();
  }

  void rerun() { cnt.store(0, std::memory_order_relaxed); }

  void reset(int total_) {
    total.store(total_, std::memory_order_relaxed);
    cnt.store(0, std::memory_order_relaxed);
  }
};
} // namespace SkyMerge

#endif
