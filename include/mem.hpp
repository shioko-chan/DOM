#ifndef ORTHO_LRU_HPP
#define ORTHO_LRU_HPP

#include <atomic>
#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>

#include "log.hpp"

namespace Ortho {

class ManageAble {
public:

  virtual inline size_t size() const noexcept = 0;

  virtual ~ManageAble() = default;
};

using Lock = std::unique_lock<std::mutex>;

struct RefGuard {
public:

  RefGuard(ManageAble* ptr, Lock lock, std::atomic_uint64_t* available, std::condition_variable* cv) noexcept :
      ptr(ptr), lock(std::move(lock)), available(available), cv(cv), valid(true) {}

  RefGuard(RefGuard&& other) noexcept :
      ptr(other.ptr), lock(std::move(other.lock)), available(other.available), cv(other.cv), valid(other.valid) {
    other.valid = false;
  }

  ~RefGuard() {
    if(valid) {
      cleanup();
    }
  }

  template <typename T>
    requires std::derived_from<T, ManageAble>
  T& get() {
    if(!valid) {
      throw std::runtime_error("Deref on a released ref!");
    }
    return *dynamic_cast<T*>(ptr);
  }

  void unlock() {
    if(!valid) {
      throw std::runtime_error("Unlock a lock already unlocked!");
    }
    cleanup();
  }

private:

  void cleanup() {
    valid = false;
    lock.unlock();
    available->fetch_add(ptr->size());
    cv->notify_all();
  }

  ManageAble*              ptr;
  Lock                     lock;
  std::atomic_uint64_t*    available;
  std::condition_variable* cv;
  bool                     valid;
};

using ManageAblePtr = std::unique_ptr<ManageAble>;
using SwapInFunc    = std::function<ManageAble*(void)>;
using SwapOutFunc   = std::function<void(ManageAblePtr)>;

class LRU {
private:

  struct Unit {
  public:

    Unit(ManageAblePtr ptr, SwapInFunc swap_in, SwapOutFunc swap_out) :
        ptr(std::move(ptr)), swap_in(std::move(swap_in)), swap_out(std::move(swap_out)) {}

    ManageAblePtr ptr;
    SwapInFunc    swap_in;
    SwapOutFunc   swap_out;
    std::mutex    mtx;
  };

  using List = std::list<Unit>;
  using UMap = std::unordered_map<std::string, List::iterator>;

  std::mutex              lru_mtx;
  std::condition_variable cv;
  std::atomic_uint64_t    available;
  size_t                  occupied{0};
  const size_t            capacity;
  List                    lru_list;
  UMap                    k_v;

  bool ensure_space(const size_t size) {
    for(auto iter = lru_list.rbegin(); occupied + size > capacity && iter != lru_list.rend(); ++iter) {
      if(iter->ptr) {
        std::unique_lock<std::mutex> lock(iter->mtx, std::try_to_lock);
        if(!lock.owns_lock()) {
          continue;
        }
        occupied -= iter->ptr->size();
        iter->swap_out(std::move(iter->ptr));
      }
    }
    return occupied + size <= capacity;
  }

public:

  LRU(const size_t capacity = 8ul * (1ul << 30)) : capacity(capacity), available(capacity) {}

  void register_node(const std::string& key, ManageAblePtr ptr, SwapInFunc swap_in, SwapOutFunc swap_out) {
    std::unique_lock<std::mutex> lock(lru_mtx);
    auto                         it = k_v.find(key);
    if(it != k_v.end()) {
      WARN(
          "register_node: node of name \"{}\" already been registered, if this is not intended, please check your program.",
          key);
      return;
    }
    if(ptr) {
      while(!ensure_space(ptr->size())) {
        cv.wait(lock, [size_now = ptr->size(), this] { return size_now <= available; });
      }
      occupied += ptr->size();
    }
    auto iter = lru_list.emplace(lru_list.begin(), std::move(ptr), std::move(swap_in), std::move(swap_out));
    k_v.emplace(key, iter);
  }

  std::optional<RefGuard> get_node(const std::string& key) {
    std::unique_lock<std::mutex> lock(lru_mtx);
    while(true) {
      auto it = k_v.find(key);
      if(it == k_v.end()) {
        return std::nullopt;
      }
      auto                         iter = it->second;
      std::unique_lock<std::mutex> unit_lock(iter->mtx, std::try_to_lock);
      if(!unit_lock.owns_lock()) {
        lock.unlock();
        {
          std::lock_guard<std::mutex> temp_lock(iter->mtx);
        }
        lock.lock();
        continue;
      }
      if(iter->ptr) {
        lru_list.splice(lru_list.begin(), lru_list, iter);
        available.fetch_sub(iter->ptr->size());
        return std::make_optional<RefGuard>(iter->ptr.get(), std::move(unit_lock), &available, &cv);
      }
      iter->ptr.reset(iter->swap_in());
      if(!ensure_space(iter->ptr->size())) {
        size_t size_required = iter->ptr->size();
        iter->ptr.reset();
        unit_lock.unlock();
        cv.wait(lock, [size_required, this] { return size_required <= available; });
      } else {
        occupied += iter->ptr->size();
        available.fetch_sub(iter->ptr->size());
        lru_list.splice(lru_list.begin(), lru_list, iter);
        return std::make_optional<RefGuard>(iter->ptr.get(), std::move(unit_lock), &available, &cv);
      }
    }
  }
};

static inline LRU mem{8ul * (1ul << 30)};

} // namespace Ortho

#endif