#ifndef SKYMERGE_LRU_HPP
#define SKYMERGE_LRU_HPP

#include <condition_variable>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "tools/log.hpp"
#include "tools/report.hpp"
#include "types.hpp"

namespace SkyMerge {

class Managed {
public:

  Managed() noexcept = default;

  Managed(const Managed&)                    = delete;
  Managed(Managed&&)                         = delete;
  auto operator=(const Managed&) -> Managed& = delete;
  auto operator=(Managed&&) -> Managed&      = delete;

  virtual ~Managed() noexcept = default;

  [[nodiscard]] virtual inline auto size() const noexcept -> std::uint64_t = 0;
};

struct alignas(64) RefGuard {
public:

  RefGuard(
      Managed*                 managed_ptr,
      Lock                     ref_lock,
      std::uint64_t*           available,
      std::mutex*              available_mtx,
      std::condition_variable* condition_variable) noexcept :
      managed_ptr(managed_ptr), ref_lock(std::move(ref_lock)), available(available), available_mtx(available_mtx),
      cv(condition_variable) {}

  RefGuard(const RefGuard&) = delete;

  RefGuard(RefGuard&& other) noexcept :
      managed_ptr(other.managed_ptr), ref_lock(std::move(other.ref_lock)), available(other.available),
      available_mtx(other.available_mtx), cv(other.cv), valid(other.valid) {
    other.valid = false;
  }

  auto operator=(const RefGuard&) -> RefGuard& = delete;

  auto operator=(RefGuard&& other) noexcept -> RefGuard& {
    managed_ptr = other.managed_ptr;
    ref_lock    = std::move(other.ref_lock);
    available   = other.available;
    cv          = other.cv;
    valid       = other.valid;
    other.valid = false;
    return *this;
  }

  ~RefGuard() noexcept {
    if(valid) {
      cleanup();
    }
  }

  template <typename T>
    requires std::derived_from<T, Managed>
  auto get() noexcept -> T& {
    if(!valid) {
      terminate_with_error("[RefGuard] Deref on a released ref!");
    }
    return *dynamic_cast<T*>(managed_ptr);
  }

  void unlock() noexcept {
    if(!valid) {
      terminate_with_error("[RefGuard] Unlock a lock already unlocked!");
    }
    cleanup();
  }

private:

  void cleanup() noexcept {
    valid = false;
    ref_lock.unlock();
    TempLock avail_lock(*available_mtx);
    *available += managed_ptr->size();
    cv->notify_all();
  }

  Managed*                 managed_ptr;
  Lock                     ref_lock;
  std::mutex*              available_mtx;
  std::uint64_t*           available;
  std::condition_variable* cv;
  bool                     valid{true};
};

using ManagedPtr = std::unique_ptr<Managed>;

using SwapInFunc  = std::function<ManagedPtr(void)>;
using SwapOutFunc = std::function<void(ManagedPtr)>;

class LRU {
public:

  explicit LRU(const std::uint64_t capacity = 8UL * (1UL << 30U)) noexcept : capacity(capacity), available(capacity) {}

  auto contain_node(const std::string& key) noexcept -> bool {
    TempLock lru_lock{lru_mtx};
    return k_v.find(key) != k_v.end();
  }

  void register_node(std::string key, ManagedPtr managed_ptr, SwapInFunc swap_in, SwapOutFunc swap_out) {
    Lock lru_lock{lru_mtx};
    auto k_v_iter = k_v.find(key);
    if(k_v_iter != k_v.end()) {
      THIS_LOG_WARN(
          "register_node: node of name \"{}\" already been registered, if this is not intended, please check your program.",
          key);
      return;
    }
    if(managed_ptr) {
      while(managed_ptr->size() > available) {
        cv.wait(lru_lock, [&managed_ptr, this] noexcept { return managed_ptr->size() <= available; });
      }
      ensure_space(managed_ptr->size());
      occupied += managed_ptr->size();
    }
    auto iter = lru_list.emplace(lru_list.begin(), std::move(managed_ptr), std::move(swap_in), std::move(swap_out));
    k_v.emplace(std::move(key), iter);
  }

  auto get_node(const std::string& key) noexcept -> std::optional<RefGuard> {
    Lock lru_lock(lru_mtx);
    auto k_v_iter = k_v.find(key);
    if(k_v_iter == k_v.end()) {
      return std::nullopt;
    }
    auto iter = k_v_iter->second;
    lru_lock.unlock();
    Lock unit_lock(iter->mtx, std::defer_lock);
    try {
      std::lock(unit_lock, lru_lock);
    } catch(const std::system_error& exception) {
      terminate_with_error(exception, "[LRU] failed to lock unit mutex for key \"{}\".", key);
    }
    auto& managed_ptr = iter->managed_ptr;
    if(managed_ptr) {
      lru_list.splice(lru_list.begin(), lru_list, iter);
      available -= managed_ptr->size();
      return std::make_optional<RefGuard>(managed_ptr.get(), std::move(unit_lock), &available, &lru_mtx, &cv);
    }
    managed_ptr                 = iter->swap_in();
    std::uint64_t size_required = managed_ptr->size();
    if(size_required > available) {
      managed_ptr.reset();
      while(size_required > available) {
        cv.wait(lru_lock, [&size_required, this] noexcept { return size_required <= available; });
      }
      managed_ptr = iter->swap_in();
    }
    ensure_space(size_required);
    occupied += size_required;
    lru_list.splice(lru_list.begin(), lru_list, iter);
    available -= size_required;
    return std::make_optional<RefGuard>(managed_ptr.get(), std::move(unit_lock), &available, &lru_mtx, &cv);
  }

  void release_node_mem(const std::string& key) noexcept {
    TempLock lru_lock(lru_mtx);
    auto     k_v_iter = k_v.find(key);
    if(k_v_iter == k_v.end()) {
      return;
    }
    auto iter = k_v_iter->second;
    swap_out_node(*iter);
  }

private:

  struct alignas(128) Unit {
    Unit(ManagedPtr managed_ptr, SwapInFunc swap_in, SwapOutFunc swap_out) :
        managed_ptr(std::move(managed_ptr)), swap_in(std::move(swap_in)), swap_out(std::move(swap_out)) {}

    ManagedPtr  managed_ptr;
    SwapInFunc  swap_in;
    SwapOutFunc swap_out;
    std::mutex  mtx;
  };

  std::condition_variable cv;

  std::mutex lru_mtx;

  std::uint64_t       available;
  std::uint64_t       occupied{0};
  const std::uint64_t capacity;

  using List = std::list<Unit>;
  using UMap = std::unordered_map<std::string, List::iterator>;

  List lru_list;
  UMap k_v;

  void swap_out_node(Unit& unit) {
    if(unit.managed_ptr) {
      std::unique_lock<std::mutex> unit_lock{unit.mtx, std::try_to_lock};
      if(!unit_lock.owns_lock()) {
        return;
      }
      occupied -= unit.managed_ptr->size();
      unit.swap_out(std::move(unit.managed_ptr));
    }
  }

  void ensure_space(const std::uint64_t size) noexcept {
    for(auto iter = lru_list.rbegin(); occupied + size > capacity && iter != lru_list.rend(); ++iter) {
      swap_out_node(*iter);
    }
  }
};

} // namespace SkyMerge

#endif