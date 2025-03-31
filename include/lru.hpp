#ifndef ORTHO_LRU_HPP
#define ORTHO_LRU_HPP

#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace Ortho {

using Lock = std::unique_lock<std::mutex>;

template <typename T>
using TRefLockPair = std::pair<const T&, Lock>;

template <typename T>
concept Hashable = requires(T a) {
  { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

template <typename K>
  requires Hashable<K>
struct CacheElem {
public:

  using key_type = K;

  bool                        in_mem;
  std::unique_ptr<std::mutex> mtx;

  CacheElem() : mtx(std::make_unique<std::mutex>()), in_mem(false) {};

  CacheElem(const bool in_mem) : mtx(std::make_unique<std::mutex>()), in_mem(in_mem) {}

  virtual void                   swap_in()       = 0;
  virtual void                   swap_out()      = 0;
  virtual inline const key_type& get_key() const = 0;
  virtual inline std::size_t     size() const    = 0;
};

template <typename T>
  requires std::derived_from<T, CacheElem<typename T::key_type>>
struct LRU {
private:

  using Key       = typename T::key_type;
  using ListT     = std::list<T>;
  using UMapValue = typename ListT::iterator;
  using UMapT     = std::unordered_map<Key, UMapValue>;

  std::mutex  mtx;
  std::size_t capacity, occupied = 0;
  ListT       lru_list;
  UMapT       k_v;

  void ensure_space(const size_t size) {
    auto iter = lru_list.rbegin();
    while(occupied + size > capacity && iter != lru_list.rend()) {
      if(iter->in_mem) {
        std::unique_lock<std::mutex> lock(*iter->mtx, std::try_to_lock);
        if(!lock.owns_lock()) {
          ++iter;
          continue;
        }
        occupied -= iter->size();
        iter->swap_out();
        iter->in_mem = false;
      }
      ++iter;
    }
  }

public:

  LRU(const size_t capacity = 8ul * (1ul << 30)) : capacity(capacity) {}

  void put(T&& value) {
    Key key = value.get_key();

    std::lock_guard<std::mutex> lock(mtx);

    auto it = k_v.find(key);
    if(it != k_v.end()) {
      lru_list.erase(it->second);
      k_v.erase(it);
    }

    std::size_t need_size = value.size();
    ensure_space(need_size);
    if(occupied + need_size > capacity) {
      throw std::runtime_error("LRU cache is full");
    }
    occupied += need_size;

    auto iter = lru_list.insert(lru_list.begin(), std::move(value));
    try {
      k_v.emplace(key, iter);
    } catch(const std::exception& e) {
      lru_list.erase(iter);
      throw std::runtime_error("LRU cache insertion failed");
    }
  }

  std::optional<TRefLockPair<T>> get(const Key& key) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = k_v.find(key);
    if(it != k_v.end()) {
      T& value = *it->second;

      std::unique_lock<std::mutex> lock(*value.mtx, std::try_to_lock);
      if(!lock.owns_lock()) {
        return std::nullopt;
      }
      if(!value.in_mem) {
        value.swap_in();
        std::size_t need_size = value.size();
        ensure_space(need_size);
        if(occupied + need_size > capacity) {
          throw std::runtime_error("LRU cache is full");
        }
        occupied += need_size;
        value.in_mem = true;
      }
      lru_list.splice(lru_list.begin(), lru_list, it->second);
      return std::make_pair(std::ref(value), std::move(lock));
    }
    return std::nullopt;
  }
};
} // namespace Ortho
#endif