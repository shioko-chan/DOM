#ifndef ORTHO_LRU_HPP
#define ORTHO_LRU_HPP

#include <list>
#include <mutex>
#include <unordered_map>

namespace Ortho {
template <typename T>
concept Hashable = requires(T a) {
  { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

template <typename K>
  requires Hashable<K>
struct CacheElem {
public:

  using key_type     = K;
  std::size_t size   = 0;
  bool        in_mem = false;

  void swap_in_() {
    swap_in();
    in_mem = true;
  }

  void swap_out_() {
    swap_out();
    in_mem = false;
  }

  virtual void            swap_in()       = 0;
  virtual void            swap_out()      = 0;
  virtual const key_type& get_key() const = 0;
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

  void release_space(const size_t size) {
    auto iter = lru_list.rbegin();
    while(occupied + size > capacity && iter != lru_list.rend()) {
      if(iter->in_mem) {
        iter->swap_out_();
        occupied -= iter->size;
      }
      ++iter;
    }
  }

public:

  LRU(const size_t capacity = 16 * (1 << 30)) : capacity(capacity) {}

  void put(T&& value) {
    auto& key = value.get_key();

    std::lock_guard<std::mutex> lock(mtx);

    auto it = k_v.find(key);
    if(it != k_v.end()) {
      lru_list.erase(it->second);
      k_v.erase(it);
    }

    release_space(value.size);
    if(occupied + value.size > capacity) {
      throw std::runtime_error("LRU cache is full");
    }

    auto iter = lru_list.insert(lru_list.begin(), std::move(value));
    try {
      k_v.emplace(key, iter);
    } catch(const std::exception& e) {
      lru_list.erase(iter);
      throw std::runtime_error("LRU cache insertion failed");
    }
    occupied += value.size;
  }

  std::optional<const T&> get(const Key& key) {
    std::lock_guard<std::mutex> lock(mtx);

    auto it = k_v.find(key);
    if(it != k_v.end()) {
      const T& value = *it->second;
      if(!value.in_mem) {
        value.swap_in_();
        occupied += value.size;
      }
      lru_list.splice(lru_list.begin(), lru_list, it->second);
      return value;
    }
    return std::nullopt;
  }
};
} // namespace Ortho
#endif