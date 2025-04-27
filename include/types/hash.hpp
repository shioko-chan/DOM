#ifndef ORTHO_HASH_HPP
#define ORTHO_HASH_HPP

#include <cstdint>
#include <functional>

namespace Ortho {

template <typename T>
void hash_append(uint64_t& seed, const T& val) noexcept {
  seed ^= std::hash<T>()(val) + 0x9e3779b9 + (seed << 6U) + (seed >> 2U);
}

template <typename... Args>
auto hash(const Args&... args) noexcept -> uint64_t {
  uint64_t seed = 0;
  (hash_append(seed, args), ...);
  return seed;
}

} // namespace Ortho

#endif