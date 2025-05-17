#ifndef SKYMERGE_MEM_HPP
#define SKYMERGE_MEM_HPP

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "config.hpp"
#include "ds/lru.hpp"

namespace SkyMerge {

template <std::uint64_t LIMIT>
class Mem {
public:

  template <typename SwapInFuncT, typename SwapOutFuncT>
    requires std::is_nothrow_invocable_r_v<ManagedPtr, SwapInFuncT>
             && std::is_nothrow_invocable_v<SwapOutFuncT, ManagedPtr>
  static void register_node(std::string_view key, ManagedPtr managed_ptr, SwapInFuncT swap_in, SwapOutFuncT swap_out) {
    mem.register_node(std::string{key}, std::move(managed_ptr), SwapInFunc{swap_in}, SwapOutFunc{swap_out});
  }

  static auto get_node(const std::string& key) noexcept -> std::optional<RefGuard> { return mem.get_node(key); }

  static void release_node_mem(const std::string& key) noexcept { mem.release_node_mem(key); }

  static auto contain_node(const std::string& key) noexcept -> bool { return mem.contain_node(key); }

private:

  static inline LRU mem{LIMIT};
};

using HostMem   = Mem<MEM_LIMIT>;
using DeviceMem = Mem<GPU_MEM_LIMIT>;

} // namespace SkyMerge

#endif