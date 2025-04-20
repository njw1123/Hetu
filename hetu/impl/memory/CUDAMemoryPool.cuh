#pragma once

#include "hetu/core/memory_pool.h"

namespace hetu {
namespace impl {

extern bool use_torch_memory_pool;


class CUDAMemoryPool : public MemoryPool {
 public:
  CUDAMemoryPool(DeviceIndex device_id, std::string&& name)
  : MemoryPool(Device(kCUDA, device_id), std::move(name)) {}

  ~CUDAMemoryPool() = default;

  inline size_t get_data_alignment() const noexcept {
    return 256;
  }
};


bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size);

void FreeFromCUDACache(const Device& device, void* ptr);

} // namespace impl
} // namespace hetu
