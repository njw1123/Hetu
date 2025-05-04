#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/impl/memory/TorchMemoryPool.h"
#include "hetu/impl/memory/CUDACachingMemoryPool.cuh"

namespace hetu {
namespace impl {


bool use_torch_memory_pool = []() -> bool {
  const char* env_str = std::getenv("HETU_USE_TORCH_MEMORY_POOL");
  if (env_str != nullptr) {
    try {
      return std::stoi(env_str) != 0;
    } catch (const std::exception& e) {
      HT_LOG_ERROR << "Invalid HETU_USE_TORCH_MEMORY_POOL environment variable value: " << env_str 
                  << ", please provide an integer, default value will be used in this process.";
    }
  }
  return false;
}();



bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size) {
  if (use_torch_memory_pool) {
    auto caching_mempool = std::dynamic_pointer_cast<TorchMemoryPool>(GetMemoryPool(device));
    bool ret = caching_mempool->AllocPtr(ptr, size);
    return ret;
  } else {
    auto caching_mempool = std::dynamic_pointer_cast<CUDACachingMemoryPool>(GetMemoryPool(device));
    return caching_mempool->AllocPtr(ptr, size) || caching_mempool->WaitUntilAlloc(ptr, size);
  }
}

void FreeFromCUDACache(const Device& device, void* ptr) {
  if (use_torch_memory_pool) {
    auto caching_mempool = std::dynamic_pointer_cast<TorchMemoryPool>(GetMemoryPool(device));
    caching_mempool->FreePtr(ptr);
  } else {
    auto caching_mempool = std::dynamic_pointer_cast<CUDACachingMemoryPool>(GetMemoryPool(device));
    caching_mempool->FreePtr(ptr);
  }
}





}
}