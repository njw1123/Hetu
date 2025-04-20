#include "hetu/impl/memory/TorchMemoryPool.h"
#include "hetu/impl/stream/CUDAStream.h" // Include necessary stream implementations
#include <c10/cuda/CUDAGuard.h>     // For device guard
#include <sstream>                  // For summary formatting

namespace hetu {
namespace impl {


// bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size) {
//   auto caching_mempool = std::dynamic_pointer_cast<TorchMemoryPool>(GetMemoryPool(device));
//   return caching_mempool->AllocPtr(ptr, size);
// }

// void FreeFromCUDACache(const Device& device, void* ptr) {
//   auto caching_mempool = std::dynamic_pointer_cast<TorchMemoryPool>(GetMemoryPool(device));
//   caching_mempool->FreePtr(ptr);
// }


TorchMemoryPool::TorchMemoryPool(int device_id)
    : CUDAMemoryPool(device_id, "TorchGPUMemPool" + std::to_string(device_id)),
      _device_id(device_id) {
  std::cout << "Create TorchMemoryPool " << _device_id << std::endl;
  _data_ptr_info.reserve(8192); // Pre-allocate some space
  HT_LOG_DEBUG << "TorchMemoryPool created for device " << _device_id;
}

TorchMemoryPool::~TorchMemoryPool() {
    HT_LOG_DEBUG << "Destroying TorchMemoryPool for device " << _device_id;
}

// 从Torch分配器分配原始内存
bool TorchMemoryPool::AllocPtr(void*& ptr, size_t size) {
  if (this == nullptr) {
    std::cerr << "Error: TorchMemoryPool::AllocPtr called on a null object!" << std::endl;
    return false;
  }
  std::cout << "TorchMemoryPool::AllocPtr" << std::endl;
  std::cout << "_device_id: " << _device_id << std::endl;
  c10::cuda::CUDAGuard device_guard(_device_id);
  std::cout << "TorchMemoryPool::AllocPtr 1" << std::endl;
  try {
    HT_LOG_DEBUG << "Attempting raw_alloc on device " << _device_id << " with size " << size;
    ptr = c10::cuda::CUDACachingAllocator::raw_alloc(size);
    HT_LOG_DEBUG << "raw_alloc succeeded on device " << _device_id << ". Pointer: " << ptr;
    std::cout << "TorchMemoryPool::AllocPtr success" << std::endl;
    // c10::DataPtr data_ptr = c10::cuda::CUDACachingAllocator::allocate(size, _device_id);
    // ptr = data_ptr.get();
    return true;
  } catch (const c10::Error& e) {
    // 内存分配失败，清除错误并返回false
    HT_LOG_ERROR << "Torch CUDACachingAllocator failed to allocate "
                 << size << " bytes on device " << _device_id << ". Error: "
                 << e.what();
    (void)cudaGetLastError();
    return false;
  } catch (...) {
    // 处理其他未知异常
    HT_LOG_ERROR << "Torch CUDACachingAllocator failed with unknown exception.";
    (void)cudaGetLastError();
    return false;
  }
}

// 将原始内存释放回Torch分配器
void TorchMemoryPool::FreePtr(void* ptr) {
  if (ptr == nullptr) return;
  
  c10::cuda::CUDAGuard device_guard(_device_id);
  try {
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  } catch (const c10::Error& e) {
    HT_LOG_ERROR << "Failed to free memory in TorchMemoryPool: " << e.what();
  } catch (...) {
    HT_LOG_ERROR << "Failed to free memory in TorchMemoryPool: Unknown error";
  }
}


DataPtr TorchMemoryPool::AllocDataSpace(size_t num_bytes, const Stream& stream) {

  HT_VALUE_ERROR_IF(!stream.device().is_cuda())
    << "Cuda arrays must be allocated on cuda streams. Got " << stream;

  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<DataPtrId>(-1)};

  c10::cuda::CUDAGuard device_guard(_device_id);

  void* ptr = nullptr;
  try {
    ptr = c10::cuda::CUDACachingAllocator::raw_alloc(num_bytes);
    // c10::DataPtr data_ptr = c10::cuda::CUDACachingAllocator::allocate(num_bytes, _device_id);
    // ptr = data_ptr.get();
  } catch (const c10::Error& e) {
    HT_RUNTIME_ERROR << "Torch CUDACachingAllocator failed to allocate "
                 << num_bytes << " bytes on device " << _device_id << ". Error: "
                 << e.what();
    throw; 
  } catch (...) {
    HT_RUNTIME_ERROR << "Torch CUDACachingAllocator failed with unknown exception.";
    throw;
  }

  DataPtr data_ptr{ptr, num_bytes, device(), next_id()};
  data_ptr.is_new_malloc = true; 

  _data_ptr_info[data_ptr.id] = std::make_shared<TorchDataPtrInfo>(
    ptr,
    num_bytes,
    stream,
    data_ptr.id
  );  
  MarkDataSpaceUsedByStream(data_ptr, stream);

  return data_ptr;
}

DataPtr TorchMemoryPool::BorrowDataSpace(void* ptr, size_t num_bytes,
                                         DataPtrDeleter deleter,
                                         const Stream& stream) {
  HT_LOG_TRACE << "BorrowDataSpace: " << ptr << ", " << num_bytes << ", " << stream;
  HT_VALUE_ERROR_IF(ptr == nullptr)
      << "Borrowing an empty storage is not allowed";
  HT_VALUE_ERROR_IF(!deleter)
      << "Deleter must not be empty when borrowing storages";
  // Consider if stream validation is needed for borrowed GPU pointers

  if (num_bytes == 0)
    return DataPtr{nullptr, 0, device(), static_cast<DataPtrId>(-1)};
  Stream borrow_stream = stream.is_defined() ? stream : Stream(device(), kBlockingStream);
  DataPtr data_ptr{ptr, num_bytes, device(), next_id()};
  data_ptr.is_new_malloc = false; 

  _data_ptr_info[data_ptr.id] = std::make_shared<TorchDataPtrInfo>(
    ptr,
    num_bytes,
    borrow_stream,
    data_ptr.id
  );  

  MarkDataSpaceUsedByStream(data_ptr, borrow_stream);
  return data_ptr;
}

void TorchMemoryPool::FreeDataSpace(DataPtr data_ptr) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return;
  try
  {
    c10::cuda::CUDACachingAllocator::raw_delete(data_ptr.ptr);
  }
  catch(const std::exception& e)
  {
    HT_LOG_ERROR << "Failed to free data space: " << e.what();
  }
  catch(...)
  {
    HT_LOG_ERROR << "Failed to free data space: Unknown error";
  }
  _data_ptr_info.erase(data_ptr.id);
}



void TorchMemoryPool::EmptyCache() {
    c10::cuda::CUDAGuard device_guard(_device_id);
    try {
        c10::cuda::CUDACachingAllocator::emptyCache();
        HT_LOG_INFO << "Cleared Torch CUDACachingAllocator cache for device " << _device_id;
    } catch (const c10::Error& e) {
        HT_LOG_ERROR << "Failed to empty Torch cache for device " << _device_id << ": " << e.what();
    } catch (...) {
        HT_LOG_ERROR << "Unknown error emptying Torch cache for device " << _device_id;
    }
}

void TorchMemoryPool::MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                                const Stream& stream) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0 || stream.is_blocking())
    return;
  HT_LOG_TRACE << "MarkDataSpaceUsedByStream: " << data_ptr << ", " << stream;
  try {
    int device_idx = stream.device().index();
    c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
    c10::cuda::CUDAStream torch_stream = GetTorchStream(stream);

  // 标记该内存块被这个stream使用
    c10::DataPtr torch_data_ptr(data_ptr.ptr, c10::Device(c10::DeviceType::CUDA, device_idx));
    _data_ptr_info[data_ptr.id]->insert_used_stream(stream.pack());
    c10::cuda::CUDACachingAllocator::recordStream(torch_data_ptr, torch_stream);
  } catch (const std::exception& e) {
    HT_LOG_ERROR << "Failed to mark data space used by stream: " << e.what();
  } catch (...) {
    HT_LOG_ERROR << "Failed to mark data space used by stream: Unknown error";
  }
}

void TorchMemoryPool::MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                                 const Stream& stream) {
  if (stream.is_blocking() || data_ptrs.empty())
    return;

  for (const auto& data_ptr : data_ptrs) {
    MarkDataSpaceUsedByStream(data_ptr, stream);
  }
}

std::future<void> TorchMemoryPool::WaitDataSpace(DataPtr data_ptr, bool async) {
  if (data_ptr.ptr == nullptr || data_ptr.size == 0)
    return async ? std::async([]() {}) : std::future<void>();

  auto it = _data_ptr_info.find(data_ptr.id);
  HT_RUNTIME_ERROR_IF(it == _data_ptr_info.end())
    << "Cannot find data " << data_ptr << " from info";
  PackedStreamId alloc_stream = it->second->alloc_stream;
  auto& used_streams = it->second->used_streams;
  if (used_streams.empty()) {
    // This only happens when alloc_stream and all used_streams are blocking
    return async ? std::async([]() {}) : std::future<void>();
  }

  // TODO: Avoid synchronizing allocation and used streams again 
  // when freeing the memory. However, remember that it necessitates 
  // tracking whether each async waits has completed or not.
  Stream wait_stream;
  if (used_streams.size() == 1 && *used_streams.begin() == alloc_stream) {
    wait_stream = Stream::unpack(alloc_stream);
  } else {
    Stream join_stream(data_ptr.device, kJoinStream);
    for (auto& used_stream : used_streams) {
      CUDAEvent event(data_ptr.device, false);
      event.Record(Stream::unpack(used_stream));
      event.Block(join_stream);
    }
    wait_stream = join_stream;
  }

  if (async) {
    return std::async([wait_stream]() { CUDAStream(wait_stream).Sync(); });
  } else {
    CUDAStream(wait_stream).Sync();
    return std::future<void>();
  }
}


void TorchMemoryPool::PrintSummary() {
  c10::cuda::CUDAGuard device_guard(_device_id);
  std::stringstream ss;
  ss << name() << " (Device " << _device_id << ") Summary:\n";

  try {
    const auto stats = c10::cuda::CUDACachingAllocator::getDeviceStats(_device_id);
    ss << "  Aggregated Stats:\n";
    ss << "    Allocated: "
       << stats.allocated_bytes[static_cast<int>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current
       << " bytes\n";
    ss << "    Reserved: "
       << stats.reserved_bytes[static_cast<int>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current
       << " bytes\n";
    ss << "    Active: "
       << stats.active_bytes[static_cast<int>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].current
       << " bytes\n";
     ss << "    Peak Allocated: "
       << stats.allocated_bytes[static_cast<int>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].peak
       << " bytes\n";
     ss << "    Peak Reserved: "
       << stats.reserved_bytes[static_cast<int>(c10::cuda::CUDACachingAllocator::StatType::AGGREGATE)].peak
       << " bytes\n";
     ss << "    Num Alloc Retries: " << stats.num_alloc_retries << "\n";
     ss << "    Num OOMs: " << stats.num_ooms << "\n";
  } catch (const c10::Error& e) {
      ss << "  Failed to get LibTorch allocator stats: " << e.what() << "\n";
  } catch (...) {
       ss << "  Failed to get LibTorch allocator stats: Unknown error\n";
  }

  HT_LOG_INFO << ss.str();
}

// --- Registration ---
namespace {

bool ParseUseTorchMemoryPool() {
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
}

struct TorchMemoryPoolRegister {
  TorchMemoryPoolRegister() {
    if (!ParseUseTorchMemoryPool()) {
      return;
    }
    HT_LOG_INFO << "TorchMemoryPoolRegister constructor started.";
    int num_cuda_devices = 0;
    cudaError_t count_err = cudaGetDeviceCount(&num_cuda_devices);
    if (count_err != cudaSuccess) {
          num_cuda_devices = 0;
          HT_RUNTIME_ERROR << "CUDA not available or failed to get device count. TorchMemoryPool won't be registered.";
    }
    for (int i = 0; i < num_cuda_devices && i < HT_MAX_DEVICE_INDEX; ++i) {
      Device device(kCUDA, i);
      try {
        at::empty({1}, at::TensorOptions().device(at::kCUDA, i));
        RegisterMemoryPoolCtor(device, [i]() -> std::shared_ptr<MemoryPool> {
            return std::make_shared<TorchMemoryPool>(i);
        });
      } catch (const std::exception& e) {
        HT_LOG_ERROR << "Error during registration for device " << i << ": " << e.what();
      } catch (...) {
        HT_LOG_ERROR << "Unknown error during registration for device " << i;
      }
    }
  }
};

static TorchMemoryPoolRegister torch_memory_pool_register;

} // namespace

} // namespace impl
} // namespace hetu