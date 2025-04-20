#pragma once

#include "hetu/impl/memory/CUDAMemoryPool.cuh"
#include "hetu/impl/memory/memory_manager.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/utils/task_queue.h"
#include "hetu/utils/emhash7_hashmap.h"
#include "hetu/utils/robin_hood_hashing.h"
#include <deque>
#include <map>
#include <c10/cuda/CUDACachingAllocator.h>
#include "hetu/impl/kernel/TorchUtils.h"

namespace hetu {
namespace impl {


// bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size);

// void FreeFromCUDACache(const Device& device, void* ptr);


class TorchMemoryPool final : public CUDAMemoryPool {
 public:
  // Constructor takes the specific CUDA device ID
  TorchMemoryPool(int device_id);

  ~TorchMemoryPool();

  // Disable copy and move
  TorchMemoryPool(const TorchMemoryPool&) = delete;
  TorchMemoryPool& operator=(const TorchMemoryPool&) = delete;
  TorchMemoryPool(TorchMemoryPool&&) = delete;
  TorchMemoryPool& operator=(TorchMemoryPool&&) = delete;

  // Allocate raw memory from Torch allocator
  bool AllocPtr(void*& ptr, size_t size);

  // Free raw memory back to Torch allocator
  void FreePtr(void* ptr);

  DataPtr AllocDataSpace(size_t num_bytes,
                         const Stream& stream = Stream()) override;

  DataPtr BorrowDataSpace(void* ptr, size_t num_bytes, DataPtrDeleter deleter,
                          const Stream& stream = Stream()) override;

  void FreeDataSpace(DataPtr data_ptr) override;

  void EmptyCache() override;

  void MarkDataSpaceUsedByStream(DataPtr data_ptr,
                                 const Stream& stream) override;

  void MarkDataSpacesUsedByStream(DataPtrList& data_ptrs,
                                  const Stream& stream) override;

  std::future<void> WaitDataSpace(DataPtr data_ptr, bool async = true) override;

  void PrintSummary() override;

  friend bool AllocAfterFreeFromCUDACache(const Device& device, void*& ptr, size_t size);

  friend void FreeFromCUDACache(const Device& device, void* ptr);

 private:
  // Record stream info of an allocated pointer.
  struct TorchDataPtrInfo {
    void* ptr;
    size_t num_bytes;
    PackedStreamId alloc_stream;
    std::unordered_set<PackedStreamId> used_streams;
    DataPtrId id;
    

    TorchDataPtrInfo(void* ptr_, size_t num_bytes_, const Stream& alloc_stream_,
                    DataPtrId id_)
    : ptr(ptr_),
      num_bytes(num_bytes_),
      alloc_stream(alloc_stream_.pack()),
      id(id_) {
      if (!alloc_stream_.is_blocking())
        used_streams.insert(alloc_stream);
    }

    inline void insert_used_stream(PackedStreamId used_stream) {
      if (used_stream != alloc_stream) {
        used_streams.insert(used_stream);
      }
    }
  };
  int _device_id;  
  emhash7::HashMap<DataPtrId, std::shared_ptr<TorchDataPtrInfo>> _data_ptr_info;
};

} // namespace impl
} // namespace hetu
