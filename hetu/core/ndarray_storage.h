#pragma once

#include "hetu/core/memory_pool.h"
#include "hetu/core/device.h"
#include <functional>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "torch/torch.h"
#include <ATen/ATen.h>

namespace hetu {

void ncclGroupStart_safe();
void ncclGroupEnd_safe();

class ncclGroupMemCtx {
 public:
  void Free() {
    for (const auto& ptr : _free_data_ptr_list) {
      FreeToMemoryPool(ptr);
    }
    _free_data_ptr_list.clear();
  } 

  void AddFreeDataPtr(const DataPtr& ptr) {
    _free_data_ptr_list.emplace_back(ptr);
  } 

 protected:
  std::vector<DataPtr> _free_data_ptr_list{};
};

class NDArrayStorage {
 public:
  NDArrayStorage(DataPtr ptr, bool in_mempool = true): 
    _ptr(ptr), 
    _in_mempool(in_mempool) {
  } 

  NDArrayStorage(at::Storage torch_storage){
    void* storage_data_ptr = torch_storage.data_ptr().get(); 
    size_t storage_nbytes = torch_storage.nbytes();
    hetu::Device hetu_device;
    if (torch_storage.device().type() == c10::DeviceType::CPU) {
      hetu_device = hetu::Device(hetu::DeviceType::CPU);
    } else if (torch_storage.device().type() == c10::DeviceType::CUDA) {
      hetu_device = hetu::Device(hetu::DeviceType::CUDA, torch_storage.device().index());
    } else {
      HT_RUNTIME_ERROR << "不支持的 Torch 设备类型: " << c10::toString(torch_storage.device().type());
    }
    _ptr = DataPtr(storage_data_ptr, storage_nbytes, hetu_device, 0);
    _in_mempool = false;
    _torch_storage = torch_storage;
    is_torch_storage = true;
  }

  ~NDArrayStorage();

  inline size_t size() const {
    return _ptr.size;
  }

  inline void* mutable_data() {
    return _ptr.ptr;
  }

  inline const void* data() const {
    return _ptr.ptr;
  }

  inline const Device& device() const {
    return _ptr.device;
  }

  inline Device& device() {
    return _ptr.device;
  }

  inline DataPtr data_ptr() const {
    return _ptr;
  }

  inline bool is_new_malloc() const {
    return _ptr.is_new_malloc;
  }

  // 将data_ptr的id暴露给外头并不是一个好的主意
  // 但为了更方便的debug只好暂时这么做
  inline uint64_t ptr_id() const {
    return _ptr.id;
  }

  inline uint64_t split_from_ptr_id() const {
    return _ptr.split_from_id;
  }

 protected:
  DataPtr _ptr;
  bool _in_mempool;
  bool _writable{true};
  bool is_torch_storage{false};
  at::Storage _torch_storage;

};

} // namespace hetu
