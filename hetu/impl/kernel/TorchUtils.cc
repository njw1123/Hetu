#include "hetu/impl/kernel/TorchUtils.h"
#include <hetu/core/dtype.h>
#include <cuda_runtime.h>
namespace hetu {
namespace impl {

at::ScalarType GetTorchDtype(const DataType& dtype) {
  if (dtype == DataType::FLOAT32) {
    return at::ScalarType::Float;
  } else if (dtype == DataType::BFLOAT16) {
    return at::ScalarType::BFloat16;
  } else if (dtype == DataType::INT16) {
    return at::ScalarType::Short;
  } else if (dtype == DataType::INT32) {
    return at::ScalarType::Int;
  } else if (dtype == DataType::INT64) {
    return at::ScalarType::Long;
  } else if (dtype == DataType::BOOL) {
    return at::ScalarType::Bool;
  } else if (dtype == DataType::FLOAT64) {
    return at::ScalarType::Double;
  } else{
    HT_RUNTIME_ERROR << "不支持的数据类型: " << static_cast<int>(dtype);
  }
  // 默认返回float32
  return at::ScalarType::Float;
}

at::Tensor TransNDArray2Tensor(const NDArray& ndarray) {
  // 获取NDArray的属性
  HT_LOG_TRACE << " Trans NDArray to Tensor begin ...";
  void* cuda_memory = ndarray->raw_data_ptr();
  // 将vector<int64_t>转换为c10::IntArrayRef
  c10::IntArrayRef shape = ndarray->shape();
  c10::IntArrayRef stride = ndarray->stride(); 
  // c10::Device device = ndarray->device().toTorchDevice();
  c10::Device torch_device(c10::DeviceType::CUDA, ndarray->device().index());
  at::ScalarType dtype = GetTorchDtype(ndarray->dtype());

  // 检查指针是否为空
  if (cuda_memory == nullptr) {
    HT_RUNTIME_ERROR << "Error: NDArray raw_data_ptr() returned null!";
    return at::Tensor();
  }
  

  // 设置 Tensor 选项
  auto options = at::TensorOptions().dtype(dtype).device(torch_device);
  at::Tensor tensor = torch::from_blob(
      cuda_memory,   // 使用新分配的内存
      shape,             // 形状
      stride,            // 步长 (必须是元素步长)
      [cuda_memory](void*){},  // 释放内存的deleter
      options            // Tensor 选项 (包含设备和类型)
  );
  
  HT_LOG_TRACE << "Trans NDArray to Tensor done ...";
  return tensor;
}

c10::cuda::CUDAStream GetTorchStream(const Stream& stream) {
    cudaStream_t cuda_stream_handle = hetu::impl::CUDAStream(stream).cuda_stream();
    return c10::cuda::getStreamFromExternal(cuda_stream_handle, stream.device().index());
}


}
}