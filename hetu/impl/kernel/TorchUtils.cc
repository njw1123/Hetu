#include "hetu/impl/kernel/TorchUtils.h"
#include <hetu/core/dtype.h>
#include <cuda_runtime.h>
namespace hetu {
namespace impl {


// Helper to convert Hetu ReductionType to c10d::ReduceOp::RedOpType
c10d::ReduceOp::RedOpType ToTorchRedType(ReductionType hetu_op) {
    switch (hetu_op) {
        case kSUM: return c10d::ReduceOp::SUM;
        case kPROD: return c10d::ReduceOp::PRODUCT;
        case kMAX: return c10d::ReduceOp::MAX;
        case kMIN: return c10d::ReduceOp::MIN;
        case kMEAN: return c10d::ReduceOp::AVG; // Torch supports AVG
        case kNONE:
             HT_NOT_IMPLEMENTED << "Reduction type cannot be none";
             __builtin_unreachable();
        default:
            HT_NOT_IMPLEMENTED << "Reduction type " << hetu_op
                               << " is not supported for TorchNCCL.";
            __builtin_unreachable();
    }
}

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
  }else if (dtype == DataType::UINT8) { // Add this line
    return at::ScalarType::Byte;   
  }else{
    HT_RUNTIME_ERROR << "不支持的数据类型: " << static_cast<int>(dtype);
  }
  // 默认返回float32
  return at::ScalarType::Float;
}

at::Tensor TransNDArray2Tensor(const NDArray& ndarray) {
  // 获取NDArray的属性
  HT_LOG_TRACE << " Trans NDArray to Tensor begin ...";



  void* raw_memory = ndarray->raw_data_ptr();
  // 将vector<int64_t>转换为c10::IntArrayRef
  c10::IntArrayRef shape = ndarray->shape();
  c10::IntArrayRef stride = ndarray->stride(); 

  c10::Device torch_device = c10::Device(c10::DeviceType::CPU);
  if(ndarray->device().is_cuda()) {
    torch_device = c10::Device(c10::DeviceType::CUDA, ndarray->device().index());
  }

  at::ScalarType dtype = GetTorchDtype(ndarray->dtype());

  // 检查指针是否为空
  if (raw_memory == nullptr) {
    HT_RUNTIME_ERROR << "Error: NDArray raw_data_ptr() returned null!";
  }  

  // 设置 Tensor 选项
  auto options = at::TensorOptions().dtype(dtype).device(torch_device);
  at::Tensor tensor = torch::from_blob(
      raw_memory,   // 使用新分配的内存
      shape,             // 形状
      stride,            // 步长 (必须是元素步长)
      [raw_memory](void*){},  // 释放内存的deleter
      options            // Tensor 选项 (包含设备和类型)
  );
  
  HT_LOG_TRACE << "Trans NDArray to Tensor done ...";
  return tensor;
}

hetu::DataType GetHetuDtype(const at::ScalarType& dtype) {
  if (dtype == at::ScalarType::Float) {
    return DataType::FLOAT32;
  } else if (dtype == at::ScalarType::Double) {
    return DataType::FLOAT64;
  } else if (dtype == at::ScalarType::Short) {
    return DataType::INT16;
  } else if (dtype == at::ScalarType::Int) {
    return DataType::INT32;
  } else if (dtype == at::ScalarType::Long) {
    return DataType::INT64;
  } else if (dtype == at::ScalarType::Bool) {
    return DataType::BOOL;
  } else if (dtype == at::ScalarType::BFloat16) {
    return DataType::BFLOAT16;
  } else {
    // 使用 c10::toString 获取 ScalarType 的字符串表示形式
    HT_RUNTIME_ERROR << "不支持的 Torch 数据类型: " << c10::toString(dtype);
    // 返回一个默认值以满足编译器，尽管 HT_RUNTIME_ERROR 应该会终止程序
    return DataType::UNDETERMINED;
  }
}

hetu::Device GetHetuDevice(const c10::Device& device) {
  if (device.type() == c10::DeviceType::CPU) {
    return hetu::Device(hetu::DeviceType::CPU);
  } else if (device.type() == c10::DeviceType::CUDA) {
    return hetu::Device(hetu::DeviceType::CUDA, device.index());
  } else {
    HT_RUNTIME_ERROR << "不支持的 Torch 设备类型: " << c10::toString(device.type());
  }
}


hetu::NDArray TransTensor2NDArray(const at::Tensor& tensor) {
  // 获取NDArray的属性
  HT_LOG_TRACE << " Trans Tensor to NDArray begin ...";

  // 1. 获取 Tensor 属性
  const auto& torch_shape_ref = tensor.sizes();
  const auto& torch_stride_ref = tensor.strides();
  at::ScalarType torch_dtype = tensor.scalar_type();
  c10::Device torch_device = tensor.device();
  void* tensor_data_ptr = tensor.data_ptr(); // 指向张量数据的第一个元素

  at::Storage torch_storage = tensor.storage(); // 获取底层存储
  void* storage_data_ptr = torch_storage.data_ptr().get(); // 指向存储的起始位置
  size_t storage_offset_bytes = reinterpret_cast<uint8_t*>(tensor_data_ptr) - reinterpret_cast<uint8_t*>(storage_data_ptr);
  auto ndarray_storage = std::make_shared<hetu::NDArrayStorage>(torch_storage);


  // 2. 将 Torch 类型转换为 Hetu 类型
  hetu::DataType hetu_dtype = GetHetuDtype(torch_dtype); // 需要实现 GetHetuDtype
  hetu::Device hetu_device = GetHetuDevice(torch_device); // 需要实现 GetHetuDevice
  hetu::HTShape hetu_shape(torch_shape_ref.begin(), torch_shape_ref.end());
  // Torch stride 是元素数量的步长，与 Hetu 定义一致
  hetu::HTStride hetu_stride(torch_stride_ref.begin(), torch_stride_ref.end());

  // 计算以元素为单位的存储偏移量
  int64_t hetu_storage_offset = static_cast<int64_t>(storage_offset_bytes / DataType2Size(hetu_dtype));

  // 5. 直接创建 ndarray
  hetu::NDArrayMeta meta(hetu_shape, hetu_dtype, hetu_device, hetu_stride);
  auto ndarray = hetu::NDArray(meta, ndarray_storage, hetu_storage_offset);


  HT_LOG_TRACE << "Trans Tensor to NDArray done ...";
  // 函数的目标是返回 NDArray
  return ndarray;
}



c10::cuda::CUDAStream GetTorchCudaStream(const Stream& stream) {

  if(stream.stream_index() == 0) {
    auto torch_stream = c10::cuda::getDefaultCUDAStream();
    return torch_stream;
  }
  else {
    cudaStream_t cuda_stream_handle = hetu::impl::CUDAStream(stream).cuda_stream();
    // 判断cuda_stream_handle是否有效
    if (cuda_stream_handle == nullptr) {
        HT_RUNTIME_ERROR << "无效的CUDA流句柄: cuda_stream_handle为空";
    }
    HT_LOG_TRACE << "GetTorchCudaStream: 设备=" << stream.device().index() 
                << ", cuda_stream_handle=" << cuda_stream_handle;
    auto torch_stream = c10::cuda::getStreamFromExternal(cuda_stream_handle, stream.device().index());
    return torch_stream;
  }
}

c10::cuda::CUDAStream GetTorchCudaStream(const StreamIndex& stream_index) {
  auto& local_device = hetu::impl::comm::GetLocalDevice();
  return GetTorchCudaStream(Stream(local_device, stream_index));
}

}
}