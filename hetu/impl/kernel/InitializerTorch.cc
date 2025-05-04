#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

#include <ATen/Functions.h> // For ATen functions like normal_, uniform_, etc.
#include <ATen/Generator.h> // For at::Generator
#include <ATen/cuda/CUDAGeneratorImpl.h> // For CUDA generator creation/seeding
#include <c10/core/Scalar.h> // For at::scalar_tensor

namespace hetu {
namespace impl {

at::Generator GetTorchGenerator(int device_idx, uint64_t seed) {
  HT_ASSERT(seed != 0) << "seed must be non-zero";
  auto gen = at::cuda::detail::createCUDAGenerator(device_idx);
  gen.set_current_seed(seed);
  return gen;
}


void NormalInitsTorch(NDArray& data, double mean, double stddev, uint64_t seed,
                     const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  size_t size = data->numel();
  if (size == 0)
    return;


  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = data->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  // 转换NDArray到Tensor
  auto data_tensor = TransNDArray2Tensor(data).to(torch_device);
  auto generator = GetTorchGenerator(device_idx, seed);

  auto mean_scalar = c10::scalar_to_tensor(mean, torch_device);

  HT_LOG_TRACE << "execute torch normal init";
  auto mean_tensor = at::full_like(data_tensor, mean);
  at::normal_out(data_tensor, mean_tensor, stddev, generator);

  HT_LOG_TRACE << "execute torch normal init done";

  NDArray::MarkUsedBy({data}, stream);
}

// Renamed from UniformInitsCuda to UniformInitsTorch
void UniformInitsTorch(NDArray& data, double lb, double ub, uint64_t seed,
                      const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);
  HT_ASSERT(lb < ub) << "Invalid range for uniform random init: "
                     << "[" << lb << ", " << ub << ").";
  size_t size = data->numel();
  if (size == 0)
    return;

  // 设置CUDA和Stream守卫
  int device_idx = data->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto data_tensor = TransNDArray2Tensor(data).to(torch_device);
  auto generator = GetTorchGenerator(device_idx, seed);

  HT_LOG_TRACE << "execute torch uniform init";
  at::uniform(data_tensor, lb, ub, generator);
  HT_LOG_TRACE << "execute torch uniform init done";

  NDArray::MarkUsedBy({data}, stream);
}


void TruncatedNormalInitsTorch(NDArray& data, double mean, double stddev,
                              double lb, double ub, uint64_t seed,
                              const Stream& stream) {
  HT_NOT_IMPLEMENTED << "Torch TruncatedNormalInitsTorch not implemented";
}



} // namespace impl
} // namespace hetu
