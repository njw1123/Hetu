#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"


namespace hetu {
namespace impl {


void AddElewiseTorch(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_CUDA_DEVICE(inputB);
  HT_ASSERT_CUDA_DEVICE(output);

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = inputA->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto inputA_tensor = TransNDArray2Tensor(inputA).to(torch_device);
  auto inputB_tensor = TransNDArray2Tensor(inputB).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  at::add_out(output_tensor, inputA_tensor, inputB_tensor);

  NDArray::MarkUsedBy({inputA, inputB, output}, stream);

}

void SubElewiseTorch(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_CUDA_DEVICE(inputB);
  HT_ASSERT_CUDA_DEVICE(output);

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = inputA->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto inputA_tensor = TransNDArray2Tensor(inputA).to(torch_device);
  auto inputB_tensor = TransNDArray2Tensor(inputB).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  at::sub_out(output_tensor, inputA_tensor, inputB_tensor);

  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

void MulElewiseTorch(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_CUDA_DEVICE(inputB);
  HT_ASSERT_CUDA_DEVICE(output);

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = inputA->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);  

  auto inputA_tensor = TransNDArray2Tensor(inputA).to(torch_device);
  auto inputB_tensor = TransNDArray2Tensor(inputB).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  at::mul_out(output_tensor, inputA_tensor, inputB_tensor);

  NDArray::MarkUsedBy({inputA, inputB, output}, stream);
}

void DivElewiseTorch(const NDArray& inputA, const NDArray& inputB,
                    NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(inputA);
  HT_ASSERT_CUDA_DEVICE(inputB);
  HT_ASSERT_CUDA_DEVICE(output);

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = inputA->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto inputA_tensor = TransNDArray2Tensor(inputA).to(torch_device);
  auto inputB_tensor = TransNDArray2Tensor(inputB).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  at::div_out(output_tensor, inputA_tensor, inputB_tensor);

  NDArray::MarkUsedBy({inputA, inputB, output}, stream);  
  
  
}

} // namespace impl
} // namespace hetu
