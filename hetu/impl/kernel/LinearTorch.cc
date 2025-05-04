#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void LinearTorch(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
                const NDArray& bias, NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);

  size_t size = output->numel();
  if (size == 0) {
    return;
  }

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = a->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto a_tensor = TransNDArray2Tensor(a).to(torch_device);
  auto b_tensor = TransNDArray2Tensor(b).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  if (trans_a) {
    a_tensor = a_tensor.t(); // to [N, in_features]
  }
  if (!trans_b) {
    b_tensor = b_tensor.t(); // to [out_features, N]
  }

  HT_LOG_TRACE << "execute torch linear";
  if (bias.is_defined()) {
    auto bias_tensor = TransNDArray2Tensor(bias).to(torch_device);
    at::linear_out(output_tensor, a_tensor, b_tensor, bias_tensor);
  } else {
    at::linear_out(output_tensor, a_tensor, b_tensor);
  }
  HT_LOG_TRACE << "execute torch linear done";

  NDArray::MarkUsedBy({a, b, bias, output}, stream);
}

} // namespace impl
} // namespace hetu
