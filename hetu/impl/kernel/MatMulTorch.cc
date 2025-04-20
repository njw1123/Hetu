#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void MatMulTorch(const NDArray& a, bool trans_a, const NDArray& b, bool trans_b,
               NDArray& output, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, b);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(b, 2);
  HT_ASSERT_NDIM(output, 2);
  HT_ASSERT_SAME_DTYPE(a, b);
  HT_ASSERT_SAME_DTYPE(a, output);


  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = a->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto a_tensor = TransNDArray2Tensor(a).to(torch_device);
  auto b_tensor = TransNDArray2Tensor(b).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  if (trans_a) {
    a_tensor = a_tensor.t();
  }
  if (!trans_b) {
    b_tensor = b_tensor.t();
  }

  HT_LOG_TRACE << "execute torch matmul";
  at::matmul_out(output_tensor, a_tensor, b_tensor);
  HT_LOG_TRACE << "execute torch matmul done";


  NDArray::MarkUsedBy({a, b, output}, stream);
}

} // namespace impl
} // namespace hetu
