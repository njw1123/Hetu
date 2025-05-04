#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h" // 使用TorchUtils
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h" // 保留 CUDADeviceGuard
#include <optional> // 用于可选的 guard

namespace hetu {
namespace impl {


// 实现 DataTransferTorch 函数
void DataTransferTorch(const NDArray& from, NDArray& to, const Stream& stream) {
  HT_ASSERT_SAME_SHAPE(from, to);
  size_t numel = from->numel();
  if (numel == 0) {
    return;
  }

  HT_ASSERT(stream.device().is_cuda())
      << "DataTransferTorch only supports CUDA streams";

  // 避免自复制（如果数据指针和设备相同）
  if (from->raw_data_ptr() == to->raw_data_ptr() && from->device() == to->device()) {
      if (from->dtype() == to->dtype()) {
          NDArray::MarkUsedBy({from, to}, stream); // 仍然标记为已使用
          return;
      }
  }

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = stream.device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto from_tensor = TransNDArray2Tensor(from);
  auto to_tensor = TransNDArray2Tensor(to);


  HT_LOG_TRACE << "execute torch data transfer from " << from->device() << " to " << to->device();
  to_tensor.copy_(from_tensor, /*non_blocking=*/true);
  HT_LOG_TRACE << "execute torch data transfer done";

  NDArray::MarkUsedBy({from, to}, stream);
}

} // namespace impl
} // namespace hetu
