#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void ArraySetTorch(NDArray& data, double value, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(data);


  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  int device_idx = data->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  HT_LOG_TRACE << "execute torch array set";  
  auto data_tensor = TransNDArray2Tensor(data).to(torch_device);
  data_tensor.fill_(value);
  HT_LOG_TRACE << "execute torch array set end";

  NDArray::MarkUsedBy({data}, stream);
}

} // namespace impl
} // namespace hetu
