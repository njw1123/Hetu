#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

#include <ATen/Functions.h> // For at::sum_out, at::mean_out, etc.
#include <c10/util/ArrayRef.h> // For c10::IntArrayRef

namespace hetu {
namespace impl {

void ReduceTorch(const NDArray& input, NDArray& output, const HTAxes& axes,
                 ReductionType red_type, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_SAME_DTYPE(input, output);

  // 解析轴，处理负轴等情况
  HTAxes parsed_axes = NDArrayMeta::ParseAxes(axes, input->ndim());
  bool keepdim = (input->shape().size() == output->shape().size());

  // 设置CUDA和Stream守卫
  int device_idx = input->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  // 转换NDArray到Tensor
  auto input_tensor = TransNDArray2Tensor(input).to(torch_device);
  auto output_tensor = TransNDArray2Tensor(output).to(torch_device);

  // 转换HTAxes到c10::IntArrayRef
  c10::IntArrayRef torch_axes(parsed_axes);

  HT_LOG_TRACE << "execute torch reduce: " << static_cast<int32_t>(red_type);
  switch (red_type) {
    case kSUM:
      at::sum_out(output_tensor, input_tensor, torch_axes, keepdim);
      break;
    case kMEAN:
      at::mean_out(output_tensor, input_tensor, torch_axes, keepdim);
      break;
    case kMAX:
      at::amax_out(output_tensor, input_tensor, torch_axes, keepdim);
      break;
    case kMIN:
      at::amin_out(output_tensor, input_tensor, torch_axes, keepdim);
      break;
    case kPROD:
      HT_ASSERT(torch_axes.size() == 1) << "torch_axes.size() = " << torch_axes.size();
      at::prod_out(output_tensor, input_tensor, torch_axes.at(0));
      break;
    case kNONE:
      HT_NOT_IMPLEMENTED << "Reduction type cannot be none";
      __builtin_unreachable();
    default:
      HT_VALUE_ERROR << "Unknown reduction type: "
                     << static_cast<int32_t>(red_type);
      __builtin_unreachable();
  }
  HT_LOG_TRACE << "execute torch reduce done";

  NDArray::MarkUsedBy({input, output}, stream);
}

} // namespace impl
} // namespace hetu
