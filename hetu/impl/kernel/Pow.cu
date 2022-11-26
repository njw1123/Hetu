#include "hetu/core/ndarray.h"
#include "hetu/impl/stream/CUDAStream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include "hetu/impl/utils/cuda_math.h"

namespace hetu {
namespace impl {

template <typename spec_t>
__global__ void pow_kernel(const spec_t* input, spec_t exponent, size_t size,
                           spec_t* output) {
  auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size)
    return;
  output[idx] = hetu::cuda::cuda_pow(input[idx], exponent);
}

void PowCuda(const NDArray& input, double exponent, NDArray& output,
             const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  size_t size = output->numel();
  if (size == 0)
    return;
  dim3 blocks, threads;
  threads.x = MIN(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  blocks.x = DIVUP(size, HT_DEFAULT_NUM_THREADS_PER_BLOCK);
  CUDAStream cuda_stream(stream);
  hetu::cuda::CUDADeviceGuard guard(cuda_stream.device_id());
  HT_DISPATCH_FLOATING_TYPES(input->dtype(), spec_t, "PowCuda", [&]() {
    pow_kernel<spec_t><<<blocks, threads, 0, cuda_stream>>>(
      input->data_ptr<spec_t>(), static_cast<spec_t>(exponent), size,
      output->data_ptr<spec_t>());
  });
}

} // namespace impl
} // namespace hetu