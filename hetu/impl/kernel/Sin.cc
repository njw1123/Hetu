#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/omp_utils.h"
#include "hetu/impl/stream/CPUStream.h"
#include <cmath>

namespace hetu {
namespace impl {

template <typename spec_t>
void sin_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::sin(input[idx]);
  }
}

void SinCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());

  size_t size = output->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "SinCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      sin_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                      output->data_ptr<spec_t>());
      }, "Sin");
      //cpu_stream.Sync();
    });
}

template <typename spec_t>
void cos_cpu(const spec_t* input, size_t size, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (size_t idx = 0; idx < size; ++idx) {
    output[idx] = std::cos(input[idx]);
  }
}

void CosCpu(const NDArray& input, NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(input);
  HT_ASSERT_SAME_DEVICE(input, output);
  HT_ASSERT_EXCHANGABLE(input, output);

  CPUStream cpu_stream(stream);
  dnnl::engine eng(dnnl::engine::kind::cpu, cpu_stream.stream_id());

  size_t size = input->numel();
  if (size == 0)
    return;
  HT_DISPATCH_INTEGER_AND_FLOATING_TYPES(
    input->dtype(), spec_t, "CosCpu", [&]() {
      auto _future = cpu_stream.EnqueueTask(
      [input, output, size]() {
      cos_cpu<spec_t>(input->data_ptr<spec_t>(), size,
                      output->data_ptr<spec_t>());
      }, "Cos");
      //cpu_stream.Sync();
    });
}

} // namespace impl
} // namespace hetu