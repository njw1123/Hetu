
#pragma once

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "hetu/impl/stream/CUDAStream.h"
#include "torch/torch.h"
#include <ATen/ATen.h>

namespace hetu {
namespace impl {

  at::ScalarType GetTorchDtype(const DataType& dtype);
  at::Tensor TransNDArray2Tensor(const NDArray& ndarray);
  c10::cuda::CUDAStream GetTorchStream(const Stream& stream);

}
}