
#pragma once

#include "hetu/impl/communication/comm_group.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "hetu/impl/stream/CUDAStream.h"
#include "torch/torch.h"
#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>


namespace hetu {
namespace impl {

  at::ScalarType GetTorchDtype(const DataType& dtype);
  at::Tensor TransNDArray2Tensor(const NDArray& ndarray);
  NDArray TransTensor2NDArray(const at::Tensor& tensor);
  hetu::Device GetHetuDevice(const at::Device& device);
  c10::cuda::CUDAStream GetTorchCudaStream(const Stream& stream);
  c10::cuda::CUDAStream GetTorchCudaStream(const StreamIndex& stream_index);
  c10d::ReduceOp::RedOpType ToTorchRedType(ReductionType hetu_op);

}
}