#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"


namespace hetu {
namespace impl {



void DataTransferTorch(const NDArray& from, NDArray& to, const Stream& stream) {
  HT_ASSERT_CUDA_DEVICE(from);
  HT_ASSERT_CUDA_DEVICE(to);



  HT_NOT_IMPLEMENTED << "DataTransferTorch is not implemented yet";

  NDArray::MarkUsedBy({from, to}, stream);
}

} // namespace impl
} // namespace hetu
