#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"
#include <vector>
#include <optional>
#include <torch/torch.h>



namespace hetu {
namespace impl {



void FlashAttnTorch(
  const NDArray& q, // batch_size x seqlen_q x num_heads x head_size
  const NDArray& k, // batch_size x seqlen_k x num_heads_k x head_size
  const NDArray& v, // batch_size x seqlen_k x num_heads_k x head_size
  NDArray& out_, // batch_size x seqlen_q x num_heads x head_size
  NDArray& q_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& k_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& v_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& out_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& softmax_lse, // batch_size × num_heads × seqlen_q
  NDArray& p, // batch_size × num_heads × seqlen_q_rounded × seqlen_k_rounded
  NDArray& rng_state, // 2  kCUDA  kInt64
  const float p_dropout, const float softmax_scale, const bool is_causal,
  const bool return_softmax, const Stream& stream) {

  // 设置CUDA和Stream守卫
  int device_idx = q->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  HT_LOG_TRACE << "execute torch flash attention forward";

  auto q_tensor = TransNDArray2Tensor(q).to(torch_device);
  auto k_tensor = TransNDArray2Tensor(k).to(torch_device);
  auto v_tensor = TransNDArray2Tensor(v).to(torch_device);

  ::std::optional<double> scale_opt = softmax_scale;
  auto result_tuple = at::_flash_attention_forward(
      q_tensor,
      k_tensor,
      v_tensor,
      ::std::nullopt, // cum_seq_q: 非 Varlen 不需要
      ::std::nullopt, // cum_seq_k: 非 Varlen 不需要
      0,              // max_q: 非 Varlen 时忽略
      0,              // max_k: 非 Varlen 时忽略
      static_cast<double>(p_dropout),
      is_causal,
      return_softmax, // 映射到 return_debug_mask
      scale_opt       // 传递 softmax_scale
  );


  auto [out_tensor, logsumexp_tensor, philox_seed, philox_offset, std_ignore] = result_tuple;
  out_ = TransTensor2NDArray(out_tensor);
  softmax_lse = TransTensor2NDArray(logsumexp_tensor);
  rng_state->data_ptr<int64_t>()[0] = philox_seed.item<int64_t>();
  rng_state->data_ptr<int64_t>()[1] = philox_offset.item<int64_t>();

  HT_LOG_TRACE << "execute torch flash attention forward end";

  HT_LOG_TRACE << "Finished torch flash attention forward (non-varlen)";

  NDArray::MarkUsedBy({q, k, v, out_, q_padded, k_padded, v_padded,
                       softmax_lse, p, rng_state},
                      stream);
}


void FlashAttnVarlenTorch(
  const NDArray& q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
  const NDArray& k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& cu_seqlens_q, // b+1
  const NDArray& cu_seqlens_k, // b+1
  NDArray& out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
  NDArray& q_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& k_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& v_padded, // batch_size x seqlen_k x num_heads_k x head_size_rounded
  NDArray& out_padded, // batch_size x seqlen_q x num_heads x head_size_rounded
  NDArray& softmax_lse, // num_heads × total_q
  NDArray& p, // batch_size × num_heads × seqlen_q_rounded × seqlen_k_rounded
  NDArray& rng_state, // 2  kCUDA  kInt64
  const int max_seqlen_q, const int max_seqlen_k, const float p_dropout,
  const float softmax_scale, const bool zero_tensors, const bool is_causal,
  const bool return_softmax, const Stream& stream) { // zero_tensors 未使用

  // Set CUDA device and stream guards
  int device_idx = q->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  HT_LOG_TRACE << "Executing torch flash attention forward (varlen)";

  // Convert input NDArrays to Torch Tensors
  auto q_tensor = TransNDArray2Tensor(q).to(torch_device);
  auto k_tensor = TransNDArray2Tensor(k).to(torch_device);
  auto v_tensor = TransNDArray2Tensor(v).to(torch_device);

  // Ensure cu_seqlens are Int32
  HT_ASSERT(cu_seqlens_q->dtype() == kInt32 && cu_seqlens_k->dtype() == kInt32)
      << "cu_seqlens must be Int32 for torch flash attention varlen forward";
  auto cu_seqlens_q_tensor = TransNDArray2Tensor(cu_seqlens_q).to(torch_device);
  auto cu_seqlens_k_tensor = TransNDArray2Tensor(cu_seqlens_k).to(torch_device);

  // Prepare optional arguments
  ::std::optional<at::Tensor> cum_seq_q_opt = cu_seqlens_q_tensor;
  ::std::optional<at::Tensor> cum_seq_k_opt = cu_seqlens_k_tensor;
  ::std::optional<double> scale_opt = static_cast<double>(softmax_scale);

  // Call PyTorch _flash_attention_forward
  auto result_tuple = at::_flash_attention_forward(
      q_tensor,
      k_tensor,
      v_tensor,
      cum_seq_q_opt,
      cum_seq_k_opt,
      static_cast<int64_t>(max_seqlen_q),
      static_cast<int64_t>(max_seqlen_k),
      static_cast<double>(p_dropout),
      is_causal,
      return_softmax, // Maps to return_debug_mask in torch <= 2.1, ignored later
      scale_opt
      // Other optional args (window_size, alibi_slopes) use defaults
  );

  // Extract results and convert back to NDArray
  auto [out_tensor, logsumexp_tensor, philox_seed, philox_offset, debug_attn_mask] = result_tuple;
  out_ = TransTensor2NDArray(out_tensor);
  // PyTorch returns logsumexp with shape (num_heads, total_q) for varlen
  softmax_lse = TransTensor2NDArray(logsumexp_tensor);

  // Store RNG state (seed and offset)
  HT_ASSERT(rng_state->shape().size() == 1 && rng_state->shape(0) == 2 && rng_state->dtype() == kInt64)
      << "rng_state must have shape {2} and dtype Int64";
  rng_state->data_ptr<int64_t>()[0] = philox_seed.item<int64_t>();
  rng_state->data_ptr<int64_t>()[1] = philox_offset.item<int64_t>();

  NDArray::MarkUsedBy({q, k, v, cu_seqlens_q, cu_seqlens_k, out_, softmax_lse},
                      stream);
}

void FlashAttnGradientTorch(
  const NDArray& dout, // batch_size x seqlen_q x num_heads, x head_size_og
  const NDArray& q, // batch_size x seqlen_q x num_heads x head_size
  const NDArray& k, // batch_size x seqlen_k x num_heads_k x head_size
  const NDArray& v, // batch_size x seqlen_k x num_heads_k x head_size
  NDArray& out, // batch_size x seqlen_q x num_heads x head_size
  NDArray& softmax_lse, // b x h x seqlen_q
  NDArray& rng_state, // {2}, kInt64
  NDArray& dq_, // batch_size x seqlen_q x num_heads x head_size
  NDArray& dk_, // batch_size x seqlen_k x num_heads_k x head_size
  NDArray& dv_, // batch_size x seqlen_k x num_heads_k x head_size
  const float p_dropout, // probability to drop
  const float softmax_scale, const bool is_causal, const Stream& stream) {


  // 设置CUDA和Stream守卫
  int device_idx = q->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  HT_LOG_TRACE << "Executing torch flash attention backward (non-varlen)";

  // Convert input NDArrays to Torch Tensors
  auto dout_tensor = TransNDArray2Tensor(dout).to(torch_device);
  auto q_tensor = TransNDArray2Tensor(q).to(torch_device);
  auto k_tensor = TransNDArray2Tensor(k).to(torch_device);
  auto v_tensor = TransNDArray2Tensor(v).to(torch_device);
  auto out_tensor = TransNDArray2Tensor(out).to(torch_device);
  auto softmax_lse_tensor = TransNDArray2Tensor(softmax_lse).to(torch_device);
  auto rng_state_tensor = at::tensor({0, 0}, torch_device);
  rng_state_tensor[0] = rng_state->data_ptr<int64_t>()[0];
  rng_state_tensor[1] = rng_state->data_ptr<int64_t>()[1];
//   auto rng_state_tensor = TransNDArray2Tensor(rng_state).to(torch_device);

  // 准备可选参数
  ::std::optional<double> scale_opt = static_cast<double>(softmax_scale);
  at::Tensor unused_tensor; // Create an undefined tensor for unused optional args

  // Call PyTorch backward function (non-varlen version)
  auto grad_tuple = at::_flash_attention_backward(
      dout_tensor,
      q_tensor,
      k_tensor,
      v_tensor,
      out_tensor,
      softmax_lse_tensor,
      unused_tensor, // cum_seq_q: 非 Varlen 不需要
      unused_tensor, // cum_seq_k: 非 Varlen 不需要
      static_cast<int64_t>(q->shape(1)),              // max_q: 非 Varlen 时忽略
      static_cast<int64_t>(k->shape(1)),              // max_k: 非 Varlen 时忽略
      static_cast<double>(p_dropout),
      is_causal,
      rng_state_tensor,
      unused_tensor, // 传递 undefined tensor
      scale_opt
      // window_size_left, window_size_right 使用默认值 nullopt
  );

  // 从返回的元组中提取梯度 Tensor
  at::Tensor returned_dq = ::std::get<0>(grad_tuple);
  at::Tensor returned_dk = ::std::get<1>(grad_tuple);
  at::Tensor returned_dv = ::std::get<2>(grad_tuple);

  dq_ = TransTensor2NDArray(returned_dq);
  dk_ = TransTensor2NDArray(returned_dk);
  dv_ = TransTensor2NDArray(returned_dv);

  HT_LOG_TRACE << "Finished torch flash attention backward (non-varlen)";

  NDArray::MarkUsedBy(
    {dout, q, k, v, out, softmax_lse, rng_state, dq_, dk_, dv_}, stream);
}



void FlashAttnVarlenGradientTorch( // Renamed from FlashAttnVarlenGradientCuda
  const NDArray& dout, // total_q x num_heads x head_size
  const NDArray& q, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
  const NDArray& k, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& v, // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
  const NDArray& cu_seqlens_q, // b+1, int32
  const NDArray& cu_seqlens_k, // b+1, int32
  NDArray& out, // total_q x num_heads x head_size (forward pass output)
  NDArray& softmax_lse, // b x h x max_seqlen_q (logsumexp from forward) - NOTE: Shape assumption!
  NDArray& rng_state, // {2}, kInt64
  NDArray& dq_, // total_q x num_heads x head_size, gradient for q
  NDArray& dk_, // total_k x num_heads_k x head_size, gradient for k
  NDArray& dv_, // total_k x num_heads_k x head_size, gradient for v
  const int max_seqlen_q,
  const int max_seqlen_k, // max sequence length to choose the kernel
  const float p_dropout, // probability to drop
  const float softmax_scale, const bool zero_tensors, // zero_tensors unused in torch call
	const bool is_causal, const Stream& stream) {

  // Set CUDA device and stream guards
  int device_idx = q->device().index();
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  HT_LOG_TRACE << "Executing torch flash attention backward (varlen)";

  // Convert input NDArrays to Torch Tensors
  auto dout_tensor = TransNDArray2Tensor(dout).to(torch_device);
  auto q_tensor = TransNDArray2Tensor(q).to(torch_device);
  auto k_tensor = TransNDArray2Tensor(k).to(torch_device);
  auto v_tensor = TransNDArray2Tensor(v).to(torch_device);
  auto out_tensor = TransNDArray2Tensor(out).to(torch_device);

  // Ensure cu_seqlens are Int32
  HT_ASSERT(cu_seqlens_q->dtype() == kInt32 && cu_seqlens_k->dtype() == kInt32)
      << "cu_seqlens must be Int32 for torch flash attention varlen backward";
  auto cu_seqlens_q_tensor = TransNDArray2Tensor(cu_seqlens_q).to(torch_device);
  auto cu_seqlens_k_tensor = TransNDArray2Tensor(cu_seqlens_k).to(torch_device);

  // Verify softmax_lse shape (expected: num_heads x total_q)
  // Note: total_q = q->shape(0)
  HT_ASSERT(softmax_lse->shape().size() == 2 &&
            softmax_lse->shape(0) == q->shape(1) && // num_heads
            softmax_lse->shape(1) == q->shape(0))   // total_q
      << "softmax_lse shape mismatch for torch varlen backward. Expected ("
      << q->shape(1) << ", " << q->shape(0) << "), but got ("
      << softmax_lse->shape(0) << ", " << softmax_lse->shape(1) << ")";
  auto softmax_lse_tensor = TransNDArray2Tensor(softmax_lse).to(torch_device);
  auto rng_state_tensor = at::tensor({0, 0}, torch_device);
  rng_state_tensor[0] = rng_state->data_ptr<int64_t>()[0];
  rng_state_tensor[1] = rng_state->data_ptr<int64_t>()[1];


  // Prepare optional arguments
  ::std::optional<double> scale_opt = static_cast<double>(softmax_scale);
  at::Tensor unused_tensor; // Create an undefined tensor for unused optional args

  // Call PyTorch backward function (varlen version)
  auto grad_tuple = at::_flash_attention_backward(
      dout_tensor,
      q_tensor,
      k_tensor,
      v_tensor,
      out_tensor,
      softmax_lse_tensor, // Shape: num_heads x total_q
      cu_seqlens_q_tensor,
      cu_seqlens_k_tensor,
      static_cast<int64_t>(max_seqlen_q),
      static_cast<int64_t>(max_seqlen_k),
      static_cast<double>(p_dropout),
      is_causal,
      rng_state_tensor, // Pass the {seed, offset} tensor
      unused_tensor, // grad_scale: Pass undefined tensor
      scale_opt
      // window_size_left, window_size_right use default nullopt
  );

  // Extract gradient Tensors and convert back to NDArray
  at::Tensor returned_dq = ::std::get<0>(grad_tuple);
  at::Tensor returned_dk = ::std::get<1>(grad_tuple);
  at::Tensor returned_dv = ::std::get<2>(grad_tuple);


  dq_ = TransTensor2NDArray(returned_dq);
  dk_ = TransTensor2NDArray(returned_dk);
  dv_ = TransTensor2NDArray(returned_dv);

  HT_LOG_TRACE << "Finished torch flash attention backward (varlen)";

  // Mark dependencies
  NDArray::MarkUsedBy(
    {dout, q, k, v, cu_seqlens_q, cu_seqlens_k,
		 out, softmax_lse, rng_state, dq_, dk_, dv_}, stream);
}

} // namespace impl
} // namespace hetu
