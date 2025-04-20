#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void FusedLayerNormTorch(const NDArray& in_arr, const NDArray& ln_scale,
                        const NDArray& ln_bias, NDArray& mean_arr, NDArray& var_arr,
                        NDArray& out_arr, int64_t reduce_dims, 
                        float eps, const Stream& stream) {
  HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  HT_ASSERT_SAME_DEVICE(in_arr, ln_bias);
  HT_ASSERT_SAME_DEVICE(in_arr, mean_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  int device_idx = in_arr->device().index();

  // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  hetu::cuda::CUDADeviceGuard guard(device_idx);
  c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  c10::cuda::CUDAGuard device_guard(torch_device);
  c10::cuda::CUDAStream torch_stream = GetTorchStream(stream);
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto input = TransNDArray2Tensor(in_arr).to(torch_device);
  auto scale = TransNDArray2Tensor(ln_scale).to(torch_device);
  auto bias = TransNDArray2Tensor(ln_bias).to(torch_device);
  auto output = TransNDArray2Tensor(out_arr).to(torch_device);
  auto mean = TransNDArray2Tensor(mean_arr).to(torch_device);
  auto var = TransNDArray2Tensor(var_arr).to(torch_device);

  // 构建normalized_shape，包含最后reduce_dims个维度
  std::vector<int64_t> normalized_shape;
  auto input_sizes = input.sizes();
  int ndim = input_sizes.size();
  for (int i = 0; i < reduce_dims; i++) {
    normalized_shape.push_back(input_sizes[ndim - reduce_dims + i]);
  }

  HT_LOG_TRACE << "execute torch fused layer norm";
  at::native_layer_norm_out(
      output,               // 输出结果
      mean,                 // 输出均值
      var,                  // 输出方差
      input,                // 输入张量
      normalized_shape,     // 归一化形状，包含最后reduce_dims个维度
      scale,                // 权重
      bias,                 // 偏置
      eps                   // epsilon
  );
  HT_LOG_TRACE << "execute torch fused layer norm done";

  NDArray::MarkUsedBy({in_arr, ln_scale, ln_bias, mean_arr, var_arr, out_arr}, stream);
}

void FusedRMSNormTorch(const NDArray& in_arr, const NDArray& ln_scale,
                      NDArray& var_arr, NDArray& out_arr, int64_t reduce_dims, 
                      float eps, const Stream& stream) {
  HT_RUNTIME_ERROR << "FusedRMSNormTorch is not implemented";
  // HT_ASSERT_CUDA_DEVICE(in_arr);
  // HT_ASSERT_SAME_DEVICE(in_arr, ln_scale);
  // HT_ASSERT_SAME_DEVICE(in_arr, var_arr); 
  // HT_ASSERT_SAME_DEVICE(in_arr, out_arr);

  // int device_idx = in_arr->device().index();

  // // 设置CUDA和Stream守卫，确保操作在正确的设备上执行
  // hetu::cuda::CUDADeviceGuard guard(device_idx);
  // c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  // c10::cuda::CUDAGuard device_guard(torch_device);
  // c10::cuda::CUDAStream torch_stream = GetTorchStream(stream);
  // c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  // auto input = TransNDArray2Tensor(in_arr).to(torch_device);
  // auto scale = TransNDArray2Tensor(ln_scale).to(torch_device);
  // auto output = TransNDArray2Tensor(out_arr).to(torch_device);
  // auto var = TransNDArray2Tensor(var_arr).to(torch_device);

  // // 构建normalized_shape，包含最后reduce_dims个维度
  // std::vector<int64_t> normalized_shape;
  // auto input_sizes = input.sizes();
  // int ndim = input_sizes.size();
  // for (int i = 0; i < reduce_dims; i++) {
  //   normalized_shape.push_back(input_sizes[ndim - reduce_dims + i]);
  // }  


  // HT_LOG_TRACE << "execute torch fused rms norm";
  // auto output = at::rms_norm(
  //     input,
  //     normalized_shape,
  //     scale,
  //     eps
  // );
  // HT_LOG_TRACE << "execute torch fused rms norm done";
  NDArray::MarkUsedBy({in_arr, ln_scale, var_arr, out_arr}, stream);
}

void FusedLayerNormGradientTorch(const NDArray& out_grads, const NDArray& in_arr,
                                const NDArray& ln_scale, const NDArray& ln_bias, NDArray& grad_arr,
                                NDArray& grad_scale, NDArray& grad_bias,
                                const NDArray& mean_arr, const NDArray& var_arr,
                                int64_t reduce_dims, float eps, bool inplace, const Stream& stream) {
  HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, mean_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
  HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);
  HT_ASSERT_SAME_DEVICE(out_grads, grad_bias);

  // 获取CUDA流
  int device_idx = out_grads->device().index();
  at::cuda::set_device(device_idx);
  c10::cuda::CUDAGuard device_guard(device_idx);
  c10::cuda::CUDAStream torch_stream = GetTorchStream(stream);
  // 设置CUDA设备守卫，确保操作在正确的设备上执行
  c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  auto out_grads_tensor = TransNDArray2Tensor(out_grads);
  auto in_arr_tensor = TransNDArray2Tensor(in_arr);
  auto ln_scale_tensor = TransNDArray2Tensor(ln_scale);
  auto ln_bias_tensor = TransNDArray2Tensor(ln_bias);
  auto grad_arr_tensor = TransNDArray2Tensor(grad_arr);
  auto grad_scale_tensor = TransNDArray2Tensor(grad_scale);
  auto grad_bias_tensor = TransNDArray2Tensor(grad_bias);
  auto mean_arr_tensor = TransNDArray2Tensor(mean_arr);
  auto var_arr_tensor = TransNDArray2Tensor(var_arr);

  std::cout << "start execute torch fused layer norm gradient" << std::endl;
  
  // 创建输出掩码，表示我们需要计算所有三种梯度
  std::array<bool, 3> output_mask = {true, true, true};
  // 构建normalized_shape，包含最后reduce_dims个维度
  std::vector<int64_t> normalized_shape;
  auto input_sizes = in_arr_tensor.sizes();
  int ndim = input_sizes.size();
  for (int i = 0; i < reduce_dims; i++) {
    normalized_shape.push_back(input_sizes[ndim - reduce_dims + i]);
  }
  
  // 调用PyTorch的layer_norm_backward函数
  at::native_layer_norm_backward_out(
      grad_arr_tensor,    // 输入梯度输出
      grad_scale_tensor,  // 权重梯度输出
      grad_bias_tensor,   // 偏置梯度输出
      out_grads_tensor,   // 输出梯度
      in_arr_tensor,      // 输入张量
      normalized_shape,      // 归一化形状
      mean_arr_tensor,    // 均值
      var_arr_tensor,     // 方差
      ln_scale_tensor,    // 权重
      ln_bias_tensor,     // 偏置
      output_mask         // 输出掩码
  );

  std::cout << "execute torch fused layer norm gradient done" << std::endl;

  NDArray::MarkUsedBy({out_grads, in_arr, ln_scale, ln_bias, grad_arr,
                       grad_scale, grad_bias, mean_arr, var_arr}, stream);
}

// void FusedRMSNormGradientCuda(const NDArray& out_grads, const NDArray& in_arr,
//                               const NDArray& ln_scale, NDArray& grad_arr,
//                               NDArray& grad_scale, const NDArray& var_arr,
//                               int64_t reduce_dims, float eps, bool inplace, const Stream& stream) {
//   HT_ASSERT_CUDA_DEVICE(out_grads);
//   HT_ASSERT_SAME_DEVICE(out_grads, ln_scale);
//   HT_ASSERT_SAME_DEVICE(out_grads, in_arr);
//   HT_ASSERT_SAME_DEVICE(out_grads, var_arr); 
//   HT_ASSERT_SAME_DEVICE(out_grads, grad_scale);
//   HT_ASSERT_SAME_DEVICE(out_grads, grad_arr);


//   NDArray::MarkUsedBy({out_grads, in_arr, ln_scale, grad_arr,
//                        grad_scale, var_arr}, stream);
// }

} // namespace impl
} // namespace hetu
