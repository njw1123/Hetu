#include "hetu/core/ndarray.h"
#include "hetu/core/stream.h"
#include "hetu/impl/kernel/TorchUtils.h"
#include "hetu/impl/utils/common_utils.h"
#include "hetu/impl/utils/cuda_utils.h"

namespace hetu {
namespace impl {

void SGDUpdateTorch(const NDArray& grad, NDArray& param, NDArray& velocity,
                   float lr, float momentum, bool nesterov,
                   const Stream& stream) {
  HT_NOT_IMPLEMENTED << "SGDUpdateCuda not implemented";
}


void AdamTorch(const NDArray& grad, NDArray& param, NDArray& mean,
              NDArray& variance, NDArray& step,
              float lr, float beta1, float beta2,
              float eps, float weight_decay, bool update_step,
              const Stream& stream) {
  HT_RUNTIME_ERROR << "AdamTorch not implemented";
  // HT_ASSERT_CUDA_DEVICE(grad);
  // HT_ASSERT_CUDA_DEVICE(param);
  // HT_ASSERT_CUDA_DEVICE(mean);
  // HT_ASSERT_CUDA_DEVICE(variance);
  // HT_ASSERT_SAME_DEVICE(grad, param);
  // HT_ASSERT_SAME_DEVICE(grad, mean);
  // HT_ASSERT_SAME_DEVICE(grad, variance);
  // HT_ASSERT(step->ndim() == 1 && step->shape(0) == 1) << "Step must be a 1-element 1D NDArray.";


  // size_t size = grad->numel();
  // if (size == 0)
  //   return;

  // std::cout << "grad: " << grad << std::endl;
  // std::cout << "param: " << param << std::endl;
  // std::cout << "mean: " << mean << std::endl;
  // std::cout << "variance: " << variance << std::endl;
  // std::cout << "step: " << step << std::endl;


  // // 设置CUDA和Stream守卫
  // int device_idx = grad->device().index();
  // hetu::cuda::CUDADeviceGuard guard(device_idx);
  // c10::Device torch_device(c10::DeviceType::CUDA, device_idx);
  // c10::cuda::CUDAGuard device_guard(torch_device);
  // c10::cuda::CUDAStream torch_stream = GetTorchCudaStream(stream);
  // c10::cuda::CUDAStreamGuard stream_guard(torch_stream);

  // // 转换NDArray到Tensor
  // auto grad_tensor = TransNDArray2Tensor(grad).to(torch_device);
  // auto param_tensor = TransNDArray2Tensor(param).to(torch_device);
  // auto mean_tensor = TransNDArray2Tensor(mean).to(torch_device);
  // auto variance_tensor = TransNDArray2Tensor(variance).to(torch_device);
  // auto step_tensor = TransNDArray2Tensor(step).to(torch_device).to(c10::kLong);


  // // 检查梯度类型是否与参数类型匹配
  // if (grad_tensor.scalar_type() != param_tensor.scalar_type()) {
  //     grad_tensor = grad_tensor.to(param_tensor.scalar_type());
  // }




  // HT_LOG_TRACE << "execute torch fused adam update";

  // // Prepare TensorLists for the fused kernel
  // at::TensorList self_list = {param_tensor};
  // at::TensorList grads_list = {grad_tensor};
  // at::TensorList exp_avgs_list = {mean_tensor};
  // at::TensorList exp_avg_sqs_list = {variance_tensor};
  // at::TensorList state_steps_list = {step_tensor};
  // // max_exp_avg_sqs is only used for AMSGrad, create an empty list if not using AMSGrad
  // std::vector<at::Tensor> max_exp_avg_sqs_vec;
  // at::TensorList max_exp_avg_sqs_list = max_exp_avg_sqs_vec;

  // std::cout << "param_tensor " << param_tensor.sum() << std::endl;
  // std::cout << "grad_tensor " << grad_tensor.sum() << std::endl;
  // std::cout << "mean_tensor " << mean_tensor.sum() << std::endl;
  // std::cout << "variance_tensor " << variance_tensor.sum() << std::endl;
  // std::cout << "step_tensor " << step_tensor.sum() << std::endl;
  // std::cout << "beta1 " << beta1 << std::endl;
  // std::cout << "beta2 " << beta2 << std::endl;
  // std::cout << "lr " << lr << std::endl;
  // std::cout << "weight_decay " << weight_decay << std::endl;
  // std::cout << "eps " << eps << std::endl;

  // at::_fused_adam_(
  //     /*self=*/ self_list, // Input parameters (param) - updated in-place
  //     /*grads=*/ grads_list, // Gradients (grad)
  //     /*exp_avgs=*/ exp_avgs_list, // Exponential moving averages of gradients (mean) - updated in-place
  //     /*exp_avg_sqs=*/ exp_avg_sqs_list, // Exponential moving averages of squared gradients (variance) - updated in-place
  //     /*max_exp_avg_sqs=*/ max_exp_avg_sqs_list, // Max exp_avg_sqs (for AMSGrad) - empty list
  //     /*state_steps=*/ state_steps_list, // Step counter - updated in-place
  //     /*lr=*/ static_cast<double>(lr), // Learning rate as double
  //     /*beta1=*/ static_cast<double>(beta1),
  //     /*beta2=*/ static_cast<double>(beta2),
  //     /*weight_decay=*/ static_cast<double>(weight_decay),
  //     /*eps=*/ static_cast<double>(eps),
  //     /*amsgrad=*/ false, // Assuming standard Adam, not AMSGrad
  //     /*maximize=*/ false, // Assuming minimizing loss
  //     /*grad_scale=*/ {}, // Optional grad_scale
  //     /*found_inf=*/ {} // Optional found_inf
  // );

  // CUDAStream(stream).Sync();
  // std::cout << "after update param_tensor " << param_tensor.sum() << std::endl;
  

  // if (update_step)
  //   step->data_ptr<int64_t>()[0] = (step->data_ptr<int64_t>()[0] + 1);
  
  // auto grad_fp32 = TransTensor2NDArray(grad_tensor);

  // HT_LOG_TRACE << "execute torch fused adam update done";

  // NDArray::MarkUsedBy({grad, param, mean, variance, step, grad_fp32}, stream);
}

} // namespace impl
} // namespace hetu
