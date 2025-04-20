#include "hetu/graph/ops/KLDivLoss.h"
#include "hetu/graph/headers.h"
#include "hetu/graph/ops/kernel_links.h"
#include <numeric>

namespace hetu {
namespace graph {

using KLDivOpImpl = KLDivLossOpImpl;
using KLDivGradOpImpl = KLDivLossGradientOpImpl;

void KLDivOpImpl::DoCompute(Operator& op,
                            const NDArrayList& inputs, NDArrayList& outputs,
                            RuntimeContext& ctx) const {
  NDArray::kldiv(inputs.at(0), inputs.at(1), reduction(), 
                 op->instantiation_ctx().stream_index, outputs.at(0));
}

TensorList KLDivOpImpl::DoGradient(Operator& op, const TensorList& grad_outputs) const {
  auto grad_input = op->requires_grad(0) ? MakeKLDivLossGradientOp(
                                          op->input(0), op->input(1), grad_outputs.at(0), reduction(),
                                          op->grad_op_meta().set_name(op->grad_name()))
                                        : Tensor();
  return {grad_input, Tensor()};
}

HTShapeList KLDivOpImpl::DoInferShape(Operator& op,
                                      const HTShapeList& input_shapes,
                                      RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  if (reduction() != kNONE)
    return {{1}};
  else 
    return {input_shapes.at(0)};
}

void KLDivOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                 const OpMeta& op_meta,
                                 const InstantiationContext& inst_ctx) const {
  const DistributedStates& ds_preds = inputs.at(0)->get_distributed_states();
  const DistributedStates& ds_labels = inputs.at(1)->get_distributed_states();
  HT_ASSERT(ds_preds.is_valid() && ds_labels.is_valid()
            && ds_preds.get_device_num() == ds_labels.get_device_num())
    << "KLDivOpImpl: distributed states for inputs tensor must be valid!";
  HT_ASSERT(ds_preds.get_dim(-2) == 1 && ds_labels.get_dim(-2) == 1)
    << "Inputs tensor shouldn't be partial!";
  HT_ASSERT(ds_preds.check_equal(ds_labels))
    << "Distributed states among preds and labels should be equal!";
  HT_ASSERT(ds_preds.check_max_dim(1))
    << "KLDiv only support data parallel!";
  outputs.at(0)->set_distributed_states(ds_preds);  
}

void KLDivGradOpImpl::DoCompute(Operator& op,
                                const NDArrayList& inputs, NDArrayList& outputs,
                                RuntimeContext& ctx) const {
  NDArray broadcasted =
    reduction() == kNONE ? inputs.at(2) : NDArray::empty_like(inputs.at(0));
  if (reduction() == kMEAN) {
    HT_DISPATCH_HETU_KERNEL_CPU_AND_CUDA(
      op->instantiation_ctx().placement.type(), type(), hetu::impl::BroadcastShapeMul, inputs.at(2),
      1.0f / broadcasted->numel(), broadcasted, HTAxes(), op->instantiation_ctx().stream());
  } else if (reduction() == kSUM) {
    HT_DISPATCH_HETU_KERNEL_CPU_AND_CUDA(op->instantiation_ctx().placement.type(), type(),
                                    hetu::impl::BroadcastShape, inputs.at(2),
                                    broadcasted, HTAxes(), op->instantiation_ctx().stream());
  }
  HT_DISPATCH_HETU_KERNEL_CPU_AND_CUDA(
    op->instantiation_ctx().placement.type(), type(), hetu::impl::KLDivLossGradient,
    inputs.at(0), inputs.at(1), broadcasted, outputs.at(0), op->instantiation_ctx().stream());
}


HTShapeList KLDivGradOpImpl::DoInferShape(Operator& op,
                                          const HTShapeList& input_shapes,
                                          RuntimeContext& ctx) const {
  HT_ASSERT_GE(input_shapes.at(0).size(), 2)
    << "Invalid shape for " << type() << ": " << input_shapes.at(0);
  return {input_shapes.at(0)};
}

void KLDivGradOpImpl::DoDeduceStates(const TensorList& inputs, TensorList& outputs, 
                                 const OpMeta& op_meta,
                                 const InstantiationContext& inst_ctx) const {
  outputs.at(0)->set_distributed_states(inputs.at(0)->get_distributed_states());
}

Tensor MakeKLDivLossOp(Tensor preds, Tensor labels,
                       ReductionType reduction,
                       OpMeta op_meta) {
  TensorList inputs = {preds, labels};
  return Graph::MakeOp(
          std::make_shared<KLDivLossOpImpl>(reduction),
          std::move(inputs),
          std::move(op_meta))->output(0); 
}

Tensor MakeKLDivLossOp(Tensor preds, Tensor labels,
                       const std::string& reduction,
                       OpMeta op_meta) {
  TensorList inputs = {preds, labels};
  return Graph::MakeOp(
          std::make_shared<KLDivLossOpImpl>(Str2ReductionType(reduction)),
          std::move(inputs),
          std::move(op_meta))->output(0); 
}

Tensor MakeKLDivLossGradientOp(Tensor preds, Tensor labels, Tensor grad_output,
                               ReductionType reduction,
                               OpMeta op_meta) {
  return Graph::MakeOp(
          std::make_shared<KLDivLossGradientOpImpl>(reduction),
          {std::move(preds), std::move(labels), std::move(grad_output)},
          std::move(op_meta))->output(0); 
}

} // namespace graph
} // namespace hetu
