#pragma once

#include "hetu/graph/operator.h"
#include "hetu/graph/utils/tensor_utils.h"

namespace hetu {
namespace graph {

class AsStridedOpImpl;
class AsStridedOp;
class AsStridedGradientOpImpl;
class AsStridedGradientOp;

class AsStridedOpImpl : public OpInterface {

 public:
  AsStridedOpImpl(HTShape outshape, HTShape stride)
  : OpInterface(quote(AsStridedOp)),
  _outshape(outshape),
  _stride(stride){
  }

  inline HTShape outshape() const{
    return _outshape;
  }

  inline HTShape get_stride() const{
    return _stride;
  }

 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = NDArrayMeta().set_dtype(inputs[0]->dtype())
                                           .set_shape(outshape())
                                           .set_device(inputs[0]->device());
    return {output_meta};
  }

  TensorList DoGradient(Operator& op,
                        const TensorList& grad_outputs) const override;

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTShape _outshape;

  HTShape _stride;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AsStridedOpImpl&>(rhs);
      return (outshape() == rhs_.outshape()
              && get_stride() == rhs_.get_stride()); 
    }
    return false;
  }
};

Tensor MakeAsStridedOp(Tensor input, HTShape outshape, HTShape stride, OpMeta op_meta = OpMeta());

class AsStridedGradientOpImpl : public OpInterface {

 public:
  AsStridedGradientOpImpl(HTShape stride)
  : OpInterface(quote(AsStridedGradientOp)),
  _stride(stride) {
  }

  inline HTShape get_stride() const{
    return _stride;
  }


 protected:
  std::vector<NDArrayMeta> 
  DoInferMeta(const TensorList& inputs) const override {
    HT_ASSERT_TENSORS_SAME_DTYPE(inputs);
    NDArrayMeta output_meta = inputs[1]->meta();
    return {output_meta};
  }

  HTShapeList DoInferShape(Operator& op, const HTShapeList& input_shapes,
                           RuntimeContext& runtime_ctx) const override;

  void DoCompute(Operator& op, const NDArrayList& inputs, NDArrayList& outputs,
                 RuntimeContext& runtime_ctx) const override;
  HTShape _outshape;

  HTShape _stride;
 public:
  bool operator==(const OpInterface& rhs) const override {
    if (OpInterface::operator==(rhs)) {
      const auto& rhs_ = reinterpret_cast<const AsStridedGradientOpImpl&>(rhs);
      return (get_stride() == rhs_.get_stride()); 
    }
    return false;
  }

};

Tensor MakeAsStridedGradientOp(Tensor grad_output, Tensor input, HTShape stride,
                               OpMeta op_meta = OpMeta());

} // namespace graph
} // namespace hetu