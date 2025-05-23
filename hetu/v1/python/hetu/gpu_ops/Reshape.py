from __future__ import absolute_import
from .Node import Op
from .._base import DNNL_LIB
from ..cpu_links import reshape as cpu_reshape
from ..gpu_links import array_reshape


class Array_ReshapeOp(Op):
    def __init__(self, node_A, output_shape, ctx=None):
        super().__init__(Array_ReshapeOp, [node_A], ctx)
        self.output_shape = output_shape
        self.splits = None
        self.dtype = node_A.dtype

    def compute(self, input_vals, output_val, stream_handle=None):

        assert(len(input_vals) == 1)
        input_size = 1
        for i in range(len(input_vals[0].shape)):
            input_size *= input_vals[0].shape[i]
        # check if there exists -1 in output_shape
        idx = -1
        cnt = 0
        output_size = 1
        output_shape = self.get_output_shape()
        for i in range(len(output_shape)):
            if(output_shape[i] == -1):
                idx = i
                cnt = cnt + 1
                assert(cnt != 2)
            output_size *= output_shape[i]

        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size // output_size
        output_shape = tuple(output_shape)
        if self.on_cpu:
            if DNNL_LIB['cpu_Reshape']:
                cpu_reshape(input_vals[0], output_val)
            else:
                output_val[:] = input_vals[0].asnumpy().reshape(output_shape)
        else:
            if self.inplace:
                input_vals[0].reshape(output_shape, output_val)
            else:
                array_reshape(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        return [array_reshape_gradient_op(self, output_grad, ctx=self.raw_ctx)]

    def infer_shape(self, input_shapes):

        assert (len(input_shapes) == 1)
        input_size = 1
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            input_size *= input_shape[i]
        self.input_shape = input_shape

        # check if there exists -1 in output_shape
        idx = -1
        cnt = 0
        output_size = 1
        output_shape = self.get_output_shape()
        for i in range(len(output_shape)):
            if(output_shape[i] == -1):
                idx = i
                cnt = cnt + 1
                assert(cnt != 2)
            output_size *= output_shape[i]
        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size // output_size
        output_shape = tuple(output_shape)
        return output_shape

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list

    def get_output_shape(self):
        output_shape = list(self.output_shape)
        if self.splits is not None:
            if self.raw_ctx is None or not self.raw_ctx.is_mp:
                self.splits = None
            else:
                for k, v in self.splits.items():
                    if output_shape[k] > 0:
                        output_shape[k] //= v
        return output_shape

    def reset_status(self):
        self.splits = None

    def reset_status(self):
        self.splits = None


class Array_Reshape_GradientOp(Op):
    def __init__(self, node_in, node_out, ctx=None):
        super().__init__(Array_Reshape_GradientOp, [node_out], ctx)
        self.node_in = node_in
        self.dtype = node_in.dtype

    def compute(self, input_vals, output_val, stream_handle=None):
        # the size of input_array
        shapeIn = self.node_in.input_shape
        if self.on_cpu:
            if DNNL_LIB['cpu_Reshape']:
                cpu_reshape(input_vals[0], output_val)
            else:
                output_val[:] = input_vals[0].asnumpy().reshape(shapeIn)
        else:
            if self.inplace:
                input_vals[0].reshape(shapeIn, output_val)
            else:
                array_reshape(input_vals[0], output_val, stream_handle)

    def gradient(self, output_grad):
        raise NotImplementedError

    def infer_shape(self, input_shapes):
        return self.node_in.input_shape

    def backward_hook(self, config):
        self.inplace = config.enable_lazy and self not in config.eval_node_list


def array_reshape_op(node, output_shape, ctx=None):
    """Reshapes an input array without copy.

    Parameters:
    ----
    node : Node
        Input variable.
    output_shape: tuple(int)
        Expected shape of the output array.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Array_ReshapeOp(node, output_shape, ctx=ctx)


def array_reshape_gradient_op(node_in, node_out, ctx=None):
    """Gradient of reshape operation.

    Parameters:
    ----
    node_in : Node
        Input node of reshape operation.
    node_out: Node
        Previous gradient node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Array_Reshape_GradientOp(node_in, node_out, ctx=ctx)
