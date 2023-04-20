import tensorflow as tf
from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import operation, Identity
from deephyper.nas import KSearchSpace
import abc
from .variable_add_op_utils import (
    add_conv_op_,
    add_activation_op_,
    add_pooling_op_,
    add_dense_op_,
    add_dropout_op_,
)

# from conv1d_operation import operation_conv1d
# from conv1d_operation import Conv1D


Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)
Flatten = operation(tf.keras.layers.Flatten)
Reshape = operation(tf.keras.layers.Reshape)


class NT3SearchSpace(KSearchSpace):
    def __init__(
        self,
        input_shape=(4, 1),
        output_shape=(2,),
        num_classes=2,
        problem_size="small",
        seed=None,
    ):
        super().__init__(input_shape, output_shape, seed=seed)
        self.num_classes = num_classes
        self.problem_size = problem_size

    def build(self, *args, **kwargs):
        problem_size = self.problem_size
        seed = kwargs.get("seed", None)
        inpt = self.input_nodes[0]

        n1 = VariableNode("N1ConvOp")
        if problem_size == "small":
            filter_sizes_to_try = range(3, 7)
            num_filters_to_try = [4, 8]
        else:
            filter_sizes_to_try = [15, 20, 25]
            num_filters_to_try = [48, 56, 60]

        activations_to_try = ["relu", "tanh"]
        # activations_to_try = ['relu', 'tanh', 'sigmoid']
        add_conv_op_(n1, filter_sizes_to_try, num_filters_to_try, activations_to_try)
        self.connect(inpt, n1)

        n2 = VariableNode("N2PoolingOp")

        if problem_size == "small":
            pool_sizes_to_try = range(2, 7)
            strides_to_try = range(2, 7)
            padding = "valid"
        else:
            pool_sizes_to_try = range(3, 4)
            strides_to_try = [4, 5, 6]
            # pool_sizes_to_try = range(1,4)
            # strides_to_try = [None, 1, 2]
            padding = "same"

        add_pooling_op_(n2, pool_sizes_to_try, strides_to_try, padding=padding)
        self.connect(n1, n2)

        if problem_size == "large":
            n3 = VariableNode("N3ConvOp")
            add_conv_op_(
                n3, filter_sizes_to_try, num_filters_to_try, activations_to_try
            )
            self.connect(n2, n3)
            n4 = VariableNode("N4PoolingOp")
            # pool_sizes_to_try = range(1, 10, 20)
            # strides_to_try = [None, 1, 2]
            pool_sizes_to_try = range(3, 5)
            strides_to_try = [4, 5, 6]
            add_pooling_op_(n4, pool_sizes_to_try, strides_to_try, padding="same")
            self.connect(n3, n4)

        n5 = ConstantNode(op=Flatten(), name="N5Flatten")
        if problem_size == "small":
            self.connect(n2, n5)
        else:
            self.connect(n4, n5)

        n6 = VariableNode("N6DenseOp")
        if problem_size == "small":
            units_to_try = [32, 97, 16]
        else:
            units_to_try = [100, 150, 200]
            # units_to_try = range(150, 450, 50)

        add_dense_op_(n6, units_to_try)
        self.connect(n5, n6)

        n7 = VariableNode("N7ActivOp")
        add_activation_op_(n7, activations_to_try)
        self.connect(n6, n7)

        n8 = VariableNode("N8DropoutOp")
        rates_to_try = [0.5, 0.4, 0.3, 0.2]
        # rates_to_try = [ 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        add_dropout_op_(n8, rates_to_try)
        self.connect(n7, n8)

        units_to_try = [16, 20, 24, 28, 32]
        # units_to_try = [ 80, 97, 16]
        n9 = VariableNode("N9DenseOp")
        add_dense_op_(n9, units_to_try)
        self.connect(n8, n9)

        n10 = VariableNode("N10ActivOp")
        add_activation_op_(n10, activations_to_try)
        self.connect(n9, n10)

        n11 = VariableNode("N11DropoutOp")
        add_dropout_op_(n11, rates_to_try)
        self.connect(n10, n11)

        output = ConstantNode(op=Dense(self.output_shape[0], activation="softmax"))

        self.connect(n11, output)
        return self


def create_nt3_search_space(input_shape, output_shape, problem_size="small", **kwargs):
    return NT3SearchSpace(
        input_shape, output_shape, problem_size=problem_size, **kwargs
    )
