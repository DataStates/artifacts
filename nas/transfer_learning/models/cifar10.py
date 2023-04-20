#!/usr/bin/env python


from itertools import cycle
import tensorflow as tf
import deephyper.keras.layers
from tensorflow.keras.datasets.cifar10 import load_data as keras_load
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as layers
from deephyper.nas import KSearchSpace
from deephyper.nas.node import (
    ConstantNode,
    VariableNode,
    MimeNode,
)
from deephyper.nas.operation import Zero, Concatenate, Connect, Operation


class AddByPadding(Operation):
    """Add operation. If tensor are of different shapes a padding will be applied before adding them.

    Args:
        search_space (KSearchSpace): [description]. Defaults to None.
        activation ([type], optional): Activation function to apply after adding ('relu', tanh', 'sigmoid'...). Defaults to None.
        stacked_nodes (list(Node)): nodes to add.
        axis (int): axis to concatenate.
    """

    def __init__(self, search_space, stacked_nodes=None, activation=None, axis=-1):
        self.search_space = search_space
        self.node = None  # current_node of the operation
        self.stacked_nodes = stacked_nodes
        self.activation = activation
        self.axis = axis

    def init(self, current_node):
        self.node = current_node
        if self.stacked_nodes is not None:
            for n in self.stacked_nodes:
                self.search_space.connect(n, self.node)

    def __call__(self, values, **kwargs):
        # case where there is no inputs
        if len(values) == 0:
            return []

        values = values[:]
        max_len_shp = max([len(x.get_shape()) for x in values])

        # zeros padding
        if len(values) > 1:

            for i, v in enumerate(values):

                if len(v.get_shape()) < max_len_shp:
                    values[i] = tf.keras.layers.Reshape(
                        (
                            *tuple(v.get_shape()[1:]),
                            *tuple(1 for i in range(max_len_shp - len(v.get_shape()))),
                        )
                    )(v)

            def max_dim_i(i):
                return max(map(lambda x: int(x.get_shape()[i]), values))

            max_dims = [None] + list(map(max_dim_i, range(1, max_len_shp)))

            def paddings_dim_i(i):
                return list(map(lambda x: max_dims[i] - int(x.get_shape()[i]), values))

            paddings_dim = list(map(paddings_dim_i, range(1, max_len_shp)))

            for i in range(len(values)):
                paddings = list()
                for j in range(len(paddings_dim)):
                    p = paddings_dim[j][i]
                    lp = p // 2
                    rp = p - lp
                    paddings.append([lp, rp])
                if sum(map(sum, paddings)) != 0:
                    values[i] = deephyper.keras.layers.Padding(paddings)(values[i])

        # concatenation
        if len(values) > 1:
            out = tf.keras.layers.Add()(values)
            if self.activation is not None:
                out = tf.keras.layers.Activation(self.activation)(out)
        else:
            out = values[0]
        return out


from deephyper.nas.operation import operation, Identity
from tensorflow.nn import relu
import collections

Input = layers.Input
Conv2D = operation(layers.Conv2D)
SeparableConv2D = operation(layers.SeparableConv2D)
AvgPool2D = operation(layers.AvgPool2D)
MaxPool2D = operation(layers.MaxPool2D)
Dense = operation(layers.Dense)
Dropout = operation(layers.Dropout)


X_train = None
Y_train = None
X_test = None
Y_test = None


def load_data(with_test=True):
    global X_train
    global Y_train
    global X_test
    global Y_test
    if X_train is None:
        ((X_train, Y_train), (X_test, Y_test)) = keras_load()
        Y_train = to_categorical(Y_train, 10)
        Y_test = to_categorical(Y_test, 10)
        X_train = X_train.astype("float32")
        X_test = X_test.astype("float32")

    if with_test:
        return ((X_train, Y_train), (X_test, Y_test))
    else:
        return (X_train, Y_train)


normal_nodes = []
cycle_normal_nodes = cycle(normal_nodes)

reduction_nodes = []
cycle_reduction_nodes = cycle(reduction_nodes)


def generate_conv_node(strides, mime=False, first=False, num_filters=8):
    if mime:
        if strides > 1:
            node = MimeNode(next(cycle_reduction_nodes), name="Conv")
        else:
            node = MimeNode(next(cycle_normal_nodes), name="Conv")
    else:
        node = VariableNode(name="Conv")
        if strides > 1:
            reduction_nodes.append(node)
        else:
            normal_nodes.append(node)

    padding = "same"
    if first:
        node.add_op(Identity())
    else:
        node.add_op(Zero())
    node.add_op(Identity())
    node.add_op(
        Conv2D(
            filters=num_filters,
            kernel_size=(3, 3),
            strides=strides,
            padding=padding,
            activation=relu,
        )
    )
    node.add_op(
        Conv2D(
            filters=num_filters,
            kernel_size=(5, 5),
            strides=strides,
            padding=padding,
            activation=relu,
        )
    )
    node.add_op(AvgPool2D(pool_size=(3, 3), strides=strides, padding=padding))
    node.add_op(MaxPool2D(pool_size=(3, 3), strides=strides, padding=padding))
    node.add_op(
        SeparableConv2D(
            kernel_size=(3, 3), filters=num_filters, strides=strides, padding=padding
        )
    )
    node.add_op(
        SeparableConv2D(
            kernel_size=(5, 5), filters=num_filters, strides=strides, padding=padding
        )
    )
    if strides == 1:
        node.add_op(
            Conv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                strides=strides,
                padding=padding,
                dilation_rate=2,
            )
        )
        node.add_op(
            Conv2D(
                filters=num_filters,
                kernel_size=(5, 5),
                strides=strides,
                padding=padding,
                dilation_rate=2,
            )
        )
    return node


def generate_block(
    ss, anchor_points, strides=1, mime=False, first=False, num_filters=8
):

    # generate block
    n1 = generate_conv_node(
        strides=strides, mime=mime, first=first, num_filters=num_filters
    )
    n2 = generate_conv_node(strides=strides, mime=mime, num_filters=num_filters)
    add = ConstantNode(op=AddByPadding(ss, [n1, n2], activation=None))

    if first:
        source = anchor_points[-1]
        ss.connect(source, n1)

    if mime:
        if strides > 1:
            if not first:
                src_node = next(cycle_reduction_nodes)
                skipco1 = MimeNode(src_node, name="SkipCo1")
            src_node = next(cycle_reduction_nodes)
            skipco2 = MimeNode(src_node, name="SkipCo2")
        else:
            if not first:
                src_node = next(cycle_normal_nodes)
                skipco1 = MimeNode(src_node, name="SkipCo1")
            src_node = next(cycle_normal_nodes)
            skipco2 = MimeNode(src_node, name="SkipCo2")
    else:
        if not first:
            skipco1 = VariableNode(name="SkipCo1")
        skipco2 = VariableNode(name="SkipCo2")
        if strides > 1:
            if not first:
                reduction_nodes.append(skipco1)
            reduction_nodes.append(skipco2)
        else:
            if not first:
                normal_nodes.append(skipco1)
            normal_nodes.append(skipco2)
    for anchor in anchor_points:
        if not first:
            skipco1.add_op(Connect(ss, anchor))
            ss.connect(skipco1, n1)

        skipco2.add_op(Connect(ss, anchor))
        ss.connect(skipco2, n2)
    return add


def generate_cell(
    ss, hidden_states, num_blocks=5, strides=1, mime=False, num_filters=8
):
    anchor_points = [h for h in hidden_states]
    boutputs = []
    for i in range(num_blocks):
        bout = generate_block(
            ss,
            anchor_points,
            strides=1,
            mime=mime,
            first=i == 0,
            num_filters=num_filters,
        )
        anchor_points.append(bout)
        boutputs.append(bout)

    concat = ConstantNode(op=Concatenate(ss, boutputs))
    return concat


class CIFAR10SearchSpace(KSearchSpace):
    def __init__(self, input_shape=(32, 32, 3), output_shape=(10,), seed=None):
        super().__init__(input_shape, output_shape, seed=seed)
        self.repititions = 3
        self.reduction_cells = 1
        self.normal_cells = 2
        self.num_blocks = 4
        self.num_filters = 8

    def build(self, *args, **kwargs):
        source = self.input_nodes[0]
        hidden_states = collections.deque([source, source], maxlen=2)

        for ri in range(self.repititions):
            for nci in range(self.normal_cells):
                cout = generate_cell(
                    self,
                    hidden_states,
                    self.num_blocks,
                    strides=1,
                    mime=ri + nci > 0,
                    num_filters=self.num_filters,
                )
            hidden_states.append(cout)

            if ri < self.repititions - 1:
                for rci in range(self.reduction_cells):
                    cout = generate_cell(
                        self,
                        hidden_states,
                        self.num_blocks,
                        strides=2,
                        mime=ri + rci > 0,
                        num_filters=self.num_filters,
                    )
                hidden_states.append(cout)

        out_dense = VariableNode()
        out_dense.add_op(Identity())
        for units in [10, 20, 50, 100, 200, 500, 1000]:
            out_dense.add_op(Dense(units, activation=relu))
        self.connect(cout, out_dense)

        out_dropout = VariableNode()
        out_dropout.add_op(Identity())
        for drop_rate in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.8]:
            out_dropout.add_op(Dropout(rate=drop_rate))
        self.connect(out_dense, out_dropout)

        return self
