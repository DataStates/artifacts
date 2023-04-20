#!/usr/bin/env python

from .. import misc_utils
import copy
import tensorflow.keras.layers as layers
from deephyper.problem import NaProblem
from deephyper.nas import KSearchSpace
from deephyper.nas.node import (
    ConstantNode,
    VariableNode,
)
from deephyper.nas.operation import operation, Identity
from tensorflow.keras.datasets import mnist

"""
lenet like search space
"""
Input = layers.Input
Conv2D = operation(layers.Conv2D)
SeparableConv2D = operation(layers.SeparableConv2D)
AvgPool2D = operation(layers.AvgPool2D)
MaxPool2D = operation(layers.MaxPool2D)
Dense = operation(layers.Dense)
Dropout = operation(layers.Dropout)
Flatten = operation(layers.Flatten)
Activation = operation(layers.Activation)

X_train = None
X_test = None
Y_train = None
Y_test = None


def load_data2(prop=0.1):
    """Loads the MNIST dataset
    Returns Tuple of Numpy arrays: `(train_X, train_y), (valid_X, valid_y)`.
    """
    global X_test
    global X_train
    global Y_test
    global Y_train

    if X_train is None:
        (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

        # x_train = x_train.reshape(60000, 784)
        # x_test = x_test.reshape(10000, 784)
        # x_train = x_train.astype('float32')
        # x_test = x_test.astype('float32')
        # x_train /= 255
        # x_test /= 255
        # y_train = to_categorical(y_train, 10)
        # y_test = to_categorical(y_test, 10)
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    return (X_train, Y_train), (X_test, Y_test)


def add_conv_op_(node):
    node.add_op(Conv2D(kernel_size=3, filters=8, padding="valid"))
    node.add_op(Conv2D(kernel_size=4, filters=8, padding="valid"))

    node.add_op(Conv2D(kernel_size=3, filters=16, padding="valid"))
    node.add_op(Conv2D(kernel_size=4, filters=16, padding="valid"))

    node.add_op(Conv2D(kernel_size=3, filters=8, padding="same"))
    node.add_op(Conv2D(kernel_size=4, filters=8, padding="same"))

    node.add_op(Conv2D(kernel_size=3, filters=16, padding="same"))
    node.add_op(Conv2D(kernel_size=4, filters=16, padding="same"))


def add_dense_op_(node):
    node.add_op(Dense(units=32))
    node.add_op(Dense(units=64))
    node.add_op(Dense(units=96))
    node.add_op(Dense(units=128))
    node.add_op(Dense(units=256))
    node.add_op(Dense(units=512))


def add_activation_op_(node):
    node.add_op(Activation(activation="relu"))
    node.add_op(Activation(activation="tanh"))
    node.add_op(Activation(activation="sigmoid"))


def add_pooling_op_(node):
    node.add_op(Identity())
    node.add_op(MaxPool2D(2, padding="valid"))
    node.add_op(MaxPool2D(3, padding="valid"))
    #    node.add_op(MaxPool2D(4, padding='valid'))
    #    node.add_op(MaxPool2D(5, padding='valid'))

    node.add_op(MaxPool2D(2, padding="valid", strides=2))
    node.add_op(MaxPool2D(3, padding="valid", strides=3))


#    node.add_op(MaxPool2D(4, padding='valid', strides=4))
#    node.add_op(MaxPool2D(5, padding='valid', strides=5))


def add_dropout_op_(node):
    node.add_op(Identity())
    node.add_op(Dropout(rate=0.02))
    node.add_op(Dropout(rate=0.05))
    node.add_op(Dropout(rate=0.1))
    node.add_op(Dropout(rate=0.2))
    node.add_op(Dropout(rate=0.3))
    node.add_op(Dropout(rate=0.4))
    node.add_op(Dropout(rate=0.5))


"""
Example: 
https://d2l.ai/chapter_convolutional-neural-networks/lenet.html

"""


class MNISTSearchSpace(KSearchSpace):
    def __init__(self, input_shape=(10, 28, 28, 1), output_shape=(1,), seed=None, **kwargs):
        super().__init__(input_shape, output_shape, seed=seed)

    def build(self, *args, **kwargs):
        n1 = VariableNode("N1ConvOp")
        add_conv_op_(n1)
        self.connect(self.input_nodes[0], n1)

        n_conv_actv = VariableNode("N1ConvOp_actv")
        add_activation_op_(n_conv_actv)
        self.connect(n1, n_conv_actv)

        pool_node = VariableNode("pool_node")
        add_pooling_op_(pool_node)
        self.connect(n_conv_actv, pool_node)

        nconv2 = VariableNode("N1ConvOp2")
        add_conv_op_(nconv2)
        self.connect(pool_node, nconv2)

        n_conv_actv2 = VariableNode("N1ConvOp_actv2")
        add_activation_op_(n_conv_actv2)
        self.connect(nconv2, n_conv_actv2)

        pool_node2 = VariableNode("pool_node2")
        add_pooling_op_(pool_node2)
        self.connect(n_conv_actv2, pool_node2)

        n7 = ConstantNode(op=Flatten(), name="N7Flatten")
        self.connect(pool_node2, n7)

        n8 = VariableNode("N8DenseOp")
        add_dense_op_(n8)
        self.connect(n7, n8)

        n9 = VariableNode("N9ActivOp")
        add_activation_op_(n9)
        self.connect(n8, n9)

        n10 = VariableNode("N10DenseOp")
        add_dense_op_(n10)
        self.connect(n9, n10)

        n11 = VariableNode("N11ActivOp")
        add_activation_op_(n11)
        self.connect(n10, n11)

        n12 = VariableNode("DropoutLayer")
        add_dropout_op_(n12)
        self.connect(n11, n12)

        output = ConstantNode(op=Dense(1, activation="softmax"))

        self.connect(n12, output)

        return self


def create_mnist_search_space(input_shape, output_shape, **kwargs):
    return MNISTSearchSpace(input_shape, output_shape, **kwargs)


class MNISTProblem:
    def __init__(self, batch_size=100, num_epochs=50, download=False, **kwargs):
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def setup_problem(self, *args, **kwargs):
        Problem = NaProblem()

        Problem.load_data(load_data2)

        Problem.search_space(create_mnist_search_space)

        Problem.hyperparameters(
            batch_size=64, learning_rate=0.01, optimizer="adam", num_epochs=1, verbose=0
        )

        Problem.loss("categorical_crossentropy")

        Problem.metrics(["acc"])
        Problem.objective("val_acc__last")
        self.problem = Problem

        return self.problem

    def test_problem(self):
        if self.problem is None:
            self.setup_problem()
        problem = copy.deepcopy(self.problem)
        prob = problem.build_search_space()

        arch_seq = misc_utils.generate_a_random_archseq(prob)
        prob.set_ops(arch_seq)
        model = prob.create_model()
        model.summary()

    def get_baseline_model(self):
        return
