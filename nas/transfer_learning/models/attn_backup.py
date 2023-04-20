#!/usr/bin/env python

from deephyper.problem import NaProblem
from deephyper.nas.operation import operation
from deephyper.nas import KSearchSpace
import tensorflow.keras.layers as kf
from deephyper.nas.node import (
    ConstantNode,
    VariableNode
)

Input = operation(kf.Input)
Dense = operation(kf.Dense)
BatchNormalization = operation(kf.BatchNormalization)
Dropout = operation(kf.Dropout)
Multiply = operation(kf.Multiply)


class AttnSpace(KSearchSpace):
    def __init__(self, input_shape=(32, 32, 3), output_shape=(10,), seed=None):
        super().__init__(input_shape, output_shape, seed=seed)
        self.n_layers = 3

    def build(self, *args, **kwargs):
        source = self.input_nodes[0]

        d1 = VariableNode()
        d1.add_op(Dense(1000, activation="relu"))
        self.connect(source, d1)
        n1 = ConstantNode(op=BatchNormalization())
        self.connect(d1, n1)

        d2 = VariableNode()
        d1.add_op(Dense(1000, activation="relu"))
        self.connect(n1, d2)
        n2 = ConstantNode(op=BatchNormalization())
        self.connect(d2, n2)

        d3 = VariableNode()
        d1.add_op(Dense(500, activation="softmax"))
        self.connect(n2, d3)

        m = ConstantNode(op=Multiply([n2, d3]))
        self.connect(d3, m)

        last = m

        for i in range(self.n_layers):
            d = VariableNode()
            d.add_op(Dense(100, activation="relu"))
            self.connect(last, d)
            n = ConstantNode(op=BatchNormalization())
            self.connect(d, n)
            dr = ConstantNode(op=Dropout())
            self.connect(n, dr)
            last = dr

        d_last = VariableNode()
        d_last.add_op(Dense(2, activation="softmax"))
        self.connect(last, d_last)


        return self
