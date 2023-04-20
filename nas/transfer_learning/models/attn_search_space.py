import tensorflow as tf
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import Concatenate, operation, Zero, Connect, AddByProjecting, Identity

from deephyper.problem import NaProblem
from deephyper.nas import KSearchSpace


import collections

Activation = operation(tf.keras.layers.Activation)
Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)
Add = operation(tf.keras.layers.Add)
Flatten = operation(tf.keras.layers.Flatten)
MaxPool1D = operation(tf.keras.layers.MaxPool1D)
Dropout = operation(tf.keras.layers.Dropout)
Conv1D = operation(tf.keras.layers.Conv1D)

class MultiInputsDenseSkipCoFactory(KSearchSpace):
    def __init__(self, input_shape, output_shape, size='small', num_layers=5, seed=None ):
        super().__init__(input_shape, output_shape, seed=seed)
        self.rel_size = size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_layers = num_layers
    
    
    def build(
        self,
        regression=True,
        num_layers=10,
        **kwargs,
    ):
        sub_graphs_outputs = []

        for input_ in self.input_nodes:
            output_sub_graph = self.build_sub_graph(input_)
            sub_graphs_outputs.append(output_sub_graph)

        cmerge = ConstantNode()
        cmerge.set_op(Concatenate(self, sub_graphs_outputs))

        output_sub_graph = self.build_sub_graph(cmerge)

        output = ConstantNode(op=Dense(2, "softmax"))
        self.connect(output_sub_graph, output)

        return self

    def build_sub_graph(self, input_, num_layers=3):
        source = prev_input = input_

        # look over skip connections within a range of the 3 previous nodes
        anchor_points = collections.deque([source], maxlen=3)

        for _ in range(num_layers):
            vnode = VariableNode()
            self.add_dense_to_(vnode)

            self.connect(prev_input, vnode)

            # * Cell output
            cell_output = vnode

            cmerge = ConstantNode()
            cmerge.set_op(AddByProjecting(self, [cell_output], activation="relu"))

            for anchor in anchor_points:
                skipco = VariableNode()
                skipco.add_op(Zero())
                skipco.add_op(Connect(self, anchor))
                self.connect(skipco, cmerge)

            prev_input = cmerge

            # ! for next iter
            anchor_points.append(prev_input)

        return prev_input

    def add_dense_to_(self, node):
        node.add_op(Identity())  # we do not want to create a layer in this case

        activations = [None, tf.nn.swish, tf.nn.relu, tf.nn.tanh, tf.nn.sigmoid]
        if self.rel_size == 'small':
            for units in range(50, 2000, 25):
                for activation in activations:
                    node.add_op(Dense(units=units, activation=activation))
        else:
            for units in range(2000, 3500, 50):
                for activation in activations:
                    node.add_op(Dense(units=units, activation=activation))

def create_attn_search_space(
    input_shape=[(8,), (10,)], output_shape=(1,), num_layers=5, size='small', **kwargs
):
    return MultiInputsDenseSkipCoFactory(
        input_shape, output_shape, num_layers=num_layers, size=size, **kwargs
    )

if __name__ == "__main__":
    space = create_attn_search_space()
    print(space.size)

