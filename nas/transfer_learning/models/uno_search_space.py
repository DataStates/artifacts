import tensorflow as tf
from deephyper.nas import KSearchSpace
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import AddByPadding, Concatenate, operation, Identity

# from my_addpad_op import AddByPadding

from deephyper.nas import KSearchSpace
import abc
from variable_add_op_utils import (
    add_mlp_op_,
    add_conv_op_,
    add_activation_op_,
    add_pooling_op_,
    add_dense_op_,
    add_dropout_op_,
)


Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)
Flatten = operation(tf.keras.layers.Flatten)
Reshape = operation(tf.keras.layers.Reshape)


class UnoSearchSpace(KSearchSpace):
    def __init__(
        self,
        input_shape=[(1,), (942,), (5270,), (2048,)],
        output_shape=(1,),
        num_cells=2,
        seed=None,
    ):
        super().__init__(
            input_shape=input_shape,
            output_shape=output_shape,
            num_cells=num_cells,
            seed=seed,
        )
        self.num_cells = num_cells

    def build(self, *args, **kwargs):
        seed = kwargs.get("seed", None)
        input_nodes = self.input_nodes
        output_submodels = []
        output_submodels.append(input_nodes[0])

        num_units_to_try = [100, 500, 1000]
        rates_to_try = [0.3, 0.4, 0.5]
        activations_to_try = ["relu", "tanh", "sigmoid"]
        print(len(input_nodes))
        for i in range(1, 4):
            vnode1 = VariableNode("N1")
            add_mlp_op_(
                vnode1,
                units_to_try=num_units_to_try,
                activations_to_try=activations_to_try,
                rates_to_try=rates_to_try,
            )
            self.connect(input_nodes[i], vnode1)

            vnode2 = VariableNode("N2")
            add_mlp_op_(
                vnode2,
                units_to_try=num_units_to_try,
                activations_to_try=activations_to_try,
                rates_to_try=rates_to_try,
            )
            self.connect(vnode1, vnode2)

            vnode3 = VariableNode("N3")
            add_mlp_op_(
                vnode3,
                units_to_try=num_units_to_try,
                activations_to_try=activations_to_try,
                rates_to_try=rates_to_try,
            )
            self.connect(vnode2, vnode3)

            output_submodels.append(vnode3)

        merge1 = ConstantNode(name="Merge", op=Concatenate(self, output_submodels))

        vnode4 = VariableNode("N4")
        add_mlp_op_(
            vnode4,
            units_to_try=num_units_to_try,
            activations_to_try=activations_to_try,
            rates_to_try=rates_to_try,
        )
        self.connect(merge1, vnode4)

        prev = vnode4
        num_cells = self.num_cells
        for i in range(num_cells):
            vnode = VariableNode(f"N{i+1}")
            add_mlp_op_(
                vnode,
                units_to_try=num_units_to_try,
                activations_to_try=activations_to_try,
                rates_to_try=rates_to_try,
            )
            self.connect(prev, vnode)
            merge = ConstantNode(name="Merge", op=AddByPadding(self, [vnode, prev]))

            prev = merge

        output = ConstantNode(
            op=Dense(
                self.output_shape[0],
                activation=None,
                kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed),
            )
        )

        self.connect(prev, output)
        return self


def create_uno_search_space(
    input_shape=[(1,), (942,), (5270,), (2048,)],
    output_shape=(1,),
    num_cells=2,
    seed=None,
):
    return UnoSearchSpace(input_shape, output_shape, num_cells, seed)
