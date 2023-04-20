import deephyper as dh
import tensorflow as tf
from deephyper.nas.operation._base import Operation


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
                    values[i] = dh.layers.Padding(paddings)(values[i])

        # concatenation
        if len(values) > 1:
            out = tf.keras.layers.Add()(values)
            if self.activation is not None:
                out = tf.keras.layers.Activation(self.activation)(out)
        else:
            out = values[0]
        return out
