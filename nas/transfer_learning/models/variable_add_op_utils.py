import tensorflow as tf
from deephyper.nas.node import ConstantNode, VariableNode
from deephyper.nas.operation import operation, Zero, Connect, AddByProjecting, Identity

from deephyper.problem import NaProblem
from deephyper.nas import KSearchSpace

Activation = operation(tf.keras.layers.Activation)
Dense = operation(tf.keras.layers.Dense)
Dropout = operation(tf.keras.layers.Dropout)
Add = operation(tf.keras.layers.Add)
Flatten = operation(tf.keras.layers.Flatten)
MaxPool1D = operation(tf.keras.layers.MaxPool1D)
Dropout = operation(tf.keras.layers.Dropout)
Conv1D = operation(tf.keras.layers.Conv1D)


def add_mlp_op_(node, units_to_try, activations_to_try, rates_to_try):
    node.add_op(Identity())
    for unit in units_to_try:
        for activation in activations_to_try:
            node.add_op(Dense(units=unit, activation=activation))

    for rate in rates_to_try:
        node.add_op(Dropout(rate=rate))


def add_conv_op_(node, filter_sizes_to_try, num_filters_to_try, activations_to_try):
    for filter_size in filter_sizes_to_try:
        for num_filters in num_filters_to_try:
            for activation in activations_to_try:
                node.add_op(
                    Conv1D(
                        kernel_size=filter_size,
                        filters=num_filters,
                        padding="valid",
                        activation=activation,
                        strides=1,
                    )
                )
    # node.add_op(Identity())


def add_activation_op_(node, activations_to_try):
    for activation in activations_to_try:
        node.add_op(Activation(activation=activation))


def add_pooling_op_(node, pool_sizes_to_try, strides_to_try, padding='valid'):
    for pool_size in pool_sizes_to_try:
        for strides in strides_to_try:
            node.add_op(MaxPool1D(pool_size=pool_size, padding=padding, strides=strides))
    node.add_op(Identity())


def add_dense_op_(node, units_to_try):
    for unit in units_to_try:
        node.add_op(Dense(units=unit))


def add_dropout_op_(node, rates_to_try):
    for rate in rates_to_try:
        node.add_op(Dropout(rate=rate))
    node.add_op(Identity())
