import collections
import tensorflow as tf
import copy
import hashlib
import json


def _standardize_names(model):
    # first build the graph of the layers
    graph = collections.defaultdict(list)
    hashed_names = collections.defaultdict(str)
    counts = collections.defaultdict(int)
    if isinstance(model, tf.keras.Sequential):
        # functional model doesn't have 'inbound_nodes', simply iterate an build graph linearly
        for last_layer, current_layer in zip(model.layers, model.layers[1:]):
            graph[current_layer.name] = [last_layer.name]
    elif isinstance(model, tf.keras.Model):
        # assume instance of functional model
        model_config = model.get_config()
        for layer in model_config["layers"]:
            if len(layer["inbound_nodes"]) > 0:
                graph[layer["name"]] = [
                    inbound[0] for inbound in layer["inbound_nodes"][0]
                ]

    def inner(layer):
        # we can't assume that layers aren't shared, so force a copy
        layer_config = copy.deepcopy(layer.get_config())

        # we need the old name to lookup predecessors
        old_name = layer_config["name"]

        # we don't want to hash the name since this would make things recursively defined
        del layer_config["name"]

        if "trainable" in layer_config:
            trainable = layer_config["trainable"]
            layer_config["trainable"] = True

        # begin = time.perf_counter()

        layer_hash = hashlib.sha3_512()
        layer_hash.update(json.dumps(layer_config, sort_keys=True).encode())
        for pred_name in graph[old_name]:
            layer_hash.update(hashed_names[pred_name].encode())
        layer_hash = layer_hash.hexdigest()

        # end = time.perf_counter()

        base_name = layer.__class__.__name__ + layer_hash
        hashed_names[old_name] = base_name + "_" + str(counts[base_name])
        counts[base_name] += 1
        # create the new layer with the appropriate name
        layer_config["name"] = hashed_names[old_name]
        if "trainable" in layer_config:
            layer_config["trainable"] = trainable
        return layer.__class__.from_config(layer_config)

    return inner


def standardize_names(model: tf.keras.Model):
    return tf.keras.models.clone_model(
        model, input_tensors=model.inputs, clone_function=_standardize_names(model)
    )
