import model_client
import collections
import tensorflow as tf
import time

def build_model(args):
    # Start with a standard ResNet50 model
    model = tf.keras.applications.ResNet152(include_top=True, weights=None, classes=200)

    # The ResNet failimy shipped with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config['layers']):
        if hasattr(layer, 'kernel_regularizer'):
            regularizer = tf.keras.regularizers.l2(args.wd)
            layer_config['config']['kernel_regularizer'] = \
                {'class_name': regularizer.__class__.__name__,
                 'config': regularizer.get_config()}
        if type(layer) == tf.keras.layers.BatchNormalization:
            layer_config['config']['momentum'] = 0.9
            layer_config['config']['epsilon'] = 1e-5

    model = tf.keras.models.Model.from_config(model_config)
    opt = tf.keras.optimizers.SGD(learning_rate=args.base_lr, momentum=args.momentum)
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Args
    Args = collections.namedtuple("Args", "base_lr momentum wd")
    args = Args(0.0125, 0.9, 0.00005)

    model = build_model(args)
    repo = model_client.DataStatesModelRepo()
    repo.store(1, model, [])

    model2 = tf.keras.models.clone_model(model)
    repo.transfer(2, model2)
    repo.store(2, model, [model.get_layer('conv4_block3_2_bn')])

    repo.retire(1)
