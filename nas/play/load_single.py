#!/usr/bin/env python
import gc
import time
import random
from pathlib import Path
import tensorflow.keras as kf
import tensorflow as tf
import os

basedir = Path(os.environ.get("TMPDIR", "/tmp"))

# it doth protest too much me things; shutup tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(3)
tf.get_logger().setLevel("ERROR")


def random_path(format=".h5"):
    random.seed(time.perf_counter_ns())
    return Path(
        "".join([chr(random.randint(ord("a"), ord("z"))) for i in range(6)]) + format
    )

def build_candle(name_first, name_rest_base):
    i = 0

    def name_rest():
        nonlocal i
        i += 1
        return f"{name_rest_base}_{i}"

    model = kf.Sequential()
    model.add(kf.Input(shape=(60483, 1)))
    model.add(kf.layers.Conv1D(filters=128, kernel_size=20, strides=1, padding="valid", name=name_first))
    model.add(kf.layers.Activation("relu", name=name_rest()))
    model.add(kf.layers.MaxPooling1D(pool_size=1, name=name_rest()))
    model.add(kf.layers.Conv1D(filters=128, kernel_size=10, strides=1, padding="valid", name=name_rest()))
    model.add(kf.layers.Activation("relu", name=name_rest()))
    model.add(kf.layers.MaxPooling1D(pool_size=10, name=name_rest()))
    model.add(kf.layers.Flatten(name=name_rest()))
    model.add(kf.layers.Dense(200, name=name_rest()))
    model.add(kf.layers.Activation("relu", name=name_rest()))
    model.add(kf.layers.Dropout(0.1, name=name_rest()))
    model.add(kf.layers.Dense(20, name=name_rest()))
    model.add(kf.layers.Activation("relu", name=name_rest()))
    model.add(kf.layers.Dropout(0.1, name=name_rest()))
    model.add(kf.layers.Dense(2, name=name_rest()))
    model.add(kf.layers.Activation("softmax", name=name_rest()))
    model.compile()
    return model


basefile = basedir / "base.h5"
for by_name in [True, False]:
    for i in range(10):
        model_base = build_candle("first", "rest")
        model_base.save_weights(basefile)
        os.sync()
        del model_base
        gc.collect()

        model_one_diff = build_candle("different", "rest")
        begin = time.perf_counter()
        model_one_diff.load_weights(basefile, by_name=by_name)
        end = time.perf_counter()
        del model_one_diff
        gc.collect()
        print("model_one_diff", f"{by_name=}", end-begin)

        model_most_diff = build_candle("first", "different")
        begin = time.perf_counter()
        model_most_diff.load_weights(basefile, by_name=by_name)
        end = time.perf_counter()
        del model_most_diff
        gc.collect()
        print("model_most_diff", f"{by_name=}", end-begin)

        basefile.unlink()
