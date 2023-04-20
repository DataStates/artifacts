#!/usr/bin/env python
import tensorflow.keras as kf
import tensorflow as tf
import argparse
import random
import os
import logging
import time
import gc
import csv
import sys
import shutil
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
LOGGER = logging.getLogger("save_bench.py")


def random_path(format=".h5"):
    random.seed(time.perf_counter_ns())
    return Path(
        "".join([chr(random.randint(ord("a"), ord("z"))) for i in range(6)]) + format
    )


def build_candle():
    model = kf.Sequential()
    model.add(kf.Input(shape=(60483, 1)))
    model.add(kf.layers.Conv1D(filters=128, kernel_size=20, strides=1, padding="valid"))
    model.add(kf.layers.Activation("relu"))
    model.add(kf.layers.MaxPooling1D(pool_size=1))
    model.add(kf.layers.Conv1D(filters=128, kernel_size=10, strides=1, padding="valid"))
    model.add(kf.layers.Activation("relu"))
    model.add(kf.layers.MaxPooling1D(pool_size=10))
    model.add(kf.layers.Flatten())
    model.add(kf.layers.Dense(200))
    model.add(kf.layers.Activation("relu"))
    model.add(kf.layers.Dropout(0.1))
    model.add(kf.layers.Dense(20))
    model.add(kf.layers.Activation("relu"))
    model.add(kf.layers.Dropout(0.1))
    model.add(kf.layers.Dense(2))
    model.add(kf.layers.Activation("softmax"))
    model.compile()
    return model


def build_resnet():
    model = kf.applications.ResNet152(include_top=True, weights=None, classes=200)

    # The ResNet failimy shipped with Keras is optimized for inference.
    # Add L2 weight decay & adjust BN settings.
    model_config = model.get_config()
    for layer, layer_config in zip(model.layers, model_config["layers"]):
        if hasattr(layer, "kernel_regularizer"):
            regularizer = kf.regularizers.l2()
            layer_config["config"]["kernel_regularizer"] = {
                "class_name": regularizer.__class__.__name__,
                "config": regularizer.get_config(),
            }
        if type(layer) == kf.layers.BatchNormalization:
            layer_config["config"]["momentum"] = 0.9
            layer_config["config"]["epsilon"] = 1e-5

    model = kf.models.Model.from_config(model_config)
    opt = kf.optimizers.SGD(learning_rate=0.01, momentum=0)
    model.compile(
        loss=kf.losses.categorical_crossentropy, optimizer=opt, metrics=["accuracy"]
    )
    return model


MODELS = {
    "candle": build_candle,
    "resnet": build_resnet,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", action="append", choices=list(MODELS), default=list(MODELS)
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp"))
    )
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--tf_loglevel", type=int, default=4)
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--outfile", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--format", choices=[".tf", ".h5"], default=".h5")
    parser.add_argument("--num_procs", type=int, default=1)
    return parser.parse_args()


def run_experiment(request):
    try:
        outdir = request["outdir"]
        format = request["format"]
        model_name = request["model_name"]
        outfile: Path = outdir / random_path(format)
        model = MODELS[model_name]()

        def compute_size():
            if format == ".h5":
                return outfile.stat().st_size
            else:
                total_size = 0
                for i in outfile.glob("**/*"):
                    if i.is_file():
                        total_size += i.stat().st_size
                return total_size

        file_save_begin = time.perf_counter()
        model.save(outfile, include_optimizer=False, save_traces=False)
        os.sync()
        file_save_end = time.perf_counter()
        del model
        gc.collect()

        file_load_begin = time.perf_counter()
        model = kf.models.load_model(outfile)
        file_load_end = time.perf_counter()
        del model
        gc.collect()

        response = {
            "model": model_name,
            "savetime_sec": file_save_end - file_save_begin,
            "loadtime_sec": file_load_end - file_load_begin,
            "filesize_bytes": compute_size(),
            "status": "ok",
        }
        return response
    finally:
        try:
            if format == ".h5":
                outfile.unlink()
            else:
                shutil.rmtree(outfile)
        except FileNotFoundError:
            pass


def main():
    args = parse_args()

    # it doth protest too much me things; shutup tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tf_loglevel)
    tf.get_logger().setLevel("ERROR")

    fields = ["model", "savetime_sec", "loadtime_sec", "filesize_bytes", "status"]
    csv_out = csv.DictWriter(args.outfile, fieldnames=fields)
    csv_out.writeheader()

    if not args.dryrun:
        with ProcessPoolExecutor(args.num_procs) as pool:
            for model_name in args.models:
                for i in range(args.reps):
                    futs = []
                    for i in range(args.num_procs):
                        futs.append(
                            pool.submit(
                                run_experiment,
                                {
                                    "format": args.format,
                                    "outdir": args.outdir,
                                    "model_name": model_name,
                                },
                            )
                        )
                    wait(futs, return_when=ALL_COMPLETED)
                    for fut in futs:
                        try:
                            result = fut.result()
                            csv_out.writerow(result)
                        except Exception as e:
                            csv_out.writerow({"status": str(e)})


if __name__ == "__main__":
    main()
