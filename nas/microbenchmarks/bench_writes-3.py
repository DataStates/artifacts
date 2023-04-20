#!/usr/bin/env python
import sys
import gc
import time
import random
import argparse
from pathlib import Path
import shutil
import os
import concurrent.futures
import csv
import statistics
import multiprocessing



def random_path(format=".h5"):
    random.seed(time.perf_counter_ns())
    return Path(
        "".join([chr(random.randint(ord("a"), ord("z"))) for i in range(6)]) + format
    )


def cleanup(args, path):
    try:
        if args.format == ".h5":
            path.unlink()
        else:
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def run_task(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args['tensorflow_lv'])
    import tensorflow.keras as kf
    import tensorflow as tf
    if args['tensorflow_lv'] >= 3:
        tf.get_logger().setLevel("ERROR")
    args['device_id'] = tf.config.list_logical_devices("GPU")[args['task_id']].name

    def build_candle(num_changes=1):
        i = 0

        def name_rest():
            nonlocal i
            i += 1
            if num_changes >= i:
                return f"layer_{i}"
            else:
                return f"layer_new_{i}"

        model = kf.Sequential()
        model.add(kf.Input(shape=(60483, 1)))
        model.add(kf.layers.Conv1D(filters=128, kernel_size=20, strides=1, padding="valid", name=name_rest()))
        model.add(kf.layers.Activation("relu", name=name_rest()))
        model.add(kf.layers.MaxPooling1D(pool_size=1, name=name_rest()))
        model.add(kf.layers.Conv1D(filters=128, kernel_size=10, strides=1, padding="valid", name=name_rest()))
        model.add(kf.layers.Activation("relu", name=name_rest()))
        model.add(kf.layers.MaxPooling1D(pool_size=10, name=name_rest()))
        model.add(kf.layers.Flatten(name=name_rest()))
        model.add(kf.layers.Dense(3850, name=name_rest()))
        model.add(kf.layers.Activation("relu", name=name_rest()))
        model.add(kf.layers.Dropout(0.1, name=name_rest()))
        model.add(kf.layers.Dense(20, name=name_rest()))
        model.add(kf.layers.Activation("relu", name=name_rest()))
        model.add(kf.layers.Dropout(0.1, name=name_rest()))
        model.add(kf.layers.Dense(2, name=name_rest()))
        model.add(kf.layers.Activation("softmax", name=name_rest()))
        model.compile()
        return model


    def run_partial_writes(args):
        print("starting on ", args['device_id'], args['task_id'])
        with tf.device(args['device_id']):
            model_base = build_candle(args['num_changes'])
            barrier.wait()
            begin = time.perf_counter()
            model_base.get_weights()
            os.sync()
            end = time.perf_counter()
            del model_base
            gc.collect()
            print("stopping ", args['task_id'])
            return end-begin

    return run_partial_writes(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--num_changes", type=int, default=0)
    parser.add_argument("--format", choices=[".h5", ".tf"], default=".h5")
    parser.add_argument("--tensorflow_lv", type=int, default=3)
    parser.add_argument("--filesystem", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    return parser.parse_args()


def main():

    # it doth protest too much me things; shutup tensorflow
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(args.tensorflow_lv)

    fields = ["time_sec", "num_procs", "format", "num_changes", "successful", "result"]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()

    with concurrent.futures.ProcessPoolExecutor(args.num_procs) as ex:
        cleanup_paths = []
        futs = []
        for i in range(args.num_procs):
            outpath = args.filesystem / random_path(format=args.format)
            cleanup_paths.append(outpath)
            futs.append(ex.submit(run_task, {
                "num_changes": args.num_changes,
                "format": args.format,
                "outpath": outpath,
                "task_id": i,
                "tensorflow_lv": args.tensorflow_lv,
            }))
        barrier.wait()
        begin = time.perf_counter()
        concurrent.futures.wait(futs, return_when=concurrent.futures.ALL_COMPLETED)
        end = time.perf_counter()

        n_success = 0
        result = []
        for f in futs:
            try:
                result.append(f.result())
                n_success+=1
            except Exception as ex:
                print(ex, file=sys.stderr)

        writer.writerow({
            "time_sec": end-begin,
            "num_procs": args.num_procs,
            "format": args.format,
            "num_changes": args.num_changes,
            "successful": n_success,
            "result": statistics.mean(result)
        })
        for path in cleanup_paths:
            cleanup(args, path)

args = parse_args()
barrier = multiprocessing.Barrier(args.num_procs+1)

if __name__ == "__main__":
    main()
