import argparse
import os
from pathlib import Path
from ..transfer_methods import transfer_methods
from ..search import search_methods


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rand_seed", type=int, default=2)
    parser.add_argument("--ray_head", default=None)
    parser.add_argument("--ncpus", type=int, default=None)
    parser.add_argument("--num_cpus_per_task", type=int, default=1)
    parser.add_argument("--num_gpus_per_task", type=int, default=None)
    parser.add_argument("--ngpus", type=int, default=None)
    parser.add_argument("--search_method", choices=search_methods(), default="expiring")
    parser.add_argument("--sample_size", type=int, default=3)
    parser.add_argument("--population_size", type=int, default=5)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--search_attempts", type=int, default=30)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--port", type=int, default=5700)
    parser.add_argument("--bulk_storage_path", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")))
    parser.add_argument(
        "--application",
        type=str,
        default="synthetic",
        choices=["synthetic", "uno", "nt3", "mnist", "attn", "combo"],
    )
    parser.add_argument(
        "--synthetic_space",
        type=str,
        default="feed_forward",
        choices=["feed_forward", "dense_skip_co"],
    )
    parser.add_argument(
        "--candle_problem_size", type=str, default="small", choices=["small", "large"]
    )
    parser.add_argument(
        "--attn_problem_size", type=str, default="small", choices=["small", "large"]
    )
    # NOTE: for now we assume the candle datasets have already been downloaded. See the candle website for download instructions.
    parser.add_argument("--test_data_path", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument(
        "--transfer_method", type=str, choices=transfer_methods(), default="noop"
    )
    parser.add_argument(
        "--save_result_dir", type=str, default=os.environ.get("TMPDIR", "/tmp/")
    )
    parser.add_argument(
        "--load_data_mode",
        type=str,
        default="mpicomm",
        help="where to load data from",
        choices=["mpicomm", "disk", "ray"],
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ds_colocated", action="store_true")
    parser.add_argument("--store_weights", action="store_true")
    args = parser.parse_args()

    assert args.beta is None or args.beta >= 1, "if not None, beta must be greater than or equal to 1"
    assert (
        args.population_size >= args.sample_size
    ), "population_size must exceed sample_size"
    return args
