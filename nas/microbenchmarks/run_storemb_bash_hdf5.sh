#!/bin/bash
num_servers=$1
tasks=$2
RESULT_DIR=$3
FILE_DIR=$4
TOTAL_SIZE='4g'
VARIANCE=0
NUM_LAYERS=100


python -m mpi4py ./microbenchmarks/polaris_scale_hdf5.py --store_weights --partial_store --num_layers $NUM_LAYERS --total_size $TOTAL_SIZE --variance $VARIANCE --file_dir $FILE_DIR --result_dir $RESULT_DIR
echo $num_servers
