#!/bin/bash
num_servers=$1
tasks=$2
RESULT_DIR=$3
FILE_DIR=$4
TOTAL_SIZE='4g'
VARIANCE=0
NUM_LAYERS=100

if [[ $PMI_RANK -ge $tasks ]]; then
        ./cpp-store/server \
                --thallium_connection_string "ofi+verbs"\
                --num_threads 1 \
                --num_servers $num_servers \
                --storage_backend "map" \
                --ds_colocated 0
else
	python -m mpi4py ./microbenchmarks/polaris_scale_store.py --allow_partial_stores --store_weights --partial_store --num_layers $NUM_LAYERS --total_size $TOTAL_SIZE --variance $VARIANCE
fi
echo $num_servers
