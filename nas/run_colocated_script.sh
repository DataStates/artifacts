#!/bin/bash
METHOD="datastates"
NUM_NODES=$1
RESULT_DIR=$2
NUMBER_CLIENT_PROCS=$((5*NUM_NODES))
if [[ $PMI_RANK -ge $NUMBER_CLIENT_PROCS ]]; then
	./cpp-store/server \
		--thallium_connection_string "ofi+verbs"\
	       	--num_threads 1 \
		--num_servers $NUM_NODES \
		--storage_backend "map" \
		--ds_colocated 0
else
	python transfer_learn-mpi.py \
        --rand_seed 4 \
        --bulk_storage_path $RESULT_DIR \
        --save_result_dir $RESULT_DIR \
        --transfer_method $METHOD \
        --load_data_mode mpicomm \
        --num_gpus_per_task 1 \
	--sample_size 5 \
	--population_size 50  \
	--search_attempts 1000 \
	--num_epochs 1 \
        --beta 1 \
        --application attn \
	--store_weights \
	--ds_colocated \
        --test_data_path "/lus/grand/projects/VeloC/datastates/training_attn.h5" \
	--attn_problem_size "large" \
        --train_data_path "/lus/grand/projects/VeloC/datastates/training_attn.h5"
fi
