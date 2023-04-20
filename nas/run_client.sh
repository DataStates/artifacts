#!/usr/bin/bash

method=$1
population=$2
output_dir=$3
size=$4
if [[ $5 -eq 1 ]]; then
    full_model="--full_model"
else
    full_model=""
fi

source ~/.bashrc
source init-dh-environment2.sh
#source combined.env

echo python -m mpi4py ./microbenchmarks/mock_model_bench_mpi.py --transfer_method=$method --trace_dir ./results/2022-08-05/GeneratedPopulation/${population}/orig --query_trace_dir  ./results/2022-08-05/GeneratedPopulation/${population}/orig --output_dir "${output_dir}" --models_10k=$size $full_model
python -m mpi4py ./microbenchmarks/mock_model_bench_mpi.py --transfer_method=$method --trace_dir ./results/2022-08-05/GeneratedPopulation/${population}/orig --query_trace_dir  ./results/2022-08-05/GeneratedPopulation/${population}/orig --output_dir "${output_dir}" --models_10k=$size
