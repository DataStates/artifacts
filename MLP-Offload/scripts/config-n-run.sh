#!/bin/bash -l

# set -x
# Define default values
model_size_B=0
HIDDEN_SIZE=0
FFN_HIDDEN_SIZE=0
NUM_LAYERS=0
NUM_HEADS=0
SEQ_LENGTH=0
NUM_KV_HEADS=0
TRAIN_ITERS=0
NNODES=$(wc -l < $PBS_NODEFILE)
TP=4
MICRO_BATCH=1
GLOBAL_BATCH=1
RATIO=1
SUB_GROUP_SIZE=1000000000
DP=1
PIPELINE_RW=0
OPT_RATIO=0
OPT_PATHS="/tmp"
ENABLE_CACHING=0
SKIP_GRADS=0
SINGLE_PROC=0
BUFFER_COUNT=16
OUTPUT_POSTFIX=""
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR=$(dirname "$SCRIPT_DIR")
DIR=$PROJECT_ROOT_DIR/scripts/

while getopts ":m:H:F:N:L:U:S:K:T:M:B:R:G:D:P:O:h:C:E:g:s:b:z:" opt; do
  case $opt in
    m)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        model_size_B="$OPTARG"
      else
        echo "Invalid model_size_B: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    H)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        HIDDEN_SIZE="$OPTARG"
      else
        echo "Invalid HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    F)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        FFN_HIDDEN_SIZE="$OPTARG"
      else
        echo "Invalid FFN_HIDDEN_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    N)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_LAYERS="$OPTARG"
      else
        echo "Invalid NUM_LAYERS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    L)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_HEADS="$OPTARG"
      else
        echo "Invalid NUM_HEADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    U)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        SEQ_LENGTH="$OPTARG"
      else
        echo "Invalid SEQ_LENGTH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    S)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        NUM_KV_HEADS="$OPTARG"
      else
        echo "Invalid NUM_KV_HEADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    K)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TRAIN_ITERS="$OPTARG"
      else
        echo "Invalid TRAIN_ITERS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    T)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        TP="$OPTARG"
      else
        TP=4
      fi
      ;;
    M)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        MICRO_BATCH="$OPTARG"
      else
        echo "Invalid MICRO_BATCH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    B)
      if [[ "$OPTARG" =~ ^[0-9]+$ ]]; then
        GLOBAL_BATCH="$OPTARG"
      else
        echo "Invalid GLOBAL_BATCH: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    R)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        RATIO="$OPTARG"
      else
        echo "Invalid RATIO: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    G)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        SUB_GROUP_SIZE="$OPTARG"
      else
        echo "Invalid SUB_GROUP_SIZE: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    D)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        DP="$OPTARG"
      else
        echo "Invalid DP: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    P)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        PIPELINE_RW="$OPTARG"
      else
        echo "Invalid PIPELINE_RW: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    O)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        OPT_RATIO="$OPTARG"
      else
        echo "Invalid OPT_RATIO: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    h)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        HOGGER="$OPTARG"
      else
        HOGGER=0
      fi
      ;;
    C)
      OPT_PATHS="$OPTARG"
      ;;
    E)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        ENABLE_CACHING="$OPTARG"
      else
        echo "Invalid ENABLE_CACHING: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    g)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        SKIP_GRADS="$OPTARG"
      else
        echo "Invalid SKIP_GRADS: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    s)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        SINGLE_PROC="$OPTARG"
      else
        echo "Invalid SINGLE_PROC: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    b)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        BUFFER_COUNT="$OPTARG"
      else
        BUFFER_COUNT=16
      fi
      ;;
    z)
      if [[ -n "$OPTARG" ]]; then
        OUTPUT_POSTFIX="-$OPTARG"
      else
        OUTPUT_POSTFIX=""
      fi
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

print_vals() {
# Perform further processing with the parsed parameters
  echo "model_size_B: $model_size_B"
  echo "HIDDEN_SIZE: $HIDDEN_SIZE"
  echo "FFN_HIDDEN_SIZE: $FFN_HIDDEN_SIZE"
  echo "NUM_LAYERS: $NUM_LAYERS"
  echo "NUM_HEADS: $NUM_HEADS"
  echo "SEQ_LENGTH: $SEQ_LENGTH"
  echo "NUM_KV_HEADS: $NUM_KV_HEADS"
  echo "TRAIN_ITERS: $TRAIN_ITERS"
  echo "MICRO_BATCH: $MICRO_BATCH"
  echo "GLOBAL_BATCH: $GLOBAL_BATCH"
  echo "TWINFLOW RATIO: $RATIO" 
  echo "SUB_GROUP_SIZE: $SUB_GROUP_SIZE"
  echo "DP: $DP"
  echo "PIPELINE_RW: $PIPELINE_RW"
  echo "OPT_RATIO: $OPT_RATIO"
  echo "OPT_PATHS: $OPT_PATHS"
  echo "ENABLE_CACHING: $ENABLE_CACHING"
  echo "SKIP_GRADS: $SKIP_GRADS"
  echo "SINGLE_PROC: $SINGLE_PROC"
  echo "BUFFER_COUNT: $BUFFER_COUNT"
}
print_vals



init_conda() {
	__conda_setup="$(${CONDA_EXE:-"$HOME/miniconda3/bin/conda"} shell.bash hook)"
  eval "$__conda_setup"
  source ~/miniconda3/etc/profile.d/conda.sh  # Ensures `conda activate` works in non-login shell
  conda activate /grand/projects/VeloC/am6429/public/dspeed_env
	export PATH=/usr/local/cuda/bin:$CONDA_PREFIX/include/${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/$CONDA_PREFIX/lib/:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	export CFLAGS="-I$CONDA_PREFIX/include/"
  export LDFLAGS="-L$CONDA_PREFIX/lib/"
}
# init_conda

BASE_DATA_PATH=$PROJECT_ROOT_DIR/dataset
DATASET="${BASE_DATA_PATH}/my-gpt2_text_document"
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
USE_DEEPSPEED=1
ZERO_STAGE=3

output_dir="${PROJECT_ROOT_DIR}/logs/"

mkdir -p "$output_dir"
CONFIG_JSON="$output_dir/ds_config.json"
HOSTFILE="$output_dir/hostfile"
echo "PATH=${PATH}" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "CC=gcc" >> .deepspeed_env
echo "CXX=g++" >> .deepspeed_env
echo "IBV_FORK_SAFE=1" >> .deepspeed_env
CFLAGS="-I$CONDA_PREFIX/include/"
echo "CFLAGS=$CFLAGS" >> .deepspeed_env
LDFLAGS="-L$CONDA_PREFIX/lib/"
echo "LDFLAGS=$LDFLAGS" >> .deepspeed_env

EXIT_INTERVAL=200

LR=3e-4
MIN_LR=3e-5
DTYPE="bf16"
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1

options=" \
	--tensor-model-parallel-size $TP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH \
       --global-batch-size $GLOBAL_BATCH \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_ITERS \
       --data-path $DATASET \
       --vocab-file ${VOCAB_PATH} \
	     --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --tokenizer-type GPT2BPETokenizer \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 0 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads ${NUM_KV_HEADS} \
       --deepspeed \
       --exit-interval ${EXIT_INTERVAL} \
       --deepspeed_config=${CONFIG_JSON} \
       --zero-stage=${ZERO_STAGE} \
       --no-pipeline-parallel \
       --cpu-optimizer \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing"



log_str="${model_size_B}B-tp$TP-dp$DP-l$NUM_LAYERS-h$HIDDEN_SIZE-a$NUM_HEADS-sl$SEQ_LENGTH-gbs$GLOBAL_BATCH-mbs$MICRO_BATCH-ratio$RATIO-subg$SUB_GROUP_SIZE-pipelinerw$PIPELINE_RW-opt_ratio$OPT_RATIO-cache$ENABLE_CACHING-skip_grads$SKIP_GRADS-single_proc$SINGLE_PROC-compress0"
# Add the pathnames of OPTIMIZER offloaded paths/
IFS=';' # Set the input field separator to ';'
for path in $OPT_PATHS; do
    dir="${path#/}"       # Remove leading '/'
    dir="${dir%%/*}"      # Extract up to the first '/'
    rm -rf $dir/*
    log_str="$log_str-$dir"
done            


cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"zero_optimization": {
		"stage": $ZERO_STAGE,
        "overlap_comm": true,
        "reduce_scatter": false,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "$OPT_PATHS",
            "buffer_count": $BUFFER_COUNT,
            "pipeline_write": $PIPELINE_RW,
            "pipeline_read": $PIPELINE_RW,
            "pin_memory": true,
            "dist_opt_enable_caching": $ENABLE_CACHING,
            "dist_opt_grad_skip": $SKIP_GRADS,
            "dist_opt_ratio": $OPT_RATIO,
            "dist_opt_one_at_time": $SINGLE_PROC
        },
        "sub_group_size": $SUB_GROUP_SIZE
	},
    "aio": {
        "block_size": 2097152,
        "thread_count": 4,
        "single_submit": false,
        "overlap_events": true
    },
	"bf16": {
		"enabled": true
	}, 
	"data_types": {
		"grad_accum_dtype": "bf16"
 	},
	"wall_clock_breakdown": true,
	"memory_breakdown": true,
	"flops_profiler": {
		"enabled": false
	}
}
EOT

MEGATRON_DIR="$PROJECT_ROOT_DIR/Megatron-DeepSpeed"
run_cmd="deepspeed ${MEGATRON_DIR}/pretrain_gpt.py ${options} | tee -a $output_dir/log-$log_str.log"
echo $run_cmd
eval ${run_cmd}
