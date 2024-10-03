
# Define default values
model_size_B=0
HIDDEN_SIZE=0
FFN_HIDDEN_SIZE=0
NUM_LAYERS=0
NUM_HEADS=0
SEQ_LENGTH=0
NUM_KV_HEADS=0
TRAIN_ITERS=0
NNODES=1
TP=1
MICRO_BATCH=1
GLOBAL_BATCH=1
RATIO=1
SUB_GROUP_SIZE=1000000000
PREFETCH_OPTIMIZER=0
DP=1
PART_GRADS_ASYNC=0
PF_OPT_GAPS=2
HOGGER=0

while getopts ":m:H:F:N:L:U:S:K:T:M:B:R:G:P:D:A:O:h:" opt; do
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
    P)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        PREFETCH_OPTIMIZER="$OPTARG"
      else
        echo "Invalid RATIO: $OPTARG is not a valid integer." >&2
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
    A)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        PART_GRADS_ASYNC="$OPTARG"
      else
        echo "Invalid PART_GRADS_ASYNC: $OPTARG is not a valid integer." >&2
        exit 1
      fi
      ;;
    O)
      if [[ "$OPTARG" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        PF_OPT_GAPS="$OPTARG"
      else
        echo "Invalid PF_OPT_GAPS: $OPTARG is not a valid integer." >&2
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
echo "GLOBAL_BATH: $GLOBAL_BATCH"
echo "TWINFLOW RATIO: $RATIO" 
echo "SUB_GROUP_SIZE: $SUB_GROUP_SIZE"
echo "PREFETCH_OPTIMIZER: $PREFETCH_OPTIMIZER"
echo "DP: $DP"
echo "PART_GRADS_ASYNC: $PART_GRADS_ASYNC"
echo "PF_OPT_GAPS: $PF_OPT_GAPS"
echo "HOGGER: $HOGGER"

MASTER_BASEPATH="$HOME/dl-io"
MEGATRON_DIR=$MASTER_BASEPATH/Megatron-DeepSpeed/
cd ${MEGATRON_DIR}
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

BASE_DATA_PATH=$MASTER_BASEPATH/datasets
DATASET="${BASE_DATA_PATH}/my-gpt2_text_document"
TOKENIZER_PATH="${MASTER_BASEPATH}/datasets/tokenizer.model"
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
USE_DEEPSPEED=1
ZERO_STAGE=3

output_dir="${MASTER_BASEPATH}/deep-optimizer-states-output-${model_size_B}B/"
mkdir -p "$output_dir"


# Write the env config file.
echo "PATH=${PATH}" > .deepspeed_env
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> .deepspeed_env
echo "http_proxy=${http_proxy}" >> .deepspeed_env
echo "https_proxy=${https_proxy}" >> .deepspeed_env
echo "CC=gcc" >> .deepspeed_env
echo "CXX=g++" >> .deepspeed_env
echo "IBV_FORK_SAFE=1" >> .deepspeed_env
echo "Number of nodes found as $NNODES"


# Setup launch command to match GPU requirement.
NRANKS_PER_NODE=4 # Max number of GPUs per node.
if [ $((DP*TP)) -gt 3 ]; then
  NRANKS_PER_NODE=4
else
  NRANKS_PER_NODE=$((DP*TP))
fi

WORLD_SIZE=$(( NNODES * NRANKS_PER_NODE ))
LAUNCH_PARAMS="--include localhost:"
for ((gpu_id=0; gpu_id<NRANKS_PER_NODE; gpu_id++)); do
    LAUNCH_PARAMS+="$gpu_id"
    if [ $gpu_id -lt $((NRANKS_PER_NODE - 1)) ]; then
        LAUNCH_PARAMS+=","
    fi
done


EXIT_INTERVAL=200

# Write the config file
CONFIG_JSON="$output_dir/ds_config.json"
CHECKPOINT_PATH=/tmp/zero3-tp${TP}_dp${DP} 
cat <<EOT > $CONFIG_JSON
{
	"train_batch_size": $GLOBAL_BATCH,
	"train_micro_batch_size_per_gpu": $MICRO_BATCH,
	"steps_per_print": 1,
	"zero_optimization": {
		"stage": $ZERO_STAGE,
        "offload_optimizer": {
            "device": "cpu",
            "ratio": $RATIO,
            "pin_memory": true,
            "prefetch_optimizer": $PREFETCH_OPTIMIZER,
            "part_grads_async": $PART_GRADS_ASYNC,
            "prefetch_optimizer_gap": $PF_OPT_GAPS
        },
        "sub_group_size": $SUB_GROUP_SIZE
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


# Launch configuration
LR=3e-4
MIN_LR=3e-5
DTYPE="bf16"
LR_WARMUP_STEPS=1
WEIGHT_DECAY=0.1
GRAD_CLIP=1
EXP_DIR=${HOME}/experiments/results/ckpt_reshape
LOG_DIR="${EXP_DIR}/tensorboard/zero3-tp${TP}_dp${DP}_hd${HIDDEN}_nl${LAYERS}_gbsz${GLOBAL_BATCH}_mbsz${MICRO_BATCH}_z${ZERO_STAGE}_LR_${LR}_${MIN_LR}_${DTYPE}_cont"
mkdir -p $LOG_DIR

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
       --save $CHECKPOINT_PATH \
       --data-path $DATASET \
       --vocab-file ${VOCAB_PATH} \
	   --merge-file ${MERGE_PATH} \
       --data-impl mmap \
       --tokenizer-type GPTSentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
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
       --zero-stage=${ZERO_STAGE}
       --no-pipeline-parallel \
       --cpu-optimizer \
       --checkpoint-activations \
       --deepspeed-activation-checkpointing"




log_str="${model_size_B}B-tp$TP-dp$DP-l$NUM_LAYERS-h$HIDDEN_SIZE-a$NUM_HEADS-sl$SEQ_LENGTH-gbs$GLOBAL_BATCH-mbs$MICRO_BATCH-ratio$RATIO-subg$SUB_GROUP_SIZE-prefetch$PREFETCH_OPTIMIZER-flush_async$PART_GRADS_ASYNC-opt_gaps$PF_OPT_GAPS"
rm -rf $output_dir/log-$log_str.log
rm -rf $output_dir/log-$log_str-monitor.csv
rm -rf $output_dir/log-$log_str-monitor-*.csv
rm -rf $output_dir/log-$log_str-nsys.*
killall -9 python
killall -9 python3


run_cmd="deepspeed ${LAUNCH_PARAMS} ${MEGATRON_DIR}/pretrain_gpt.py ${options} | tee $output_dir/log-$log_str.log 2>&1"
echo $run_cmd
eval ${run_cmd}
