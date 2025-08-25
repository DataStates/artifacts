#!/bin/bash -l

set -x
echo "Starting MLP-Offload experiments:"
SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source $SCRIPT_DIR/polaris-env.sh
NVME_PATH="/local/scratch"
PFS_PATH="/grand/projects/VeloC/am6429/public/scratch" 	# "/vast/users/$USER/scratch"
PROJECT_ROOT_DIR=$(dirname "$SCRIPT_DIR")
DIR=$PROJECT_ROOT_DIR/scripts/

# This function sets the model size and other parameters based on the model size passed as an argument.
# m = Model size in billions
# H = Hidden size
# F = Feedforward size
# N = Number of layers
# L = Number of attention heads
# U = Sequence length
# S = Number of key-value heads
# K = Number of training iterations
# T = Tensor parallelism factor
# M = Micro-batch size
# B = Global batch size
# R = CPU to GPU optimizer offloading ratio (we consider no space on GPU, so this ratio always stays 1)
# G = Subgroup size
# D = Number of data parallel processes
# P = Enable pipelined read/write to NVMe (enabled by default, comes with DeepSpeed vanilla; otherwise I/O is 4x slower)
# O = Node-local NVMe to PFS offloading ratio
# g = Skip gradient flushing (design-4 of the paper)
# s = Single process flush/fetch from a tier at a time for concurrency control (design 2 of the paper)
# b = buffer slots to be used as cache for prefetching or lazy flushing from NVMe


set_model_size() {
    model_size=$1
    if [[ $model_size == 7 ]]; then
        echo "================== 7B LLAMA2 (1 node)"
        declare -g m=7
        declare -g H=4096
        declare -g F=11008
        declare -g N=32
        declare -g L=32
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=1
        declare -g R=1
        declare -g P=0
        declare -g G=10000000
        declare -g D=1
        declare -g A=1
        declare -g O=5
    elif [[ $model_size == 8 ]]; then
        echo "================== 8.3B LLAMA2 (1 node)"
        declare -g m=8.3
        declare -g H=3072
        declare -g F=11008
        declare -g N=72
        declare -g L=32
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g O=5
    elif [[ $model_size == 10 ]]; then
        echo "================== 10B LLAMA2 (1 node)"
        declare -g m=10
        declare -g H=4096
        declare -g F=12400
        declare -g N=50
        declare -g L=32
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g O=5
    elif [[ $model_size == 13 ]]; then
        echo "================== 13B LLAMA2 (1 node)"
        declare -g m=13
        declare -g H=5120
        declare -g F=13824
        declare -g N=40
        declare -g L=40
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g O=5
    elif [[ $model_size == 20 ]]; then
        echo "================== 20B ZeRO paper (1 node)"
        declare -g m=20
        declare -g H=5120
        declare -g F=20480
        declare -g N=40
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=1
        declare -g O=5
    elif [[ $model_size == 30 ]]; then
        echo "================== 30B LLAMA (https://huggingface.co/huggyllama/llama-30b/blob/main/config.json) (1 node)"
        declare -g m=30
        declare -g H=6656
        declare -g F=17920
        declare -g N=60
        declare -g L=52
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=8
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 40 ]]; then
        echo "================== 40B GPT-2 (https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf) (1 node)"
        declare -g m=40
        declare -g H=5120
        declare -g F=20480
        declare -g N=128
        declare -g L=40
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 52 ]]; then
        echo "================== FLM2 52B (https://huggingface.co/CofeAI/FLM-2-52B-Instruct-2407) (1 nodes)"
        declare -g m=52
        declare -g H=8192
        declare -g F=21824
        declare -g N=64
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=7
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=10000000
        declare -g D=4
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 66 ]]; then
        echo "================== OPT-66B (https://huggingface.co/facebook/opt-66b/tree/main) (1 nodes)"
        declare -g m=66
        declare -g H=9216
        declare -g F=36864
        declare -g N=64
        declare -g L=72
        declare -g U=2048
        declare -g S=4
        declare -g K=7
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=10000000
        declare -g D=4
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 70 ]]; then
        echo "================== 70B LLAMA2 (https://huggingface.co/meta-llama/Llama-2-70b-hf/blob/main/config.json) (1 nodes)"
        declare -g m=70
        declare -g H=8192
        declare -g F=28672
        declare -g N=80
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=10000000
        declare -g D=4
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 100 ]]; then
        echo "================== 100B LLAMA2 (https://zhuangwang93.github.io/docs/Gemini_SOSP23.pdf) (1 nodes)"
        declare -g m=100
        declare -g H=8192
        declare -g F=28672
        declare -g N=124
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 120 ]]; then
        echo "================== 120B Galatica (https://huggingface.co/facebook/galactica-120b/blob/main/config.json) (1 nodes)"
        declare -g m=120
        declare -g H=10240
        declare -g F=28672
        declare -g N=96
        declare -g L=80
        declare -g U=2048
        declare -g S=4
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=4
        declare -g A=0
        declare -g O=0
    # ================= TWINFLOW EXPTS END =================
    else
        echo "Model size not in defined list... Exiting"
        exit 1
    fi
}
# m:H:F:N:L:U:S:K:T:M:B:R:G:P:D:A:O:

# # ################ Run for diff model sizes

models=(40)
sg_all=(100000000)
PIPELINERW=1
ONEATTIME=1
BC=32

echo "Running for $models, supplied $MODELS, $SUBGROUPS, $PIPELINERW, $ONEATTIME, $BC----"
declare -g K=7
for model in "${models[@]}"; do
    echo "================= Running for Model $model =============="
    set_model_size $model
    for sg in "${sg_all[@]}"; do
        echo "================= Running for subgroup size of $sg for model $model =============="
        G=$sg
        
        B=$((M * D))

        bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 0 -C $NVME_PATH -E 0 -g 0 -s 0 -b $BC

        # bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 2 -C "/tmp/;/vast/users/amaurya/scratch/$m"  -E 1 -g 0 -s $ONEATTIME -b $BC

        bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 3 -C "$NVME_PATH;$PFS_PATH"  -E 1 -g 1 -s $ONEATTIME -b $BC

        # bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 4 -C "/tmp/;/vast/users/amaurya/scratch/$m"  -E 1 -g 0 -s $ONEATTIME -b $BC
    done
done
# ################ Run for diff model sizes

############## Run for diff buffer sizes
# models=($MODELS)
# echo "Running for diff GA $models, supplied $MODELS, $SUBGROUPS, $PIPELINERW, $ONEATTIME, $BC----"
# # sg_all=(10000000 20000000 50000000 100000000 200000000)
# buffer_sizes=(12 16 20 24 32 48 64) 
# for model in "${models[@]}"; do
#     echo "================= Running for Model $model =============="
#     set_model_size $model
#     for bsz in "${buffer_sizes[@]}"; do
#         echo "================= Running for buffer size of $bsz for model $model =============="
#         declare -g K=7
#         declare -g BC=$bsz

#         # bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 0 -C "/tmp/" -E 0 -g 0 -s $ONEATTIME -b $BC -x 0 -z "diff-buffer-counts"
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 3 -C "/tmp/;/vast/users/amaurya/scratch/$m"  -E 1 -g 1 -s $ONEATTIME -b $BC -x 0 -z "diff-buffer-counts"

#     done
# done
############## Run for diff buffer sizes

############### Run for diff gradient accumulation degress
# models=($MODELS)
# echo "Running for diff GA $models, supplied $MODELS, $SUBGROUPS, $PIPELINERW, $ONEATTIME, $BC----"
# # sg_all=(10000000 20000000 50000000 100000000 200000000)
# ga_all=(2 16) 
# for model in "${models[@]}"; do
#     echo "================= Running for Model $model =============="
#     set_model_size $model
#     for ga in "${ga_all[@]}"; do
#         declare -g M=8
#         B=$((M * ga * D ))
#         echo "================= Running for gradient accumulation of factor $ga for model $model =============="
#         declare -g K=12

#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 0 -C "/tmp/" -E 0 -g 0 -s $ONEATTIME -b $BC -x 0 -z "diff-ga"

#         # bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P $PIPELINERW -O 3 -C "/tmp/;/vast/users/amaurya/scratch/$m"  -E 1 -g 1 -s 1 -b $BC -x 1 -z "diff-ga"
#     done
# done
############### Run for diff gradient accumulation degress


############### Run for diff values of K
# model_sizes=(7 8 10)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     ### Run for unoptimized default
#     bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -O 0

#     # # # # # # Run for optimized default
#     bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -O 0

#     pf_opt_gaps=(2 3 4 6)
#     for gap in "${pf_opt_gaps[@]}"; do
#         echo "================= Running for PF opt gap of $gap =============="
#         B=$((M * D ))
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -O $gap
#     done
# done
############### Run for diff values of K

############### Run for diff gradient accumulation degress
# ga_all=(8 32 128 256) 
# set_model_size 20
# for ga in "${ga_all[@]}"; do
#     echo "================= Running for Gradient accumulation size of $ga =============="
#     B=$((M * ga * D ))
#     # Run for our approach
#     bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -O 2
#     # Run for unoptimized default
#     bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -O 0
#     # Run for optimized default
#     bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -O 0   
# done
############### Run for diff gradient accumulation degress


############### Run for different subgroup sizes
# sg_all=(10000000 20000000 50000000 100000000)
# sg_all=(20000000)
# model_sizes=(40)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     for sg in "${sg_all[@]}"; do
#         echo "================= Running for subgroup size of $sg for model $model_size =============="
#         G=$sg
#         # # No pipelinerw, No one-at-time
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P 0 -O 0 -C "/tmp/"

#         # # No pipelinerw, One-at-time
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P 0 -O 1 -C "/tmp/"

#         # # # # Pipelinerw, No one-at-time
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P 1 -O 0 -C "/tmp/"

#         ## Pipelinerw, One-at-time
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -D $D -P 1 -O 1 -C "/tmp/"
#     done
# done
############### Run for different subgroup sizes


############### Run for 100 iterations
# model_sizes=(30)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     K=5    
#     echo "================= Running for K=100 for model $model_size =============="
#     B=$((M * D ))
#     # # Run for our approach
#     # bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -O 2
#     # # Run for unoptimized default
#     # bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -O 0
#     # # Run for optimized default
#     bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -O 0
# done
############### Run for 100 iterations

############### Run for different MBS
# mbs_all=(1 2 4 8 16 32 64)
# model_sizes=(13 20)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     for mbs in "${mbs_all[@]}"; do
#         echo "================= Running for microbatch size of $mbs for model $model_size =============="
#         B=$((mbs * D ))
#         # # Run for our approach
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 1 -D $D -O 2
#         # # Run for unoptimized default
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 0 -D $D -O 0
#         # Run for optimized default
#         bash $DIR/config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 0 -D $D -O 0
#     done
# done
############### Run for different MBS


set +x

