NNODES=1
echo "NUM_OF_NODES= ${NNODES}"

###### Begin Definitions
# m = model size in billions of parameters
# H = number of hidden dimensions
# F = FFN hidden size (sets the hidden size in the feed-forward networks within a transformer layer)
# N = number of model layers
# L = number of attn heads
# U = sequence length
# S = number of KV heads
# K = number of training iterations
# T = tensor parallelism degree
# M = microbatch size
# B = global batch size
# R = twinflow offloading ratio
# G = subgroup size for zero-3
# D = data parallelism degree
# P = on-off switch to enable Deep Optimizer States
# A = on-off switch to accelerate backpass with Deep Optimizer States
# O = ``update-stride'' of Deep Optimizer States
###### End Definitions

set_model_size() {
    model_size=$1
    if [[ $model_size == 1 ]]; then
        echo "================== 1.3B OPT model (1 node)"
        declare -g m=1
        declare -g H=2048
        declare -g F=8192
        declare -g N=24
        declare -g L=32
        declare -g U=2048
        declare -g S=8
        declare -g K=10
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=1
        declare -g A=1
        declare -g O=5
    elif [[ $model_size == 3 ]]; then
        echo "================== 3B BLOOM model (1 node)"
        declare -g m=3
        declare -g H=2560
        declare -g F=8192
        declare -g N=30
        declare -g L=32
        declare -g U=2048
        declare -g S=8
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
    elif [[ $model_size == 7 ]]; then
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
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=10000000
        declare -g D=4
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
        echo "================== 30B LLAMA2 (1 node)"
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
        declare -g D=4
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 40 ]]; then
        echo "================== 40B GPT-2 (1 node)"
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
        declare -g D=8
        declare -g A=0
        declare -g O=0
    elif [[ $model_size == 70 ]]; then
        echo "================== 70B LLAMA2 (1 nodes)"
        declare -g m=70
        declare -g H=8192
        declare -g F=28672
        declare -g N=80
        declare -g L=64
        declare -g U=2048
        declare -g S=4
        declare -g K=5
        declare -g T=1
        declare -g M=1
        declare -g B=4
        declare -g R=1
        declare -g P=0
        declare -g G=100000000
        declare -g D=8
        declare -g A=0
        declare -g O=0
    else
        echo "Model config not defined list (model_size = $model_size)"
        exit 1
    fi
    }
# m:H:F:N:L:U:S:K:T:M:B:R:G:P:D:A:O:

############### Run for basic tests
set_model_size 7
D=1                 # Set data-parallelism=1 to run on single GPU
B=$((M * D ))       # Global batch = Micro-batch* DP-degree
# Run for vanilla DeepSpeed
bash config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 0 -O 0

# Run for Deep Optimizer States
bash config-n-run.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -A 1 -O 2

############### Run for basic tests




############### Run for getting memory utilized
# mbs_all=(1)
# set_model_size 20
# D=1
# for mbs in "${mbs_all[@]}"; do
#     echo "================= Running for microbatch size of $mbs for model $model_size =============="
#     B=$((mbs * D ))
#     # # # # # # Run for optimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 0 -D $D -A 1 -O 0
# done
############### Run for getting memory utilized

############### Run for getting parameter update throughput on GPU
# set_model_size 3
# B=$((M * D ))
# bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R 0 -G $G -P 0 -D $D -A 0 -O 0
############### Run for getting parameter update throughput on GPU


############### Run for getting parameter update throughput on CPU
# set_model_size 3
# B=$((M * D ))
# bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R 1 -G $G -P 0 -D $D -A 0 -O 0
############### Run for getting parameter update throughput on CPU

############### Run for diff values of K
# model_sizes=(7)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     B=$((M * D ))
#     ### Run for unoptimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 0 -O 0

#     ###### Run for optimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 1 -O 0

#     # pf_opt_gaps=(2 3 4 6)
#     pf_opt_gaps=(2 3 4 5 6)
#     for gap in "${pf_opt_gaps[@]}"; do
#         echo "================= Running for PF opt gap of $gap =============="
#         B=$((M * D ))
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -A 1 -O $gap
#     done
# done
############### Run for diff values of K

############### Run for diff hogging ratios
# hogging_ratios=(0.1 0.2 0.4 0.6 0.8)
# for hr in "${hogging_ratios[@]}"; do
#     set_model_size 20
#     echo "================= Running for 20B model for hogging ratio $hr =============="
#     B=$((M * D ))
#     # # Run for our approach
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -A 1 -O 2 -h $hr
#     # # Run for unoptimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 0 -O 0 -h $hr
#     # Run for optimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 1 -O 0 -h $hr
# done
############### Run for different hogging ratios

############### Run for 100 iterations
# model_sizes=(30)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     K=5    
#     echo "================= Running for K=100 for model $model_size =============="
#     B=$((M * D ))
#     # # Run for our approach
#     # bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -A 1 -O 2
#     # # Run for unoptimized default
#     # bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 0 -O 0
#     # # Run for optimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 1 -O 0
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
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 1 -D $D -A 1 -O 2
#         # # Run for unoptimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 0 -D $D -A 0 -O 0
#         # Run for optimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $mbs -B $B -R $R -G $G -P 0 -D $D -A 1 -O 0
#     done
# done
############### Run for different MBS




############### Run for different subgroup sizes
# sg_all=(100000000 200000000 500000000 1000000000 2000000000)
# model_sizes=(7)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     for sg in "${sg_all[@]}"; do
#         echo "================= Running for subgroup size of $sg for model $model_size =============="
#         B=$((M * D ))
#         # # Run for our approach
#         # bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $sg -P 1 -D $D -A 1 -O 2
#         # # # Run for unoptimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $sg -P 0 -D $D -A 0 -O 0
#         # # Run for optimized default
#         # bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $sg -P 0 -D $D -A 1 -O 0
#     done
# done
############### Run for different subgroup sizes

# dp_degree=(1 2 4) # 8 16 32 64)
# model_sizes=(7 8 10 13 20)
# for model_size in "${model_sizes[@]}"; do
#     set_model_size $model_size
#     for dp in "${dp_degree[@]}"; do
#         echo "================= Running for DP degree of $dp for model $model_size =============="
#         B=$((M * dp ))
#         # # Run for our approach
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $dp -A 1 -O 2
#         # # Run for unoptimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $dp -A 0 -O 0
#         # Run for optimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $dp -A 1 -O 0
#     done
# done

# sg_all=(10000000 50000000 100000000 200000000 500000000 1000000000 2000000000)
# for sg in "${sg_all[@]}"; do
#     echo "================= Running for Subgroup size of $sg =============="
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R 1 -G $sg -P 0
# done

# bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -A 1 -O $O

# ga_all=(128) 
# set_model_size 20
# for ga in "${ga_all[@]}"; do
#     echo "================= Running for Gradient accumulation size of $ga =============="
#     B=$((M * ga * D ))
#     # Run for our approach
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $D -A 1 -O 2
#     # Run for unoptimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 0 -O 0
#     # Run for optimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $D -A 1 -O 0   
# done



# ################ Run for diff DP degree
# model_sizes=(7 8 10 13 20)
# dp_degree=(1 2 4) # 8 16 32 64)
# for model_size in "${model_sizes[@]}"; do
#     for dp in "${dp_degree[@]}"; do
#         echo "================= Running for DP degree of $dp =============="
#         B=$((M * dp))
#         # # Run for our approach
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 1 -D $dp -A 1 -O $O
#         # # Run for unoptimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $dp -A 0 -O 0
#         # Run for optimized default
#         bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $R -G $G -P 0 -D $dp -A 1 -O 0
#     done
# done
# ################ Run for diff DP degree


# ################ Run for diff twinflow ratio
# set_model_size 7
# ratio_all=(0)
# for ratio in "${ratio_all[@]}"; do
#     echo "================= Running for RATIO $ratio =============="
#     # ### Run for unoptimized default
#     # bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $ratio -G $G -P 0 -D $D -A 0 -O 0
#     ### Ours approach
#     # bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $ratio -G $G -P 1 -D $D -A 1 -O 2
#     # ### Run for optimized default
#     bash ~/dl-io/test_scripts/stage-3-swap/act-my-llama2-cmd.sh  -m $m -H $H -F $F -N $N -L $L -U $U -S $S -K $K -T $T -M $M -B $B -R $ratio -G $G -P 0 -D $D -A 1 -O 0
    
# done
# ################ Run for diff twinflow ratio

