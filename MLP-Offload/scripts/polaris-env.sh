#!/bin/bash

module use /soft/modulefiles
module load spack-pe-base
module load conda/2024-10-30-workshop gcc/11.4.0 

conda activate /grand/projects/VeloC/am6429/public/dspeed_env
export CFLAGS="-I$CONDA_PREFIX/include/"
export LDFLAGS="-L$CONDA_PREFIX/lib/"
export LD_PRELOAD=/soft/spack/base/0.8.1/install/linux-sles15-x86_64/gcc-12.3.0/gcc-11.4.0-hsaeh45fsromryk5wphvhbmx3bbqtofu/lib64/../lib64/libstdc++.so.6

[ -n "$NCCL_NET_GDR_LEVEL" ] && unset NCCL_NET_GDR_LEVEL
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/grand/projects/VeloC/bogdan/aws-ofi-nccl/1.14.0/lib/:/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072

# Additional variables that might be critical to address any potential hang issue in Python applications. 
export FI_CXI_DEFAULT_TX_SIZE=131072
export FI_CXI_RDZV_PROTO=alt_read
export FI_CXI_REQ_BUF_SIZE=8388608
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16000
export FI_CXI_RDZV_THRESHOLD=2000