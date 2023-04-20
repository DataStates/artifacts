#!/bin/bash

# Necessary for Bash shells
. /etc/profile

#export HDF5_USE_FILE_LOCKING=FALSE
libfabric_alcf() {
  # use shared recv context in RXM; should improve scalability
  export FI_OFI_RXM_USE_SRX=1
  # note: disable registration cache for verbs provider for now; see
  #       discussion in https://github.com/ofiwg/libfabric/issues/5244
  export FI_MR_CACHE_MAX_COUNT=0
}
if [[ $(hostname -f ) =~ "polaris" ]]; then
   echo "loading polaris"
   source ~/.bashrc
   use_build
   spack env activate .
   source ./venv/bin/activate
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/cpp-store"
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)"
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/soft/libraries/cudnn/cudnn-11.4-linux-x64-v8.2.4.15/lib64"
   export PYTHONPATH="$PYTHONPATH:$(pwd)"
   export PATH="$PATH:$(pwd)"
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/soft/compilers/cudatoolkit/cuda-11.2.2/lib64"
   export PATH="$PATH:/soft/compilers/cudatoolkit/cuda-11.2.2/lib64"
   export PYTHONPATH="$PYTHONPATH:/soft/compilers/cudatoolkit/cuda-11.2.2/lib64"
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/mmadhya1/experiments/venv/lib/python3.8/site-packages/tmci-0.1-py3.8.egg-info"
   
   export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/cray/libfabric/1.11.0.4.125/lib64"
   export PATH="$PATH:$(pwd)/venv/lib/python3.8/site-packages/tmci-0.1-py3.8.egg-info"
   export PYTHONPATH="$PYTHONPATH:$(pwd)/venv/lib/python3.8/site-packages/tmci-0.1-py3.8.egg-info"
   export PYTHONPATH="$PYTHONPATH:$(pwd)/venv/lib/python3.8/site-packages"
   export PYTHONPATH="$PYTHONPATH:$(pwd)/venv/lib64/python3.8/site-packages"
   export PATH="$PATH:$(pwd)/venv/lib/python3.8/site-packages"
   export PATH="$PATH:$(pwd)/venv/lib64/python3.8/site-packages"
   export PYTHONPATH="$(pwd):$PYTHONPATH"
   export GPUS_PER_NODE=4
   export THALLIUM_NETWORK="verbs;ofi_rxm"
   export mpilaunch="aprun"
   export mpilaunchenv="-e PYTHONPATH=\"$PYTHONPATH\" -e LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\" -e PATH=\"$PATH\""
   libfabric_alcf
elif [[ $(hostname) =~ "defiant" ]]
then
  echo "loading defiant"
  #use_cuda
  #conda activate deephyper38
  source ~/.bashrc
  use_build
  use_cuda
  spack env activate .
  . ./venv/bin/activate
  export SAVE_BASE_DIR=/tmp/save
  export PYTHONPATH="$(pwd):$PYTHONPATH"
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$(pwd)/build"
  export GPUS_PER_NODE=1
  export mpilaunch="mpiexec --map-by node -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH"
elif [[ $(hostname) =~ "theta" ]]
then
  echo "loading theta"
  module load conda/2021-11-30
  conda activate deephyper38
  export PYTHONPATH="$(pwd):$PYTHONPATH"
  export SAVE_BASE_DIR=/lus/theta-fs0/projects/VeloC/runderwood/deephyper
  export GPUS_PER_NODE=8
  export mpilaunch="mpiexec --map-by node -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH"
  libfabric_alcf

elif [[ $(hostname) =~ "cc" ]]
then
  echo "loading cooley"
  source ~/.bashrc
  use_build
  spack env activate .
  source ./venv/bin/activate
  export LD_LIBRARY_PATH=/home/mmadhya1/gcc-install/lib64:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=/home/mmadhya1/gcc-install/lib:$LD_LIBRARY_PATH
  export PATH=/home/mmadhya1/gcc-install:$PATH  
  export PYTHONPATH="$(pwd):$PYTHONPATH"
  export CPATH=$CPATH:$HOME/cuda_install/include
  export PATH=$PATH:$HOME/cuda_install/include
  export LD_LIBRARY_PATH=$HOME/experiments/cpp-store:$LD_LIBRARY_PATH
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/cuda_install/lib64
  export HDF5_USE_FILE_LOCKING=FALSE
  export SAVE_BASE_DIR=/lus/theta-fs0/projects/VeloC/runderwood/deephyper
  export GPUS_PER_NODE=2
  export mpilaunch="mpiexec --map-by node -envlist PATH,LD_LIBRARY_PATH,PYTHONPATH,HDF5_USE_FILE_LOCKING"
  libfabric_alcf
fi


which mpiexec
