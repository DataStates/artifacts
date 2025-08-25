#!/bin/bash

set -x
set -o errexit
set -o pipefail
# Installation instructions for MLP-Offload (SC'25)

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT_DIR=$(dirname "$SCRIPT_DIR")

install_cuda() {
    cd $PROJECT_ROOT_DIR
	echo "Installing CUDA... Needs sudo access..."
	wget https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_545.23.06_linux.run
	sudo sh cuda_12.3.0_545.23.06_linux.run
	sudo apt update
	sudo apt-get install libboost-all-dev
}

install_conda() {
    cd $PROJECT_ROOT_DIR
	echo "Installing Conda"
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh
	echo "Please restart shell, or run 'source ~/.bashrc' or 'source ~/.bash_profile' "
}

init_conda() {
	__conda_setup="$(${CONDA_EXE:-"$HOME/miniconda3/bin/conda"} shell.bash hook)"
	eval "$__conda_setup"
	source ~/miniconda3/etc/profile.d/conda.sh  # Ensures `conda activate` works in non-login shell
	conda activate dspeed_env
	export PATH=/usr/local/cuda/bin:$CONDA_PREFIX/include/${PATH:+:${PATH}}
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/$CONDA_PREFIX/lib/:${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
	export CFLAGS="-I$CONDA_PREFIX/include/"
    export LDFLAGS="-L$CONDA_PREFIX/lib/"
}

build_env() {
    cd $PROJECT_ROOT_DIR
	conda create -n dspeed_env python=3.12 -y
	conda activate dspeed_env
	echo "Installing PyTorch"
	pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
	pip install nltk 'numpy<2' regex six transformers sentencepiece pybind11 einops packaging ninja pandas scikit-learn matplotlib tqdm
	conda install anaconda::libaio
	conda install conda-forge::boost
}

install_apex() {
    cd $PROJECT_ROOT_DIR
    init_conda
    conda activate dspeed_env
    echo "PATH after activation: $PATH"
    python -c "import torch" || { echo "PyTorch is not available!"; return 1; }

    echo $PATH
	COMMIT_ID=c02c6c891eedfabf91f0de8127d7636d4292356d
	echo "Installing NVIDIA Apex... This takes ~10 minutes..."
	git clone https://github.com/NVIDIA/apex
	cd apex/
	git checkout $COMMIT_ID
	# git checkout 6309120bf4158e5528 # This commit didn't give NCCL faults.
	sed -i 's/^[[:space:]]*check_cuda_torch_binary_vs_bare_metal(CUDA_HOME)/#&/' setup.py # this command is required to avoid CUDA vs PyTorch version mismatch errors.
	pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
}

install_megatron_ds() {
    cd $PROJECT_ROOT_DIR
	echo "Using git submodule of Megatron-DeepSpeed"
	git submodule update --init --recursive
	cd $PROJECT_ROOT_DIR/Megatron-DeepSpeed
	git checkout dist_nvme_opt
}

load_dataset() {
    cd $PROJECT_ROOT_DIR
    echo "Downloading dataset..."
    mkdir -p $PROJECT_ROOT_DIR/dataset
    cd dataset/
    if [ -d "$PROJECT_ROOT_DIR/dataset/oscar-1GB.jsonl" ]; then
        echo "Dataset already exists, skipping download."
        return
    fi
    wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz # training dataset
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json #Vocabulary
    wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt # Merge File

    xz -d oscar-1GB.jsonl.xz     # extract the dataset.
    cd $PROJECT_ROOT_DIR/Megatron-DeepSpeed
    init_conda
    conda activate dspeed_env
    python tools/preprocess_data.py \
        --input $PROJECT_ROOT_DIR/dataset/oscar-1GB.jsonl \
        --output-prefix $PROJECT_ROOT_DIR/dataset/my-gpt2 \
        --vocab-file $PROJECT_ROOT_DIR/dataset/gpt2-vocab.json \
        --dataset-impl mmap \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file $PROJECT_ROOT_DIR/dataset/gpt2-merges.txt \
        --append-eod \
        --workers 8
}

install_ds() {
	echo "Using git submodule of DeepSpeed..."
	git submodule update --init --recursive
	cd $PROJECT_ROOT_DIR/DeepSpeed
	git checkout dist_nvme_opt
	init_conda
	echo "CFLAGS is $CFLAGS"
	DS_BUILD_AIO=1 DS_BUILD_CCL_COMM=1 DS_BUILD_CPU_ADAM=1 DS_SKIP_CUDA_CHECK=1 DS_BUILD_CPU_LION=0 DS_BUILD_EVOFORMER_ATTN=0 DS_BUILD_FUSED_ADAM=1 DS_BUILD_FUSED_LION=0 DS_BUILD_CPU_ADAGRAD=1 DS_BUILD_FUSED_LAMB=1 DS_BUILD_QUANTIZER=0 DS_BUILD_RANDOM_LTD=0 DS_BUILD_SPARSE_ATTN=0 DS_BUILD_TRANSFORMER=1 DS_BUILD_TRANSFORMER_INFERENCE=0 DS_BUILD_STOCHASTIC_TRANSFORMER=0 pip install .
	echo "Checking if DeepSpeed installed successfully"
	ds_report
}

install_cuda
install_conda
init_conda
build_env
install_apex
install_megatron_ds
install_ds
load_dataset

# Experiments conducted on
# 1. JLSE Testbed: https://www.jlse.anl.gov/nvidia-h100
# 2. ALCF Polaris: https://docs.alcf.anl.gov/polaris/getting-started/

# Additionally, the scripts in this artifact have been tested on CloudLab system:
# https://www.wisc.cloudlab.us/portal/show-node.php?node_id=d8545-10s10505

# Ubuntu 24.04.2 LTS
# > uname -r
# 6.8.0-59-generic

set +x
