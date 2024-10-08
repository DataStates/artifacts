# Deep Optimizer States

Artifacts corresponding to Middleware'24 paper titled "Deep Optimizer States: Towards Scalable Training of Transformer Models using Interleaved Offloading"
The core idea of Deep Optimizer States is to accelerate LLM training when large optimizer states are offloaded to the CPU memory using DeepSpeed ZeRO-3 techniques. More specifically, it aims at improving the backward pass through asynchronous gradient transfers and the update phase using hybrid CPU-GPU computations.

### Installing Softwares
1. Basic pacakges

    a. [Python (>=3.10)](https://www.python.org/downloads/release/python-3100/)

    b. [CUDA toolkit version 12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive)

    c. [GCC version 11.1](https://gcc.gnu.org/install/)

2. Create a virtual environment (`dspeed_env`), using [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or [pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).

3. Install PyTorch version 2.3.1 with CUDA 12.1 support [link for previous versions](https://pytorch.org/get-started/previous-versions/) or using `pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121`

4. Clone and Install Nvidia APEX (for mixed-precision training)
    ```
    # Create a base directory to install packages and download dataset.
    export MASTER_BASEPATH="$HOME/dl-io/"

    mkdir -p $MASTER_BASEPATH

    cd $MASTER_BASEPATH

    conda activate dspeed_env

    git clone https://github.com/NVIDIA/apex

    cd apex/

    git checkout 6309120bf4158e5528 # This commit didn't give NCCL faults.

    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    ```

5. Clone and Install Megatron-DeepSpeed
    ```
    cd $MASTER_BASEPATH

    git clone https://github.com/microsoft/Megatron-DeepSpeed.git

    cd Megatron-DeepSpeed/

    pip install regex six sentencepiece pybind11 einops
    ```

6. Clone and Install Our fork of DeepSpeed:
    ```
    cd $MASTER_BASEPATH

    git clone https://github.com/DataStates/DeepSpeed.git

    cd DeepSpeed

    git checkout deep-optimizer-states

    DS_BUILD_OPS=0 DS_BUILD_CPU_ADAM=1 DS_BUILD_ASYNC_COPIER=1 DS_BUILD_UTILS=1 pip install . -v

    ds_report # check if installation was successful
    ```
    The environment variables for installing modules in DeepSpeed are detailed [here](https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops). We add a new module called `ASYNC_COPIER` for scheduling GPU<->CPU data movement for Deep Optimizer States.

### Downloading and Pre-processing Training Dataset
For evaluating Deep Optimizer States, we use a tiny dataset available from the BLOOM repository.

```
mkdir $MASTER_BASEPATH/dataset/

cd $MASTER_BASEPATH/dataset/

wget https://huggingface.co/bigscience/misc-test-data/resolve/main/stas/oscar-1GB.jsonl.xz # training dataset

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json #Vocabulary

wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt # Merge File

xz -d oscar-1GB.jsonl.xz     # extract the dataset.

cd $MASTER_BASEPATH/Megatron-DeepSpeed

python tools/preprocess_data.py \
    --input ~/dataset/oscar-1GB.jsonl \
    --output-prefix ~/dataset/my-gpt2 \
    --vocab-file ~/dataset/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file ~/dataset/gpt2-merges.txt \
    --append-eod \
    --workers 8
```

### Running Deep Optimizer States
```
cd $MASTER_BASEPATH/

git clone https://github.com/DataStates/artifacts.git deep-optimizer-states-artifact

cd $MASTER_BASEPATH/deep-optimizer-states-artifact

bash master-script.sh

# Quickly evaluate how iteration times accelerate with Deep Optimizer States (using sample-logs)
python quick-parse-results.py --vanilla-deepspeed $MASTER_BASEPATH/deep-optimizer-states-artifact/deep-optimizer-states/sample-logs/deep-optimizer-states-output-7B/log-7B-tp1-dp1-l32-h4096-a32-sl2048-gbs1-mbs1-ratio1-subg10000000-prefetch0-flush_async0-opt_gaps0.log --deep-optimizer-states $MASTER_BASEPATH/deep-optimizer-states-artifact/deep-optimizer-states/sample-logs/deep-optimizer-states-output-7B/log-7B-tp1-dp1-l32-h4096-a32-sl2048-gbs1-mbs1-ratio1-subg10000000-prefetch1-flush_async1-opt_gaps2.log

# Sample output paring starts: showing longer iteration time of vanilla deepspeed.
Interval times from Vanilla Deepspeed:
Iteration 1: 16.34359383583069 seconds
Iteration 2: 9.649179458618164 seconds
Iteration 3: 9.47140622138977 seconds
Iteration 4: 9.480634927749634 seconds
Iteration 5: 9.52421760559082 seconds
Iteration 6: 9.482650995254517 seconds
Iteration 7: 9.453252792358398 seconds
Iteration 8: 9.459012508392334 seconds
Iteration 9: 9.481554746627808 seconds
Iteration 10: 9.515002965927124 seconds


Interval times from Deep Optimizer States:
Iteration 1: 8.746816396713257 seconds
Iteration 2: 7.039402961730957 seconds
Iteration 3: 6.768996953964233 seconds
Iteration 4: 6.735331773757935 seconds
Iteration 5: 6.774142265319824 seconds
Iteration 6: 6.855173826217651 seconds
Iteration 7: 6.818261384963989 seconds
Iteration 8: 6.819939374923706 seconds
Iteration 9: 6.7979536056518555 seconds
Iteration 10: 6.824923276901245 seconds
# Sample output parsing ends 

```


### Contact
In case of questions and comments, please contact the authors on the paper.
