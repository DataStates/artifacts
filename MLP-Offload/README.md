# MLP-Offload: Multi-Level, Multi-Path Offloading for LLM Pre-training

Software artifact corresponding to SC'25 paper titled "MLP-Offload: Multi-Level, Multi-Path Offloading for LLM Pre-training to Break the GPU Memory Wall"

### High-level Summary
When training large models on limited GPU resources, additional memory tiers such as DRAM and NVMe can be leveraged as swap sapces to accomodate large model sizes. While such democratization leads to the accessiblity of training larger models, it encounters significant slowdown due to the cost of data-movement across memory tiers. *MLP-Offload* aims to mitigate these multi-tier data management challenges by a series of novel design principles such as (a) unified multi-level, multi-path asynchronous offloading using virtual tiers; (b) optimized virtual tier concurrency control for multi-path I/O; (c) cache-friendly ordering of model subgroup processing; and (d) delayed in-place mixed-precision gradient conversion during updates, that are also complemented through an I/O performance model. Please refer our paper for more details.

### Composition of Software Artifacts
The software artifacts in this repository are primarily decomposed into three modules as follows:
1. **Megatron-DeepSpeed**: This is the repository implements core transformer capabilities, central to LLMs. The original [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) library was modified for injecting DeepSpeed specific runtime optimizations, available as [Megatron-DeepSpeed](https://github.com/deepspeedai/Megatron-DeepSpeed). This folder is present as a git submodule and points to our version of Megatron-DeepSpeed that further profiles the performance of different training phases.
2. **DeepSpeed**: This repository contains the DeepSpeed LLM training runtime that proposes various optimizations such as redundancy elimination (ZeRO), offloading, subgroup-sharding, etc. For our targetted scenario of GPU constrained enviroments, we consider the ZeRO-3 optimzation that partitions the model, gradients, and optimizer states across all data-parallel ranks, leading to the most memory efficient training setup. We specifically modify the `deepspeed/runtime/zero/stage3.py` file to run using *MLP-Offload*.
3. **scripts**: This folder contains the scripts required to setup and install required packages, launch experiments using different approaches, and parsers to obtain required performance metrics.
4. **logs**: This folder contains a subset of logs of different approaches presented as sample.

### Testbed Setup

#### Hardware Requirements
1. Nvidia GPU enabled node(s) with A100 or newer architectures
2. Node-local NVMe(s)
3. Remote storage, preferably a parallel file system

#### Software Requirements and Installation
1. Basic pre-requisites include: [Python (>=3.10)](https://www.python.org/downloads/release/python-3100/); [CUDA toolkit version 12.3](https://developer.nvidia.com/cuda-12-3-0-download-archive); [GCC version 11.1+](https://gcc.gnu.org/install/)
2. All other software packages can be installed using [`scripts/installs.sh`](scripts/installs.sh).


### Evaluations

#### Running Experiments
Once the pacakges are installed, change the `NVME_PATH` and `PFS_PATH` in [scripts/launch-expt.sh](scripts/launch-expt.sh) to point to node-local disk and remote storage locations of the testbed.

Finally, running `bash scripts/launch-expt.sh` should start running a 40B model using the vanilla DeepSpeed ZeRO-3 offloading engine and then using MLP-Offloading. The logs become available in the `logs` directory.

#### Parsing Results
After the logs are created in the `logs` directory, revlevant performance profiles can be extracted through the paring script [scripts/parse-res.py](scripts/parse-res.py). Since the generated log filenames contain the root path of the file system(s) used as suffix arguments, we need to set the `LOCAL_NVME_ROOT` and `PFS_ROOT` variables to the root names of the filesystems. We supply sample logs from one of our testbed containing the profiles of 40B-120B model, representative of Figure 7, showing the average iteration time breakdown on scaling model sizes in the paper. These values deafult to `tmp` for our testbed's local NVMe; and to `vast` for our vast filesystem PFS cluster. The logs can be parsed as follows:
```python
> python scripts/parse-res.py
```
Which outputs the following:

| Model(B) | Approach          | Elapsed(ms) | FWD(ms) | BWD(ms) | UPDATE(ms) | SPEEDUP |
|----------|-------------------|-------------|---------|---------|------------|---------|
| 40       | DeepSpeed ZeRO-3  | 242280.6    | 653.89  | 27473.02| 213610.43  | 1.00    |
| 40       | MLP-Offload       | 101717.3    | 639.52  | 2043.55 | 98507.29   | 2.38    |
| 52       | DeepSpeed ZeRO-3  | 238597.6    | 512.46  | 28293.34| 209336.10  | 1.00    |
| 52       | MLP-Offload       | 92173.0     | 514.48  | 1833.24 | 89407.41   | 2.59    |
| 70       | DeepSpeed ZeRO-3  | 370562.1    | 765.63  | 32905.03| 336426.77  | 1.00    |
| 70       | MLP-Offload       | 151337.8    | 770.55  | 2946.07 | 147183.76  | 2.45    |
| 100      | DeepSpeed ZeRO-3  | 572027.2    | 1202.37 | 68341.33| 501915.92  | 1.00    |
| 100      | MLP-Offload       | 275528.8    | 1205.63 | 4563.45 | 269246.05  | 2.08    |
| 120      | DeepSpeed ZeRO-3  | 550360.6    | 1165.51 | 73194.69| 475480.44  | 1.00    |
| 120      | MLP-Offload       | 288178.5    | 1160.60 | 4201.09 | 282331.47  | 1.91    |




