import pandas as pd
import re
import copy
import os
import json

#### Initialization of constants and configurations ####
LOCAL_NVME_ROOT = "tmp"
PFS_ROOT = "vast"
GPU_PER_NODE = 4
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
parent_dir = os.path.dirname(script_dir)
logs_dir = os.path.join(parent_dir, "logs")
# Models are defined with their configuration parameters
MODELS = {
    40: { "m": 40, "F": 10480, "N":128, "L": 40, "U": 2048, "H": 5120, "T": 752563836},
    52: { "m": 52, "F": 8192, "N":64, "L": 64, "U": 2048, "H": 8192, "T": 691850688},
    70: { "m": 70, "F": 8192, "N":80, "L": 64, "U": 2048, "H": 8192, "T": 1071964372 },
    100: { "m": 100, "F": 9216, "N": 124, "L": 64, "U": 2048, "H": 8192, "T": 1654461832 },
    120: { "m": 120, "F": 10240, "N":96, "L": 80, "U": 2048, "H": 10240, "T": 1667647980 }
}

# We have several approaches to perform ablation studies. However, the core compared
# approaches are the baseline (DEFAULT_DEEPSPEED) and MLP-Offload (FULLY_OPTIMIZED).
DEFAULT_DEEPSPEED = "DeepSpeed ZeRO-3"
ENABLE_CACHING = "Enable Caching"
SKIP_GRADS = "Skip Gradients"
SINGLE_PROC = "Process Atomic R/W"
COMPRESSED = "Compressed"
MULTI_PATH = "Multi-Path (with caching)"
MULTI_PATH_SKIP_GRADS = "MP Skip Grads"
MULTI_PATH_SINGLE_PROC = "MP Proc. Atomic R/W"
MULTI_PATH_COMPRESSED = "MP Compressed"
FULLY_OPTIMIZED = "MLP-Offload"

approach_code = {
    0: DEFAULT_DEEPSPEED,
    1: ENABLE_CACHING,
    2: SKIP_GRADS,
    3: SINGLE_PROC,
    4: COMPRESSED,
    5: MULTI_PATH,
    6: MULTI_PATH_SKIP_GRADS,
    7: MULTI_PATH_SINGLE_PROC,
    8: MULTI_PATH_COMPRESSED,
    9: FULLY_OPTIMIZED
}
rev_approach_code = {v: k for k, v in approach_code.items()}
base_config = {"basepath":"./", "approach": DEFAULT_DEEPSPEED, "dp": 4, "tp": 1, "ga": 1, "tf_ratio":1, 
                "act_ckpt":True, "mbs": 1, "gbs": 4, "subg": int(100000000), "opt_ratio": 0,
                "skip_grads": 0, "pipelinerw": 1, "single_proc": 0, "cache": 0, "fs": LOCAL_NVME_ROOT, "compress": 0}

# Define the columns of interest for extracting from the log files.
df_columns = [
    'elapsed_time_per_iteration_ms', 
    'TFLOPs', 
    'fwd', 
    'bwd', 
    'step',
    'bwd_inner_microstep', 
    'bwd_allreduce_microstep', 
    'step_microstep'
]
#### Initialization of constants and configurations ends ####

# Function to get the filename based on model and configuration
def get_filename(m, c):
    filename = (
            f"{c['basepath']}/log-{m['m']}B-tp{c['tp']}-dp{c['dp']}-l{m['N']}-h{m['H']}-a{m['L']}-sl{m['U']}-"
            f"gbs{c['gbs']}-mbs{c['mbs']}-ratio{c['tf_ratio']}-subg{c['subg']}-pipelinerw{c['pipelinerw']}-"
            f"opt_ratio{c['opt_ratio']}-cache{c['cache']}-skip_grads{c['skip_grads']}-"
            f"single_proc{c['single_proc']}-compress{c['compress']}-{c['fs']}"
    )
    filename += ".log"
    return filename

# Function to parse the log file of a given model and runtime configuration and return a DataFrame
def parse_log(m, c):
    log_file = get_filename(m, c)
    print(f"Reading {log_file}")       
    data = {k: [] for k in df_columns}
    with open(log_file, 'r') as file:
        for line in file:        
            match = re.search(r'elapsed time per iteration \(ms\): (\d+\.\d+)', line)
            if match:
                data['elapsed_time_per_iteration_ms'].append(float(match.group(1)))

            match = re.search(r'TFLOPs: (\d+\.\d+)', line)
            if match:
                data['TFLOPs'].append(float(match.group(1)))

            match = re.search(r'fwd: (\d+\.\d+)', line)
            if match:
                data['fwd'].append(float(match.group(1)))

            match = re.search(r'bwd: (\d+\.\d+)', line)
            if match:
                data['bwd'].append(float(match.group(1)))
            
            match = re.search(r'\|\s*step: (\d+\.\d+)', line)
            if match:
                data['step'].append(float(match.group(1)))

            match = re.search(r'bwd_inner_microstep: (\d+\.\d+)', line)
            if match:
                data['bwd_inner_microstep'].append(float(match.group(1)))

            match = re.search(r'bwd_allreduce_microstep: (\d+\.\d+)', line)
            if match:
                data['bwd_allreduce_microstep'].append(float(match.group(1)))

            match = re.search(r'step_microstep: (\d+\.\d+)', line)
            if match:
                data['step_microstep'].append(float(match.group(1)))

    if len(data['step']) == 0:  # This means that we couldn't complete even one step due to OOM
        data = {k: None for k in df_columns}
    df = pd.DataFrame(data, columns=df_columns)
    
    # We would have 10 values, select the last 5 of them
    df = df.tail(len(df.index)-1)
    return df

# Function to get average of a list
def get_avg(arr):
    try:
        return sum(arr)/len(arr)
    except Exception as e:
        print(f"Error in get_avg {e}")


############################################################################
# Parse for different model sizes the breakdown of the timing of each phase.
############################################################################
res = {}
for model in [40, 52, 70, 100, 120]:
    config = copy.deepcopy(base_config)
    res[model] = {}
    for k in [rev_approach_code[DEFAULT_DEEPSPEED], rev_approach_code[FULLY_OPTIMIZED]]:
        res[model][k] = None
        config['basepath'] = logs_dir
        config['m'] = model
        config['opt_gaps'] = k
        config['dp'] = 4
        config['mbs'] = 1
        config['tp'] = 1
        config['gbs'] = config['mbs']*config['dp']
        ans = None
        config['approach'] = approach_code[k]
        if k== rev_approach_code[DEFAULT_DEEPSPEED]:
            config['cache'] = 0
            config['skip_grads'] = 0
            config['opt_ratio'] = 0
            config['fs'] = LOCAL_NVME_ROOT
        elif k == rev_approach_code[FULLY_OPTIMIZED]:
            config['cache'] = 1
            config['skip_grads'] = 1
            config['opt_ratio'] = 3
            config['fs'] = f"{LOCAL_NVME_ROOT}-{PFS_ROOT}"
            config['compress'] = 0
            config['single_proc'] = 1
        else:
            print("Undefined K... Returning")
            import pdb; pdb.set_trace()
        ans = parse_log(MODELS[model], config)
        res[model][k] = ans


def print_summary_table(res):
    print(f"{'Model(B)':<10} {'Approach':<20} {'Elapsed(ms)':>12} {'FWD(ms)':>10} {'BWD(ms)':>10} {'UPDATE(ms)':>10} {'SPEEDUP':>8}")
    print("-" * 100)

    for model, approaches in res.items():
        default_time = 0
        for k, df in approaches.items():
            if isinstance(df, pd.DataFrame):
                elapsed = df['elapsed_time_per_iteration_ms'].mean()
                fwd = df['fwd'].mean()
                bwd = df['bwd'].mean()
                step = df['step'].mean()
                if k == rev_approach_code[DEFAULT_DEEPSPEED]:
                    default_time = elapsed
                speedup = default_time / elapsed
                print(f"{model:<10} {approach_code[k]:<20} {elapsed:12.1f} {fwd:10.2f} {bwd:10.2f} {step:10.2f} {speedup:8.2f}")
            else:
                raise Exception("DataFrame is None for model {}, approach {}".format(model, k))
    print("-" * 100)

print_summary_table(res)
##### SAMPLE SUMMARY TABLE OUTPUT #####
# Model(B)   Approach              Elapsed(ms)    FWD(ms)    BWD(ms) UPDATE(ms)  SPEEDUP
# ----------------------------------------------------------------------------------------------------
# 40         DeepSpeed ZeRO-3         242280.6     653.89   27473.02  213610.43     1.00
# 40         MLP-Offload              101717.3     639.52    2043.55   98507.29     2.38
# 52         DeepSpeed ZeRO-3         238597.6     512.46   28293.34  209336.10     1.00
# 52         MLP-Offload               92173.0     514.48    1833.24   89407.41     2.59
# 70         DeepSpeed ZeRO-3         370562.1     765.63   32905.03  336426.77     1.00
# 70         MLP-Offload              151337.8     770.55    2946.07  147183.76     2.45
# 100        DeepSpeed ZeRO-3         572027.2    1202.37   68341.33  501915.92     1.00
# 100        MLP-Offload              275528.8    1205.63    4563.45  269246.05     2.08
# 120        DeepSpeed ZeRO-3         550360.6    1165.51   73194.69  475480.44     1.00
# 120        MLP-Offload              288178.5    1160.60    4201.09  282331.47     1.91
# ----------------------------------------------------------------------------------------------------
##### SAMPLE SUMMARY TABLE OUTPUT #####