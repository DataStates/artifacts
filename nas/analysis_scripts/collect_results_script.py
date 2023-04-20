import os
import sys
import pandas as pd

argc = len(sys.argv)
if argc < 3:
    print ("Usage: Python3 collect_results_script.py <inp directory> <out directory>" )
    sys.exit(0)

input_directory = sys.argv[1]
output_directory = sys.argv[2]

isExist = os.path.exists(output_directory)

if not isExist:
  # Create a new directory because it does not exist
  os.makedirs(output_directory)

cmd = 'deephyper-analytics topk {0}/results.csv -k 13 -o {1}/topkres_temp.csv'.format(input_directory, output_directory)
os.system(cmd)

df = pd.read_csv(output_directory + '/topkres_temp.csv')

model_list = df['arch_seq'].tolist()

print(model_list)
topk_filename_list = []
for i in model_list:
    a = i.replace(']', '')
    a = a.replace('[', '')
    a = a.replace(', ', '-')
    a = a+'.h5'
    topk_filename_list.append(a)

#COPY history directory
copy_history_dir_cmd = 'cp -r {0}/save/history {1}/'.format(input_directory, output_directory)
os.system(copy_history_dir_cmd)


#COPY h5 models
for fname in topk_filename_list:
    copy_model_cmd = 'cp {0}/{1} {2}/'.format(input_directory, fname, output_directory)
    os.system(copy_model_cmd)

copy_results_cmd = 'cp {0}/results.csv {1}/'.format(input_directory, output_directory)
os.system(copy_results_cmd)
