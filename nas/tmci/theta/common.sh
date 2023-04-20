#!/bin/bash

# exit on error
set -e
module load conda
conda activate /lus/swift/home/mmadhya1/dh-tmcitest4
module swap PrgEnv-intel PrgEnv-gnu
module load cce
#module load datascience/tensorflow-2.0
