#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 2
#SBATCH -n 128
mpirun ./main 0.005  ../data/ipcc_gauge_48_96  48 48 48 96  12 24 24 12