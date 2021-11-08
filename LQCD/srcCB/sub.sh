#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 2
#SBATCH -n 128
mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  6 12 12 9

