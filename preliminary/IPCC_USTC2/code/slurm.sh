#!/bin/bash
#SBATCH -o ./job_test_raw_%j.out
#SBATCH -p amd_256
#SBATCH -J tsj_SLIC
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mail-type=ALL
#SBATCH --mail-user=shaojiemike@mail.ustc.edu.cn

source /public1/soft/modules/module.sh

module load intel/20.4.3
module load mpi/intel/20.4.3

export OMP_NUM_THREADS=32  #$SLURM_CPUS_PER_TASK

compile_option=""
SLIC_name=SLIC

icpc -std=c++11 -qopenmp $compile_option $SLIC_name.cpp -o $SLIC_name
OMP_NUM_THREADS=32 ./$SLIC_name|tee out.log
