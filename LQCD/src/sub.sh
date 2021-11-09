#!/bin/bash
#SBATCH -o ./slurmlog/job_%j_rank%t_%N_%n.out
#SBATCH -p amd_256
#SBATCH -J LQCD
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --exclude=
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ta1ly@mail.ustc.edu.cn

source /public1/soft/modules/module.sh
module purge
CC=mpiicc
CXX=mpiicpc
CXX_FLAGS=""
raw_flags="-fPIC -I../include  -std=c++11 -march=core-avx2"

MPIOPT=
computetimes="core_avx2"
taskname=so_${CC}_${CXX}_${CXX_FLAGS}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load intel/20.4.3
module load mpi/intel/20.4.3

make clean
make CC=$CC CXX=$CXX CXX_FLAGS="${CXX_FLAGS}${raw_flags}" TARGET=$taskname
mpirun ./$taskname 0.005  ../data/ipcc_gauge_48_96  48 48 48 96  12 24 24 12 > ./log/3_$taskname$computetimes.log
