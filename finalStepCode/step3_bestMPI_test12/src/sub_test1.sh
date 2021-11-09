#!/bin/bash
#SBATCH -o ./job_%j_rank%t_%N_%n.out
#SBATCH -p amd_256
#SBATCH -J LQCD
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --exclude=
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shaojiemike@mail.ustc.edu.cn

source /public1/soft/modules/module.sh
module purge
CC=mpiicc
CXX=mpiicpc
CXX_FLAGS=""
raw_flags=" -I../include  -std=c++11"

computetimes=
taskname=so_${CC}_${CXX}_${CXX_FLAGS}

module load intel/20.4.3
module load mpi/intel/20.4.3
make clean
make CC=$CC CXX=$CXX CXX_FLAGS="${CXX_FLAGS}${raw_flags}" 
# mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72  6 12 12 9 > ./main.log

totalnum=7
num=('3' '3' '3' '3')
reverseNum=5 #num[1]+num[2]+num[3]+num[4]-totalnum
for ((i=0 ; i<=num[1] ; i++)); do
    for ((j=0 ; j<=num[2] ; j++)); do
        for ((k=0 ; k<=num[3] ; k++)); do
            num4=$((reverseNum-i-j-k))
            if [[ $num4 -le 3 && $num4 -ge 0  ]]; then
                echo "$i $j $k $num4"
                ans1=$((3*2**$i))
                ans2=$((2**$j*3))
                ans3=$((3*2**$k))
                # ans1，2，3 is the same meaning
                ans4=$((9*2**$num4))
                # ans1=6
                # ans2=12
                # ans3=12
                # ans4=9
                for ((iter=0 ; iter<3 ; iter++)); do
                    echo "$iter"
                    mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72 $ans1 $ans2 $ans3 $ans4 >> ./mpilog/mpi_test1_${ans1}_${ans2}_${ans3}_${ans4}
                done
                output=`cat ./mpilog/mpi_test1_${ans1}_${ans2}_${ans3}_${ans4}|awk '$1=="Total" {print $3}'|awk '{a+=$1}END{print a/NR}'`
                echo "mpi_${ans1}_${ans2}_${ans3}_${ans4} ${output}" >> ./mpiResult_test1.log
            fi
        done
    done
done
echo "best MPI test done"