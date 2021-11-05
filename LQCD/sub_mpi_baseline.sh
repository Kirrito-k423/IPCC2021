#!/bin/bash
#SBATCH -p amd_256
#SBATCH -N 2
#SBATCH -n 128

source /public1/soft/modules/module.sh
module load mpi/intel/2017.5
make clean
make


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
                ans4=$((9*2**$num4))
                # ans1=6
                # ans2=12
                # ans3=12
                # ans4=9
                for ((iter=0 ; iter<3 ; iter++)); do
                    mpirun ./main 0.005  ../data/ipcc_gauge_24_72  24 24 24 72 $ans1 $ans2 $ans3 $ans4 >> ./mpilog/mpi_${ans1}_${ans2}_${ans3}_$ans4
                done
                output=`cat ./mpilog/mpi_${ans1}_${ans2}_${ans3}_${ans4}|awk '$1=="Total" {print $3}'|awk '{a+=$1}END{print a/NR}'`
                echo "mpi_${ans1}_${ans2}_${ans3}_${ans4} ${output}" >> ./mpiResult.log
            fi
        done
    done
done

# 24 24 24 24*3 =  2^3*3 2^3*3 2^3*3 2^3*9
# divide 128 = 2^7

# totalnum=7
# num=('3' '3' '3' '3')
# reverseNum=5 #num[1]+num[2]+num[3]+num[4]-totalnum
# for ((i=0 ; i<=num[1] ; i++)); do
#     for ((j=0 ; j<=num[2] ; j++)); do
#         for ((k=0 ; k<=num[3] ; k++)); do
#             num4=$((reverseNum-i-j-k))
#             if [[ $num4 -le 3 && $num4 -ge 0  ]]; then
#                 echo "$i $j $k $num4"
#                 ans1=$((3*2**$i))
#                 ans2=$((2**$j*3))
#                 ans3=$((3*2**$k))
#                 ans4=$((9*2**$num4))
#                 echo "$ans1 $ans2 $ans3 $ans4"
#             fi
#         done
#     done
# done
