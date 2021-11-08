#!/bin/bash
source  /opt/intel/oneapi/setvars.sh
CC=mpiicc
CXX=mpiicpc
CXX_FLAGS="-xCORE-AVX2 "
raw_flags="-fPIC -I../include  -std=c++11"

make clean
make CC=$CC CXX=$CXX CXX_FLAGS="${CXX_FLAGS}${raw_flags}"
# ./main 0.005  ../data/ipcc_gauge_24_72 24 24 24 72 24 24 24 72 |tee node5baseline.log
mpirun -n 32 ./main 0.005  ../data/ipcc_gauge_24_72 24 24 24 72 6 12 12 36 |tee node5baseline.log
