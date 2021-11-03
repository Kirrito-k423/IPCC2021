#!/bin/bash
set -e

###
 # @Descripttion: 
 # @version: 
 # @Author: Shaojie Tan
 # @Date: 2021-08-16 07:56:03
 # @LastEditors: Shaojie Tan
 # @LastEditTime: 2021-08-16 08:00:11
### 
g++ generatePPM.cpp
./a.out
mv homework1.ppm ../exe
cd ../exe
g++ -std=c++11 -fopenmp ../SLIC_raw_test_nocheck.cpp -o SLIC_raw_test_nocheck
echo "generate ppm may tasks 220s"
./SLIC_raw_test_nocheck
cp output_labels_test.ppm check_test.ppm