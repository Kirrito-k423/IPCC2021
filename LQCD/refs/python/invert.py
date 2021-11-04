#!/usr/bin/python3
# encoding: utf-8

from sys import implementation
from warnings import simplefilter
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import average
from tqdm import tqdm
from wilsonfermionQCD4D import WilsonFermion as QCD4DAction

"""
CG 算法实现 M^\dagger M  chi = M^\dagger phi 的求解器，求解出 phi
1. 给定 b = M^dagger phi, 最大迭代次数 MaxCG， 求解精度 RsdCG； 
2. 设置解 x 的初值 x = 0;
3. 计算 res = b - M^\dagger M  chi
4. 判断 res 是否小于 Rsd ？ 是结束， 返回 chi； 否则，继续
5. 根据共轭梯度重新给出 d(i) = rTr(i) / rTr(i-1) d(i-1) -r
6. 重复3,4,5
        # rTr(i-1) = r^T r(i-1)
        # r(i) = D^\dagger D x(i-1) - b
        # rsd = | D^\dagger D x(i-1) - b |
        # d(i) = rTr(i)/rTr(i-1) - r(i)
        # print("dTr = ", A.norm(d, r))
"""


def invert(phi, A, RsdCG, MaxCG=100, verbose="SIMPLE"):
    # x = np.random.uniform(-np.pi, np.pi, size=(2, A.Nx, A.Nt))
    # x = np.exp((1j)*x)
    x = A.create_fermion()
    d = A.create_fermion()
    b = A.D(phi, True)  # b = M^dagger phi
    r = b - A.DdD(x)    # r0= b - M^dagger M x
    rTr_bf = 1.0
    for ncg in tqdm(range(MaxCG)):
        rsd_v = b - A.DdD(x)
        # rsdv = phi - A.D(x)
        rsd = np.sqrt(A.norm(rsd_v, rsd_v))
        if verbose == "ALL":
            print(" CG: %d iterations, resdual = %f" % (ncg, rsd))
        if rsd < RsdCG:
            break

        rTr = np.real(A.norm(r, r))
        d = r + (rTr / rTr_bf) * d
        Ad = A.DdD(d)
        alpha = A.norm(r, d) / A.norm(d, Ad)
        rTr_bf = np.real(A.norm(r, r))
        r = r - alpha * Ad
        x = x + alpha * d
    print(" 【CG-INVERT】DONE: %d iters, rsd = %f" % (ncg, rsd))
    if ncg == MaxCG-1:
        print(" [Non-Convergence] %d iters, rsd = %f" % (ncg, rsd))
    return x


def main():
    from main import main as test
    test()
    # print("》》 NOTE 《《")
    # print("  invert() OP works for both 2D-QED and 4D-QCD. ")
    # print("  Please See main.py for tests.\n")


if __name__ == "__main__":
    from sys import argv

    main()
