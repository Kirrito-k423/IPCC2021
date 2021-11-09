#!/usr/bin/python3
# encoding: utf-8

"""
    Now 2D-QED and 4D-QCD are supported.
"""


import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from wilsonfermionQCD4D import WilsonFermion as QCD4DAction


def contract(prop1, prop2):
    # prop = np.zeros((2,2,Nx,Nt))
    # size = np.shape(prop1)
    # pmin = 2*np.pi/size[2]
    # P = np.exp((-1j)*pmin*pmode)
    Cxt = np.sum(prop1 * np.conj(prop2), axis=(0, 1))
    Ct = np.sum(Cxt, axis=0)
    return Ct


def contraction(gamma1, prop1, gamma2, prop2, A, pmode=0):
    S1 = A.create_prop()
    S2 = A.create_prop()
    if A.CLR == 1:   # QED
        for drc in range(A.DRC):
            S1[drc] = A.gamma_mul_phi(prop1[drc], gamma1)
            S2[drc] = A.gamma_mul_phi(np.conj(prop2[drc]), gamma2)
        Cxt = np.sum(S1*S2, axis=(0, 1))
        # TODO pmode
        Ct = np.sum(Cxt, axis=0)
    else:
        for drc in range(A.DRC):
            for clr in range(A.CLR):
                S1[drc, :, clr] = A.gamma_mul_phi(
                    prop1[drc, :, clr], gamma1)
                S2[drc, :, clr] = A.gamma_mul_phi(
                    np.conj(prop2[drc, :, clr]), gamma2)
        Cxt = np.sum(S1*S2, axis=(0, 1, 2, 3))
        # TODO pmode
        Ct = np.sum(Cxt, axis=(0, 1, 2))
    return Ct


def plt_stdCt(Ct, err):
    plt.figure(figsize=(12, 12))
    X = np.arange(len(Ct))
    plt.errorbar(X, Ct, err, fmt='r.')
    plt.errorbar(X, Ct, err, fmt='r.')
    plt.fill_between(X, Ct-err, Ct+err, color='y', alpha=0.3)
    # plt.ylabel("cosh{[C(t-1) + C(t+1)] / C(T)}")
    plt.ylabel("log(Ct)")
    plt.xlabel("nt")
    plt.grid()
    plt.savefig("Ct.png")
    plt.close('all')


def plt_stdEt(Et, err):
    plt.figure(figsize=(12, 12))
    # Nt = Ct.shape
    X = np.arange(len(Et))
    plt.errorbar(X, Et, err, fmt='r.')
    # plt.fill_between(X, Et-err, Et+err, color='b', alpha=0.3)
    # plt.xlim(5,len(Ct))
    # plt.ylim(-1,+1)
    plt.grid()
    plt.ylabel("Et")
    plt.xlabel("nt")
    plt.savefig("Et.png")
    plt.close('all')


def analyis(Ct):
    Ncfg, Nt = Ct.shape
    Nth = int(Nt/2)
    it = np.arange(Nt)
    it_rv = (Nt-1 - it)
    Ct = np.abs(0.5*(Ct[:, it] + Ct[:, (it_rv+1+Nt) % Nt]))
    Ctlog = np.log(Ct)
    Ctavrg = np.mean(Ctlog, axis=0)
    Cterr = np.sqrt(np.std(Ctlog, axis=0))
    plt_stdCt(Ctavrg, Cterr)

    Rt = np.abs(Ct[:, 0:Nth]+Ct[:, 2:Nth+2] / Ct[:, 1:Nth+1])
    Et = np.cosh(Rt)
    Etavrg = np.mean(Et, axis=0)
    Eterr = np.sqrt(np.std(Et, axis=0))
    plt_stdEt(Etavrg, Eterr)
    # print("avrg\n", Etavrg, "\n err\n", Eterr)


def main():
    from main import test
    test("QCD")
    # Ct = np.zeros((Ncfg, Nt))
    # Rt = np.zeros((Ncfg, int(Nt/2)))
    # phi = np.zeros((2, Nx, Nt), dtype=complex)
    # print("》》》 【 Please  See 'main.py' 】 ")


if __name__ == "__main__":
    from sys import argv

    main()
