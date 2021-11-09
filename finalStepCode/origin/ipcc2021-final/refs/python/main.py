#!/usr/bin/python3
# encoding: utf-8

import numpy as np
from numpy.core.fromnumeric import shape
from matplotlib import pyplot as plt
from tqdm import tqdm
from wilsonfermionQCD4D import WilsonFermion as QCD4DAction
from invert import invert as invert
import contraction as CT


def test(action="QCD"):
    MaxCG = 500
    eps = 1e-5
    mass = 0.01
    # a = 0.1
    Ncfg = 1

    if action == "QCD":
        X = [4, 4, 4, 16]
        A = QCD4DAction(X, mass)

    Ct = np.zeros((Ncfg, X[A.DIM-1]), dtype=complex)

    for ncfg in range(Ncfg):
        A.get_conf("unit")
        # A.get_conf("random")
        # A.load_conf("./dat/conf%d.dat" % (100+ncfg))
        print("Gauge shape", np.shape(A.U))
        prop = A.create_prop()
        print("Prop shape", np.shape(prop))

        if action == "QED":
            for drc in range(A.DRC):
                x_src = [0, 0]
                phi = A.create_pointsrc(drc, x_src)
                chi = invert(phi, A, eps, MaxCG)
                # check
                R = phi - A.D(chi)
                rsd2 = np.vdot(R, R)
                prop[drc] = chi
                print(f"  rsd = |Mx -b| = {np.sqrt(rsd2)}")
            if Ncfg == 1:
                Ct = CT.contraction(2, prop, 2, prop, A)
            else:
                Ct[ncfg] = CT.contraction(2, prop, 2, prop, A)
                # Ct[ncfg] = CT.contract(prop, prop)

        if action == "QCD":
            for drc in range(A.DRC):
                for clc in range(A.CLR):
                    x_src = [0, 0, 0, 0]
                    phi = A.create_pointsrc(drc, clc, x_src)
                    chi = invert(phi, A, eps, MaxCG)
                    # check
                    R = phi - A.D(chi)
                    rsd2 = np.vdot(R, R)
                    prop[drc, :, clc] = chi
                    print(f"  rsd = |Mx -b| = {np.sqrt(rsd2)}")
            if Ncfg == 1:
                Ct = CT.contraction(2, prop, 2, prop, A)
            else:
                Ct[ncfg] = CT.contraction(2, prop, 2, prop, A)
                # Ct[ncfg] = CT.contract(prop, prop)
            # print(f"conf.{ncfg} \n", Ct[ncfg])

    if Ncfg == 1:
        plt.figure(figsize=(12, 12))
        X = np.arange(len(Ct))
        Ct = np.abs(Ct)
        Ctlog = np.log(Ct)
        plt.plot(X, Ct, 'r.')
        # plt.plot(X, Ctlog, 'b.')
        plt.ylabel("log(Ct)")
        plt.xlabel("nt")
        plt.grid()
        plt.savefig("Ct.png")
        plt.close('all')
    else:
        CT.analyis(Ct.real)


def main():
    test("QCD")


if __name__ == "__main__":
    from sys import argv

    main()
