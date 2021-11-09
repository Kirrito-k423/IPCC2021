#!/usr/bin/python3
# encoding: utf-8

import random as rand
from math import ceil, cos, exp, log, pi, sin, sqrt

# import gvar as gv
import numpy as np
# import seaborn as sns
from matplotlib import pyplot as plt
from numpy.lib import math, ravel_multi_index
from tqdm import tqdm

Nx = 16
Ny = 1
Nz = 1
Nt = 16
V = Nx*Ny*Nz*Nt
Vh = int(V/2)
PI = math.pi


def idx(x, y, z, t):
    return (t+Nt) % Nt * Nz * Ny * Nx + (z+Nz) % Nz * Ny * Nx + (y+Ny) % Ny * Nx + (x+Nx) % Nx


def idxeo(x, y, z, t):
    return int(idx(x, y, z, t)/2) + (x+y+z+t) % 2 * Vh


def idxtmp(x, y, z, t):
    # ix = x+Nx-1 if(x%Nx == 0) else x-1
    iy = y+Ny-1 if(y % Ny == 0) else y-1
    return t*Nz*Ny*Nx + z*Ny*Nx + iy*Nx + x


def defineM(M):
    a = 1.1
    b = 1.2
    c = 1.5
    d = 1.7
    for nt in range(Nt):
        for nz in range(Nz):
            for ny in range(Ny):
                for nx in range(Nx):
                    M[idx(nx, ny, nz, nt)][idx(nx, ny, nz, nt)] = 1.0
                    M[idx(nx, ny, nz, nt)][idx(nx-1, ny, nz, nt)] = a
                    M[idx(nx, ny, nz, nt)][idx(nx+1, ny, nz, nt)] = a
                    M[idx(nx, ny, nz, nt)][idx(nx, ny-1, nz, nt)] = b
                    M[idx(nx, ny, nz, nt)][idx(nx, ny+1, nz, nt)] = b
                    M[idx(nx, ny, nz, nt)][idx(nx, ny, nz-1, nt)] = c
                    M[idx(nx, ny, nz, nt)][idx(nx, ny, nz+1, nt)] = c
                    M[idx(nx, ny, nz, nt)][idx(nx, ny, nz, nt-1)] = d
                    M[idx(nx, ny, nz, nt)][idx(nx, ny, nz, nt+1)] = d


def eoprecMeo(Meo, M):
    for nt1 in range(Nt):
        for nz1 in range(Nz):
            for ny1 in range(Ny):
                for nx1 in range(Nx):
                    for nt2 in range(Nt):
                        for nz2 in range(Nz):
                            for ny2 in range(Ny):
                                for nx2 in range(Nx):
                                    Meo[idxeo(nx1, ny1, nz1, nt1)][idxeo(nx2, ny2, nz2, nt2)] \
                                        = M[idx(nx1, ny1, nz1, nt1)][idx(nx2, ny2, nz2, nt2)]


def Meo2M(M, Meo):
    for nt1 in range(Nt):
        for nz1 in range(Nz):
            for ny1 in range(Ny):
                for nx1 in range(Nx):
                    for nt2 in range(Nt):
                        for nz2 in range(Nz):
                            for ny2 in range(Ny):
                                for nx2 in range(Nx):
                                    M[idx(nx1, ny1, nz1, nt1)][idx(nx2, ny2, nz2, nt2)] \
                                        = Meo[idxeo(nx1, ny1, nz1, nt1)][idxeo(nx2, ny2, nz2, nt2)]


def tmpprecMeo(Mtmp, M):
    for nt1 in range(Nt):
        for ny1 in range(Ny):
            for nx1 in range(Nx):
                for nt2 in range(Nt):
                    for ny2 in range(Ny):
                        for nx2 in range(Nx):
                            Mtmp[idxtmp(nx1, ny1, nt1)][idxtmp(nx2, ny2, nt2)] \
                                = M[idx(nx1, ny1, nt1)][idx(nx2, ny2, nt2)]


def crosslines(Xline, Yline, Lim):
    lst = np.linspace(0, Lim-1, 400+1)
    x_lst = [Xline for x in lst]
    y_lst = [Yline for x in lst]
    plt.plot(lst, x_lst, '--', linewidth='0.4', color='b')
    plt.plot(y_lst, lst, '--', linewidth='0.4', color='b')


def plotM(M):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.matshow(M.real, cmap='hot_r')
    # ax1.matshow(M, cmap='hsv_r')
    crosslines(Vh-0.5, Vh-0.5, V)  # center
    crosslines(Nx-0.5, Nx-0.5, V)
    crosslines(Nx*Ny-0.5, Nx*Ny-0.5, V)
    crosslines(Nx*Ny*Nz-0.5, Nx*Ny*Nz-0.5, V)
    str = f'M  : Nx={Nx}, Ny={Ny}, Nz={Nz}, Nt={Nt}'
    plt.title(str)


def plotMeo(Meo):
    fig = plt.figure()
    ax2 = fig.add_subplot(111)
    # ax2.matshow(Meo, cmap='viridis_r', vmax=0.25, vmin=0.01)
    ax2.matshow(Meo, cmap='hot_r')
    crosslines(Vh-0.5, Vh-0.5, V)
    crosslines(Nx/2-0.5, Vh+Nx/2-0.5, V)
    crosslines(Nx-0.5, Vh+Nx-0.5, V)
    crosslines(Nx*Ny/2-0.5, Vh+Nx*Ny/2-0.5, V)
    crosslines(Nx*Ny-0.5, Vh+Nx*Ny-0.5, V)
    crosslines(Nx*Ny*Nz-0.5, Vh+Nx*Ny*Nz-0.5, V)
    str = f'Meo: Nx={Nx}, Ny={Ny}, Nz={Nz}, Nt={Nt}'
    plt.title(str)


# def loopM(Rg,Lx,Ly,Lt,M0):

# M Meo
M = np.zeros((V, V))
Meo = np.zeros((V, V))
defineM(M)
plotM(M)
# Mtmp = np.dot(M,M)
# plotM(Mtmp)
eoprecMeo(Meo, M)
plotMeo(Meo)

# # loopM(1,Nx,Ny,Nt,Meo)
# # for i in range(1):
Nx = int(Nx/2)
V = int(V/2)
Vh = int(V/2)
M = np.zeros((V, V))
M1 = np.dot(Meo[0:V, V:2*V], Meo[0:V, 0:V])
M = np.dot(M1, Meo[V:2*V, 0:V])
plotM(M)
Mtmp = np.dot(M,M)
plotM(Mtmp)

# for i in range(0):
#     Nx = int(Nx/2)
#     V = int(V/2)
#     Vh = int(V/2)
#     M = np.zeros((V, V))
#     M = Meo[0:V, V:2*V]
#     # * Meo[V:2*V, 0:V]
#     plotM(M)
#     Meo = np.zeros((V, V))
#     eoprecMeo(Meo, M)
#     plotMeo(Meo)

# for i in range(0):
#     Ny = int(Ny/2)
#     V = int(V/2)
#     Vh = int(V/2)
#     M = np.zeros((V, V))
#     M = Meo[0:V, V:2*V] * Meo[0:V, 0:V] * Meo[V:2*V, 0:V]
#     plotM(M)
#     Meo = np.zeros((V, V))
#     eoprecMeo(Meo, M)
#     plotMeo(Meo)

# Mtmp <-- M
# for i in range(1):
#     Nx = int(Nx/2)
#     V = int(V/2)
#     Vh = int(V/2)
#     M = np.zeros((V, V))
#     M = Meo[0:V, V:2*V] * Meo[V:2*V, 0:V]
#     plotM(M)
#     Meo = np.zeros((V, V))
#     tmpprecMeo(Meo, M)
#     plotMeo(Meo)


# Mtmp = np.zeros((V,V))
# for nt1 in range(Nt):
#     for ny1 in range(Ny):
#         for nx1 in range(Nx):
#             for nt2 in range(Nt):
#                 for ny2 in range(Ny):
#                     for nx2 in range(Nx):
#                         Mtmp[idx(nx1, ny1, nt1)][idx(nx2, ny2, nt2)] \
#                             = M[idx4(nx1, ny1, nt1)][idx4(nx2, ny2, nt2)]
# plotM(Mtmp)


plt.show()
# plt.savefig('M_and_Meo.png', dpi=400)
