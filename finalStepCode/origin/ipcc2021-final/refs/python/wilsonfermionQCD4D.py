#!/usr/bin/python3
# encoding: utf-8

"""
    class WlisonFermion:
        定义了 Wilson 费米子 的 （Dslash * phi） and （Dslash^\dagger phi） 操作,
        其中的 Dslash 是Wilson费米子相关
    D = (Dslash + m) 形式如下：
    D * phi = { (4+m) \delta_{x;x'} - (1/2) *
        [ (1-\gamma_0) U_{0} (x) \delta_{x0+1; x0'}  + (1+\gamma_0) U^\dagger_{0}(x0-1) \delta_{x0-1; x0'}
        + (1-\gamma_1) U_{1} (x) \delta_{x1+1; x1'}  + (1+\gamma_1) U^\dagger_{1}(x1-1) \delta_{x1-1; x1'}
        + (1-\gamma_2) U_{2} (x) \delta_{x2+1; x2'}  + (1+\gamma_2) U^\dagger_{1}(x2-1) \delta_{x2-1; x2'}
        + (1-\gamma_3) U_{3} (x) \delta_{x3+1; x3'}  + (1+\gamma_3) U^\dagger_{1}(x3-1) \delta_{x3-1; x3'}
        ] } * phi(x')

    - 这里，没有下标的 x 和 x' 分别表示四维格子的一个时空点，如 x:=(x0,x1,x2,x3) (即分别对应 x y z t);  
    - 关于 delta: delta_{x0+1; x0'} := delta_{x0+1, x1,x2,x3; x0', x1', x2',x3'}, 
      即表示仅当 x0'=x0+1 时，delta=1; 在公式里我们省略了没有偏移（如x1=x1',x2=x2'）的指标
    - 色指标：
        规范场 U_\mu(x), 对于任意的确定的时空点 x 和确定的方向 mu， U_\mu(x) 代表一个3x3的SU(3)矩阵
        只作用于色空间
    - dirac 指标
        - gamma 矩阵的矩阵乘法等线性操作 仅作用于 dirac 空间 (即程序中 与 “DRC”，“drc”相关维度指标)
        - gamma矩阵的定义：
            \gamma_0 = [[0,0,0,-i], [0,0,-i,0], [0,i,0,0], [i,0,0,0]]
            \gamma_1 = [[0,0,0,-1], [0,0,1,0], [0,1,0,0], [1,0,0,0]]
            \gamma_2 = [[0,0,-i,0], [0,0,0,i], [i,0,0,0], [0,-i,0,0]]
            \gamma_3 = [[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]]
            \gamma_4 = [[1,0,0,0], [0,1,0,0], [0,0,-1,0], [0,0,0,-1]] 
            # gamma_4 对应物理上的 gamma5 在dlash中没有用到
        - gamma矩阵的一些关系式:
            - gamma矩阵（包括gamma5）之间满足反对易关系，即
                gamma_mu * gamma_nv  + gamma_nu * gamma_mu = 2 * delta_{mu, nu} 
            - 幺正性和厄米性：
                gamma_mu = gamma_mu^\dagger, 
                gamma_mu * gamma_mu = gamma_mu * gamma_mu^\dagger = 1
    - 张量积（直积）：
        上式中的 gamma 与 规范场 Uu(x)相乘，是gamma矩阵（4维,dirac空间） 与 Uu(x) (3维,色空间) 的张量积（直积），
        其结果为一个12维的矩阵。更规范的写法通常是在它门之间用一个带圈的X号表示它们直乘 
    - 费米子矩阵的 gamma_5 厄米性：
        费米子矩阵M（即上式中的D）满足关系式： gamma_5 * M * gamma_5 = M^\dagger
"""

import numpy as np
from tqdm import tqdm
import pickle


class WilsonFermion(object):
    def __init__(self, X, mass=0.1):
        self.DRC = 4
        self.CLR = 3
        self.DIM = 4
        self.X = X   # X = [Nx, Ny, Nz, Nt]
        self.mass = mass
        self.U = np.zeros((self.DIM, self.CLR, self.CLR,
                          self.X[0], self.X[1], self.X[2], self.X[3]), dtype=complex)

        self.Gammaidx = np.array([[3, 2, 1, 0],
                                 [3, 2, 1, 0],
                                 [2, 3, 0, 1],
                                 [2, 3, 0, 1],
                                 [0, 1, 2, 3]])
        self.Gamma = np.array([[-1j, -1j,  1j,  1j],
                               [-1,  1,  1, -1],
                               [-1j,  1j,  1j, -1j],
                               [1,  1,  1, 1],
                               [1, 1, -1, -1]], dtype=complex)

    def create_fermion(self):
        F = np.zeros((self.DRC, self.CLR,
                      self.X[0], self.X[1], self.X[2], self.X[3]), dtype=complex)
        return F

    def create_pointsrc(self, drc, clr, x):
        src = self.create_fermion()
        src[drc, clr, x[0], x[1], x[2], x[3]] = 1.0
        return src

    def create_prop(self):
        prop = np.zeros((self.DRC, self.DRC, self.CLR, self.CLR,
                        self.X[0], self.X[1], self.X[2], self.X[3]), dtype=complex)
        return prop

    def prop_dagger(self, prop):
        return np.transpose(np.conj(prop), 1, 0, 3, 2, 4, 5, 6, 7)

    """ 组态：从文件读取组态数据；或 产生 “unit” or “random” 组态"""

    def load_conf(self, file_in):
        f = open(file_in, "rb")
        self.U = pickle.load(f)
        f.close()
        print(f"[GAUGE] load gauge configuration from file: {file_in}")

    def get_conf(self, type="unit"):
        if type == "unit":
            for dim in range(self.DIM):
                for clr in range(self.CLR):
                    self.U[dim, clr, clr] = np.ones(
                        (self.X[0], self.X[1], self.X[2], self.X[3]), dtype=complex)
        if type == "random":
            print(f"[GAUGE] {type} gauge not finished.")
            exit
            # self.U = np.exp(1j * np.random.uniform(-np.pi,
            # np.pi, size=(2, self.Nx, self.Nt)))
        print(f"[GAUGE] {type} gauge configuration using")

    """ Dlsash 相关操作,"""

    def shift(self, phi, dim, direction):
        # dim = 0/t --> x/t-dim; direction = +1/-1
        ix = (np.arange(self.X[dim]) + direction) % self.X[dim]
        if dim == 0:
            return phi[:, :, ix, :, :, :]
        elif dim == 1:
            return phi[:, :, :, ix, :, :]
        elif dim == 2:
            return phi[:, :, :, :, ix, :]
        elif dim == 3:
            return phi[:, :, :, :, :, ix]

    def gamma_mul_phi(self, phi, dim):
        res = self.create_fermion()
        for drc in range(self.DRC):
            res[drc] = self.Gamma[dim, drc] * phi[self.Gammaidx[dim, drc]]
        return res

    def project(self, phi, dim, direction):
        # @direction： +/-, 分别对应 (1 -/+ \gamma_\mu) phi （注意刚好相反）
        return phi - direction * self.gamma_mul_phi(phi, dim)

    def Ud_mul_phi(self, dim, direction, phi):
        res = self.create_fermion()
        if direction == +1:
            Udim = self.U[dim]
        else:
            Udim = np.conj(self.U[dim]).swapaxes(0, 1)
        for clr in range(self.CLR):
            for drc in range(self.DRC):
                res[drc][clr] = np.sum(Udim[clr]*phi[drc], axis=0)
        return res

    """ Dslash:
        dag = -/+  --> dagger = False/True
    """

    def D(self, phi, dagger=False):
        dag = dagger*2 - 1
        res = self.create_fermion()
        # for direction in [+1, -1]:
        direction = +1
        for dim in range(self.DIM):
            tmp = self.Ud_mul_phi(
                dim, direction, self.shift(phi, dim, direction))
            res += self.project(tmp, dim, direction*dag)
        direction = -1
        for dim in range(self.DIM):
            tmp = self.shift(self.Ud_mul_phi(
                dim, direction, phi), dim, direction)
            res += self.project(tmp, dim, direction*dag)
        res = (4.0 + self.mass) * phi - 0.5*res
        return res

    def DdD(self, phi):
        return self.D(self.D(phi), True)

    @ staticmethod
    def norm(x, y):
        return np.real(np.vdot(x, y))


def main():
    X = [8, 8, 8, 8]
    mass = 0.1
    A = WilsonFermion(X, mass)
    A.get_conf("unit")
    # A.get_conf("random")
    # A.load_conf("./dat/conf121.dat")

    # make source
    x_src = [0, 0, 0, 0]
    src = A.create_pointsrc(0, 0, x_src)
    src = np.exp(1j * np.random.uniform(-np.pi, np.pi,
                 size=(4, 3, X[0], X[1], X[2], X[3])))

    # check
    q = src
    Dq = A.D(q)
    Ddq = A.D(q, True)
    DdDq = A.DdD(q)

    qDq = np.vdot(q, Dq)
    DqDq = np.vdot(Dq, Dq)
    qDdDq = np.vdot(q, DdDq)

    # print("  sum(D * q) =", np.sum(Dq))
    # print("  sum(D^\dagger * D * q ) =", np.sum(DdDq))
    # print("  sum(D^\dagger * q ) =", np.sum(Ddq))
    print("  qDq = q^\dagger * D * q = ", qDq)
    print("  qDdDq = q^\dagger * D^\dagger * D * q = ", qDdDq)
    print("  DqDq = (D*q)^\dagger * (D*q) =", np.vdot(Dq, Dq))
    # print("  DdqxDdQ(D^\dagger*q)^\dagger * (D^\dagger*q) =", np.vdot(Ddq, Ddq))
    # print(np.conj(A.U)*A.U)


if __name__ == "__main__":
    from sys import argv
    main()
