/**
** @file:  operator.h
** @brief:
**/

#ifndef OPERATOR_H
#define OPERATOR_H

#include <complex>

static int local_site2(const int *coord, const int *latt_size)
{
    int order = 0;
    for (int mmu = 3; mmu >= 1; --mmu) {
        order = latt_size[mmu - 1] * (coord[mmu] + order);
    }
    order += coord[0];
    return order;
}

static int get_nodenum(const int *x, int *l, int nd)
{
    int i, n;
    n = 0;
    for (i = nd - 1; i >= 0; i--) {
        int k = i;
        n = (n * l[k]) + x[k];
    }
    return n;
}

// TODO: fix
template <typename T>
double norm_2_E(const T s)
{

    std::complex<double> s1(0.0, 0.0);
    for (int i = 0; i < (s.size / 2); i++) {
        s1 += s.A[i] * conj(s.A[i]);
    }
    return s1.real();
}

// TODO: fix
template <typename T>
double norm_2_O(const T s)
{

    std::complex<double> s1(0.0, 0.0);
    for (int i = s.size / 2; i < s.size; i++) {
        s1 += s.A[i] * conj(s.A[i]);
    }
    return s1.real();
};

static int site2index(int *x, int *L)
{
    int index = x[0] % L[0] + L[0] * (x[1] % L[1] + L[1] * (x[2] % L[2] + L[2] * (x[3] % L[3])));
    return index;
}

static int site2eoindex(int *x, int *L)
{
    int Vh = L[0] * L[1] * L[2] * L[3] / 2;
    int index = site2index(x, L);
    return index / 2 + (x[0] + x[1] + x[2] + x[3]) % 2 * Vh;
}

static void index2site(int idx, int x[4], int *L)
{
    x[0] = idx % L[0];
    x[1] = (idx / L[0]) % L[1];
    x[2] = (idx / (L[0] * L[1])) % L[2];
    x[3] = (idx / (L[0] * L[1] * L[2])) % L[3];
}

#endif //LATTICECHINA_OPERATOR_H
