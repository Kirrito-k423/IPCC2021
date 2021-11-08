/**
** @file:  invert.cpp
** @brief:
**/

#include "invert.h"
#include "operator_mpi.h"
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;

int CGinvert(complex<double> *src_p, complex<double> *dest_p, complex<double> *gauge[4],
             const double mass, const int max, const double accuracy, int *subgs, int *site_vec)
{
    lattice_gauge U(gauge, subgs, site_vec);
    lattice_fermion src(src_p, subgs, site_vec);
    lattice_fermion dest(dest_p, subgs, site_vec);
    CGinvert(src, dest, U, mass, max, accuracy);
    return 0;
}

int CGinvert(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    lattice_fermion r0(src.subgs, src.site_vec);
    lattice_fermion r1(src.subgs, src.site_vec);
    lattice_fermion q(src.subgs, src.site_vec);
    lattice_fermion qq(src.subgs, src.site_vec);
    lattice_fermion p(src.subgs, src.site_vec);
    lattice_fermion Mdb(src.subgs, src.site_vec);
    lattice_fermion tmp(src.subgs, src.site_vec);

    complex<double> aphi(0);
    complex<double> beta(0);

    Dslash(src, Mdb, U, mass, true);
    for (int i = 0; i < dest.size; i++) {
        dest.A[i] = 1.0 * rand() / RAND_MAX;
    }
    // dest.A[0] = 1.0;
#ifdef DEBUG
    double src_nr2 = norm_2(src);
    double dest_nr2 = norm_2(dest);
    if (myrank == 0) {
        cout << "|src|^2 = " << src_nr2 << " , |dest|^2 = " << dest_nr2 << endl;
    }
#endif
    Dslash(dest, tmp, U, mass, false);
    Dslash(tmp, r0, U, mass, true);

    for (int i = 0; i < Mdb.size; i++) {
        r0.A[i] = Mdb.A[i] - r0.A[i];
    }

    for (int f = 1; f < max; f++) {
        if (f == 1) {
            for (int i = 0; i < r0.size; i++)
                p.A[i] = r0.A[i];
        } else {
            beta = vector_p(r0, r0) / vector_p(r1, r1);
            for (int i = 0; i < r0.size; i++)
                p.A[i] = r0.A[i] + beta * p.A[i];
        }

        Dslash(p, qq, U, mass, false);
        Dslash(qq, q, U, mass, true);

        aphi = vector_p(r0, r0) / vector_p(p, q);

        for (int i = 0; i < dest.size; i++)
            dest.A[i] = dest.A[i] + aphi * p.A[i];
        for (int i = 0; i < r1.size; i++)
            r1.A[i] = r0.A[i];
        for (int i = 0; i < r0.size; i++)
            r0.A[i] = r0.A[i] - aphi * q.A[i];
        double rsd2 = norm_2(r0);
        double rsd = sqrt(rsd2);
        if (rsd < accuracy) {
            if (myrank == 0) {
                cout << "CG: " << f << " iterations, convergence residual |r| = " << rsd << endl;
            }
            break;
        }
#ifndef VERBOSE_SIMPLE
        if (myrank == 0) {
            cout << "CG: " << f << " iter, rsd |r| = " << rsd << endl;
        }
#endif
    }
    return 0;
}
// dest 是要求的
int CGinvert2(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    lattice_fermion r0(src.subgs, src.site_vec);
    lattice_fermion r1(src.subgs, src.site_vec);
    lattice_fermion q(src.subgs, src.site_vec);
    lattice_fermion qq(src.subgs, src.site_vec);
    lattice_fermion p(src.subgs, src.site_vec);
    lattice_fermion Mdb(src.subgs, src.site_vec);
    lattice_fermion tmp(src.subgs, src.site_vec);

    complex<double> aphi(0);
    complex<double> beta(0);

    Dslash_tilde(src, Mdb, U, mass, true);
    for (int i = 0; i < dest.size; i++) {
        dest.A[i] = 1.0 * rand() / RAND_MAX;
    }
    // dest.A[0] = 1.0;
#ifdef DEBUG
    double src_nr2 = norm_2(src);
    double dest_nr2 = norm_2(dest);
    if (myrank == 0) {
        cout << "|src|^2 = " << src_nr2 << " , |dest|^2 = " << dest_nr2 << endl;
    }
#endif
    Dslash_tilde(dest, tmp, U, mass, false);
    Dslash_tilde(tmp, r0, U, mass, true);

    for (int i = 0; i < Mdb.size; i++) {
        r0.A[i] = Mdb.A[i] - r0.A[i];
    }

    for (int f = 1; f < max; f++) {
        if (f == 1) {
            for (int i = 0; i < r0.size; i++)
                p.A[i] = r0.A[i];
        } else {
            beta = vector_p(r0, r0) / vector_p(r1, r1);
            for (int i = 0; i < r0.size; i++)
                p.A[i] = r0.A[i] + beta * p.A[i];
        }

        Dslash_tilde(p, qq, U, mass, false);
        Dslash_tilde(qq, q, U, mass, true);

        aphi = vector_p(r0, r0) / vector_p(p, q);

        for (int i = 0; i < dest.size; i++)
            dest.A[i] = dest.A[i] + aphi * p.A[i];
        for (int i = 0; i < r1.size; i++)
            r1.A[i] = r0.A[i];
        for (int i = 0; i < r0.size; i++)
            r0.A[i] = r0.A[i] - aphi * q.A[i];
        double rsd2 = norm_2(r0);
        double rsd = sqrt(rsd2);
        if (rsd < accuracy) {
            if (myrank == 0) {
                cout << "CG: " << f << " iterations, convergence residual |r| = " << rsd << endl;
            }
            break;
        }
#ifndef VERBOSE_SIMPLE
        if (myrank == 0) {
            cout << "CG: " << f << " iter, rsd |r| = " << rsd << endl;
        }
#endif
    }
    return 0;
}

void Dslash(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger)
{
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    DslashEE(src, tmp, mass);
    dest = dest + tmp;
    DslashOO(src, tmp, mass);
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 0); // cb=0, EO
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 1);
    dest = dest + tmp;
}

void Dslash_tilde(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger)
{
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    lattice_fermion tmp1(src.subgs, src.site_vec);
    lattice_fermion tmp2(src.subgs, src.site_vec);
    lattice_fermion tmp3(src.subgs, src.site_vec);
    DslashEE(src, tmp, mass);
    dest = dest + tmp;
    DslashOO(src, tmp, mass);
    dest = dest + tmp;

    // tmp = Meo * X
    Dslashoffd(src, tmp, U, dagger, 0); // cb=0, EO 
    
    // tmp1 = Mee^(-1) * Meo * X
    DslashEEInv(tmp, tmp1, mass);
    
    // tmp2 = Moe * Mee^(-1) * Meo * X
    Dslashoffd(tmp1, tmp2, U, dagger, 1); // cb = 1, OE
    
    //但是tmp2 在上面。
    UpLowChange(tmp2,tmp3);
    dest = dest - tmp3;
}

void DslashEEInv(lattice_fermion &src, lattice_fermion &dest, const double mass){
    dest.clean();
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    for (int i = 0; i < subgrid_vol * 3 * 4; i++) {
        dest.A[i] = src.A[i] / (4 + mass);
    }
}

void UpLowChange(lattice_fermion &src, lattice_fermion &dest){
    dest.clean();
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    for (int i = 0; i < subgrid_vol * 3 * 4 / 2; i++) {
        dest.A[i] = src.A[i + subgrid_vol_cb];
        dest.A[i + subgrid_vol_cb] = src.A[i];
    }
}
// void Dslash_tilde_Low(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
//             const bool dagger)
// {
//     dest.clean();
//     lattice_fermion tmp(src.subgs, src.site_vec);
//     DslashEE(src, tmp, mass);
//     dest = dest + tmp;
//     DslashOO(src, tmp, mass);
//     dest = dest + tmp;
//     Dslashoffd(src, tmp, U, dagger, 0); // cb=0, EO
//     // tmp = Meo * X
//     DslashEEInv(tmp, tmp1, mass);
//     // tmp1 = Mee^(-1) * Meo * X
//     Dslashoffd_offset(tmp1, tmp2, U, dagger, 1); // cb = 1, OE
//     // tmp2 = Moe * Mee^(-1) * Meo * X
//     dest = dest - tmp2;
//     // Dslashoffd(src, tmp, U, dagger, 1);
//     // dest = dest + tmp;
// }
void LInv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass){

    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    LEE(src, tmp);
    dest = dest + tmp;
    LOO(src, tmp);
    dest = dest + tmp;
    L_OE_EE_Inv(src, tmp, U, mass);
    dest = dest - tmp; //逆矩阵

}

void LEE(lattice_fermion &src, lattice_fermion &dest)
{

    dest.clean();
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    for (int i = 0; i < subgrid_vol_cb * 3 * 4; i++) {
        dest.A[i] = (1) * src.A[i];
    }
}

void LOO(lattice_fermion &src, lattice_fermion &dest)
{
    dest.clean();
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++) {
        dest.A[i] = (1) * src.A[i];
    }
}

void L_OE_EE_Inv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass){
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    for (int i = 0; i < subgrid_vol * 3 * 4; i++) {
        tmp.A[i] = src.A[i] / (4 + mass);
    }
    Dslashoffd(tmp, dest, U, 0, 1); // cb=1 OE
}

void UInv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass){

    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    UEE(src, tmp);
    dest = dest + tmp;
    UOO(src, tmp);
    dest = dest + tmp;
    U_EE_Inv_EO(src, tmp, U, mass);
    dest = dest - tmp; //逆矩阵

}

void UEE(lattice_fermion &src, lattice_fermion &dest)
{

    dest.clean();
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    for (int i = 0; i < subgrid_vol_cb * 3 * 4; i++) {
        dest.A[i] = (1) * src.A[i];
    }
}

void UOO(lattice_fermion &src, lattice_fermion &dest)
{
    dest.clean();
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++) {
        dest.A[i] = (1) * src.A[i];
    }
}

void U_EE_Inv_EO(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass){
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    Dslashoffd(src, tmp, U, 0, 0); // cb=0 EO
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    for (int i = 0; i < subgrid_vol * 3 * 4; i++) {
        dest.A[i] = tmp.A[i] / (4 + mass);
    }
}