/**
** @file:  invert.cpp
** @brief:
**/

#include "invert.h"
#include "operator_mpi.h"
#include <stdlib.h>
#include <math.h>
#include "operator.h"
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

     // sublattice
    int N_sub[4] = {src.site_vec[0] / src.subgs[0], src.site_vec[1] / src.subgs[1],
                    src.site_vec[2] / src.subgs[2], src.site_vec[3] / src.subgs[3]};

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int site_x_f[4] = {(rank % N_sub[0] + 1) % N_sub[0], (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_x_b[4] = {(rank % N_sub[0] - 1 + N_sub[0]) % N_sub[0], (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_x_b = get_nodenum(site_x_b, N_sub, 4);
    const int nodenum_x_f = get_nodenum(site_x_f, N_sub, 4);

    int site_y_f[4] = {(rank % N_sub[0]), ((rank / N_sub[0]) % N_sub[1] + 1) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_y_b[4] = {(rank % N_sub[0]), ((rank / N_sub[0]) % N_sub[1] - 1 + N_sub[1]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_y_b = get_nodenum(site_y_b, N_sub, 4);
    const int nodenum_y_f = get_nodenum(site_y_f, N_sub, 4);

    int site_z_f[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       ((rank / (N_sub[1] * N_sub[0])) % N_sub[2] + 1) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    int site_z_b[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       ((rank / (N_sub[1] * N_sub[0])) % N_sub[2] - 1 + N_sub[2]) % N_sub[2],
                       rank / (N_sub[2] * N_sub[1] * N_sub[0])};

    const int nodenum_z_b = get_nodenum(site_z_b, N_sub, 4);
    const int nodenum_z_f = get_nodenum(site_z_f, N_sub, 4);

    int site_t_f[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) + 1) % N_sub[3]};

    int site_t_b[4] = {(rank % N_sub[0]), (rank / N_sub[0]) % N_sub[1],
                       (rank / (N_sub[1] * N_sub[0])) % N_sub[2],
                       (rank / (N_sub[2] * N_sub[1] * N_sub[0]) - 1 + N_sub[3]) % N_sub[3]};

    const int nodenum_t_b = get_nodenum(site_t_b, N_sub, 4);
    const int nodenum_t_f = get_nodenum(site_t_f, N_sub, 4);


    int subgrid[4] = {src.subgs[0], src.subgs[1], src.subgs[2], src.subgs[3]};
    int subgrid_vol = (subgrid[0] * subgrid[1] * subgrid[2] * subgrid[3]);
    // int subgrid_vol_cb = (subgrid_vol) >> 1;
    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * subgrid[1] +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * subgrid[2] +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * subgrid[3];



// void Dslashoffd(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag,
//                 int cb,
//                 int * N_sub,int rank, int size, 
//                 int * site_x_f, int * site_x_b, int nodenum_x_b, int nodenum_x_f, 
//                 int * site_y_f, int * site_y_b, int nodenum_y_b, int nodenum_y_f, 
//                 int * site_z_f, int * site_z_b, int nodenum_z_b, int nodenum_z_f, 
//                 int * site_t_f, int * site_t_b, int nodenum_t_b, int nodenum_t_f, 
//                 int subgrid_vol, int x_p
//                 )
    Dslash(src, Mdb, U, mass, true,
        N_sub, rank,  size, 
        nodenum_x_b,  nodenum_x_f, 
        nodenum_y_b,  nodenum_y_f, 
        nodenum_z_b,  nodenum_z_f, 
        nodenum_t_b,  nodenum_t_f, 
        subgrid_vol, x_p
    );
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
    Dslash(dest, tmp, U, mass, false,
        N_sub, rank,  size, 
        nodenum_x_b,  nodenum_x_f, 
        nodenum_y_b,  nodenum_y_f, 
        nodenum_z_b,  nodenum_z_f, 
        nodenum_t_b,  nodenum_t_f, 
        subgrid_vol, x_p
    );
    Dslash(tmp, r0, U, mass, true,
        N_sub, rank,  size, 
        nodenum_x_b,  nodenum_x_f, 
        nodenum_y_b,  nodenum_y_f, 
        nodenum_z_b,  nodenum_z_f, 
        nodenum_t_b,  nodenum_t_f, 
        subgrid_vol, x_p
    );

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
        Dslash(p, qq, U, mass, false,
            N_sub, rank,  size, 
            nodenum_x_b,  nodenum_x_f, 
            nodenum_y_b,  nodenum_y_f, 
            nodenum_z_b,  nodenum_z_f, 
            nodenum_t_b,  nodenum_t_f, 
            subgrid_vol, x_p
        );
        Dslash(qq, q, U, mass, true,
            N_sub, rank,  size, 
            nodenum_x_b,  nodenum_x_f, 
            nodenum_y_b,  nodenum_y_f, 
            nodenum_z_b,  nodenum_z_f, 
            nodenum_t_b,  nodenum_t_f, 
            subgrid_vol, x_p
        );
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
            const bool dagger,
            int * N_sub,int rank, int size, 
            const int nodenum_x_b, const int nodenum_x_f, 
            const int nodenum_y_b, const int nodenum_y_f, 
            const int nodenum_z_b, const int nodenum_z_f, 
            const int nodenum_t_b, const int nodenum_t_f, 
            int subgrid_vol, const int x_p
                )
{
    dest.clean();
    lattice_fermion tmp(src.subgs, src.site_vec);
    DslashEE(src, tmp, mass);
    dest = dest + tmp;
    DslashOO(src, tmp, mass);
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 0,
        N_sub, rank,  size, 
        nodenum_x_b,  nodenum_x_f, 
        nodenum_y_b,  nodenum_y_f, 
        nodenum_z_b,  nodenum_z_f, 
        nodenum_t_b,  nodenum_t_f, 
        subgrid_vol, x_p
    ); // cb=0, EO
    dest = dest + tmp;
    Dslashoffd(src, tmp, U, dagger, 1, 
        N_sub, rank,  size, 
        nodenum_x_b,  nodenum_x_f, 
        nodenum_y_b,  nodenum_y_f, 
        nodenum_z_b,  nodenum_z_f, 
        nodenum_t_b,  nodenum_t_f, 
        subgrid_vol, x_p
    );
    dest = dest + tmp;
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
