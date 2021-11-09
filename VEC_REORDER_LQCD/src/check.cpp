
/**
** @file:  check.cpp
** @brief:
**/
#include "check.h"
#include "lattice_fermion.h"
#include "lattice_gauge.h"
#include "dslash.h"
#include "invert.h"
#include "operator_mpi.h"
#include "operator.h"
#include <math.h>
#include <complex>
#include <iostream>
using namespace std;

int check(complex<double> *src_p, complex<double> *chi_p, complex<double> *gauge[4],
          const double mass, int *subgs, int *site_vec)
{
    lattice_fermion src(src_p, subgs, site_vec);
    lattice_fermion chi(chi_p, subgs, site_vec);
    lattice_gauge U(gauge, subgs, site_vec);
    check(src, chi, U, mass);
    return 0;
}

int check(lattice_fermion &src, lattice_fermion &chi, lattice_gauge &U, double mass)
{
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // x= chi; b=src;
    lattice_fermion Mx(chi.subgs, chi.site_vec);
    lattice_fermion Mdx(chi.subgs, chi.site_vec);
    lattice_fermion MMx(chi.subgs, chi.site_vec);
    lattice_fermion Mb(chi.subgs, chi.site_vec);
    lattice_fermion Mdb(chi.subgs, chi.site_vec);
    lattice_fermion MMb(chi.subgs, chi.site_vec);
    lattice_fermion tmp(chi.subgs, chi.site_vec);

    double nrm_b = sqrt(norm_2(src));
    double nrm_x = sqrt(norm_2(chi));
    // Mdb = M^\dagger b
    Dslash(src, Mb, U, mass, false);
    Dslash(src, Mdb, U, mass, true);
    Dslash(Mb, MMb, U, mass, true);

    // rsd2 = |Mdx - b| := | M^\dagger x -b|
    Dslash(chi, Mdx, U, mass, true);
    Minus(Mdx, src, tmp);
    double rsd2 = sqrt(norm_2(tmp));
    // rsd1 = |Mx - b|;
    Dslash(chi, Mx, U, mass, false);
    Minus(Mx, src, tmp);
    double rsd1 = sqrt(norm_2(tmp));
    // rsd = |M\dagger M x - Mdb|;
    Dslash(Mx, MMx, U, mass, true); // MMx = M^\dagger M x;
    Minus(MMx, Mdb, tmp);
    double rsd = sqrt(norm_2(tmp));

    // lattice_fermion g5x(chi.subgs, chi.site_vec);
    // gamma5_lattcefermion(chi, g5x);
    // Dslash(g5x, tmp, U, mass, true);
    // gamma5_lattcefermion(tmp, g5x);
    // // Minus(g5x, src, tmp);
    // double g5Mdg5x = sqrt(norm_2(g5x));
    complex<double> MbMb = vector_p(Mb, Mb);
    complex<double> bMMb = vector_p(src, MMb);
    complex<double> MxMx = vector_p(Mx, Mx);
    complex<double> xMMx = vector_p(chi, MMx);
    if (myrank == 0) {
        cout << "===================================" << endl;
        cout << "|b| = " << nrm_b << "    |x| = " << nrm_x << endl;
        cout << "(Mb, Mb) = " << MbMb << "    (b, MMb) = " << bMMb << endl;
        cout << "(Mx, Mx) = " << MxMx << "    (x, MMx) = " << xMMx << endl;
        cout << "|M x -b| = " << rsd1 << endl;
        cout << "|M^dagger M x - Mdb| = " << rsd << endl;
        cout << "===================================" << endl;
    }

#ifdef DEBUG

    Dslash(src, Mb, U, mass, false);
    double nrm_Mb = norm_2(Mb);
    double nrm_Mdb = norm_2(Mdb);
    double nrm_MMb = norm_2(MMb);
    double nrm_Mx = norm_2(Mx);
    double nrm_Mdx = norm_2(Mdx);
    double nrm_MMx = norm_2(MMx);
    double nrm_diffb = norm_2(Mb - Mdb);
    double nrm_sumb = norm_2(Mb + Mdb);
    double nrm_diffx = norm_2(Mdx + Mx);
    double nrm_sumx = norm_2(Mdx - Mx);
    if (myrank == 0) {
        // cout << "|5Md5 x -b| = " << g5Mdg5x << endl;
        // cout << "|M^dagger x -b| = " << rsd2 << endl;
        cout << "|Mb| = " << sqrt(nrm_Mb) << "    |Mdb| = " << sqrt(nrm_Mdb)
             << "  |MMb| = " << sqrt(nrm_MMb) << endl;
        cout << "|Mx| = " << sqrt(nrm_Mx) << "    |Mdx| = " << sqrt(nrm_Mdx)
             << "    |MMx| = " << sqrt(nrm_MMx) << endl;
        cout << "|Mb - Mdb| = " << sqrt(nrm_diffb) << "    |Mb + Mdb| = " << sqrt(nrm_sumb) << endl;
        cout << "|Mx - Mdx| = " << sqrt(nrm_diffx) << "    |Mx + Mdx| = " << sqrt(nrm_sumx) << endl;
        cout << "  b:=src   Mx:= M*chi " << endl;
        for (int i = 0; i < 12; i++) {
            cout << "  " << src.A[i] << "    " << Mx.A[i] << endl;
        }
    }
#endif
    return 0;
}

void check_gauge_unitary(complex<double> *U)
{
    complex<double> Usrc[3][3] = {0};
    for (int i = 0; i < 9; i++) {
        Usrc[i / 3][i % 3] = U[i];
    }
    cout << "U  = " << endl;
    cout << Usrc[0][0] << "  " << Usrc[0][1] << "  " << Usrc[0][2] << endl;
    cout << Usrc[1][0] << "  " << Usrc[1][1] << "  " << Usrc[1][2] << endl;
    cout << Usrc[2][0] << "  " << Usrc[2][1] << "  " << Usrc[2][2] << endl;
    complex<double> Ures[3][3] = {0};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                Ures[i][j] += Usrc[i][k] * conj(Usrc[j][k]);
            }
        }
    }
    cout << "U U^dagger = " << endl;
    cout << Ures[0][0] << "  " << Ures[0][1] << "  " << Ures[0][2] << endl;
    cout << Ures[1][0] << "  " << Ures[1][1] << "  " << Ures[1][2] << endl;
    cout << Ures[2][0] << "  " << Ures[2][1] << "  " << Ures[2][2] << endl << endl;
}

void check_plaq(lattice_gauge &U)
{
}

void gamma5_lattcefermion(lattice_fermion &src, lattice_fermion &dest)
{
    for (int i = 6; i < src.size; i += 6) {
        for (int k = 0; k < 6; k++) {
            dest.A[i + k] = std::complex<double>(-1.0, 0) * src.A[i + k];
        }
    }
}

void check_lattice_site(int *subgs, int *site_vec)
{
    int site[4];
    int coords[4];
    int N_sub[4];
    for (int i = 0; i < 4; i++)
        N_sub[i] = site_vec[i] / subgs[i];
    int nodenum;

    for (int t = 0; t < site_vec[3]; t++) {
        site[3] = t;
        for (int z = 0; z < site_vec[2]; z++) {
            site[2] = z;
            for (int y = 0; y < site_vec[1]; y++) {
                site[1] = y;
                for (int x = 0; x < site_vec[0]; x++) {
                    site[0] = x;
                    for (int k = 0; k < 4; k++) {
                        coords[k] = site[k] / subgs[k];
                    }
                    nodenum = get_nodenum(coords, N_sub, 4);
                    cout << "(" << x << " " << y << " " << z << " " << t
                         << ") ,    node_num= " << nodenum << endl;
                }
            }
        }
    }
}
