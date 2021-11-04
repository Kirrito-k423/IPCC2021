/**
** @file:  main.cpp
** @brief:
**/

#include <mpi.h>
#include <time.h>
#include <complex>
#include "load_gauge.h"
#include "invert.h"
#include "check.h"
#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    // params
    MPI_Init(&argc, &argv);
    int rank, nsize;
    MPI_Comm_size(MPI_COMM_WORLD, &nsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int site_vec[4] = {8, 8, 8, 8}; // 总格子大小； site_vec = { Lt Lx Ly Lz }；
    int subgs[4] = {8, 8, 8, 8};    // 子格子大小, 进程数 = \prod_{i=0}^3 subgs[i] ;
    double Mass = 0.05;
    string filename = "UNIT";
    int seedr = 0;

    if (argc == 11 || argc == 12) {
        Mass = stod(argv[1]);
        filename = string(argv[2]);
        for (int i = 0, bias1 = 3, bias2 = 7; i < 4; ++i) {
            site_vec[i] = stoi(string(argv[i + bias1]));
            subgs[i] = stoi(string(argv[i + bias2]));
        }
        if (argc == 12)
            seedr = atoi(argv[11]);
    } else if (argc != 1) {
        cerr << "mpirun -n mpi_size ./main Mass gaugefile  lattice_size   subgrid_size " << endl;
        cerr << "Example: mpirun -n 4 ./main 0.005 ./data/config.dat  24 24 24 48  24 24 12 24"
             << endl;
        return 1;
    }

    int MaxCG = 500;
    double Accuracy = 1.0e-9;

    int N_sub[4];
    int V = 1;
    int Vsub = 1;
    int Nsub_size = 1;
    for (int i = 0; i < 4; i++) {
        N_sub[i] = site_vec[i] / subgs[i];
        V *= site_vec[i];
        Vsub *= subgs[i];
        Nsub_size *= N_sub[i];
    }
    if (Nsub_size != nsize) {
        if (rank == 0) {
            cerr << "Lattice partition size " << Nsub_size << " != MPI size " << nsize << endl;
        }
        MPI_Finalize();
        return 1;
    } else {
        if (rank == 0) {
            cout << "  Mass  " << Mass << endl;
            cout << "  gauge file  " << filename << endl;
            printf("  lattice size  %d %d %d %d\n", site_vec[0], site_vec[1], site_vec[2],
                   site_vec[3]);
            printf("  subgrid size  %d %d %d %d\n", subgs[0], subgs[1], subgs[2], subgs[3]);
        }
    }

    // set seed
    int seed = rank + seedr;
    srand(seed);

    // make source
    // lattice_fermion src(subgs, site_vec);
    double *source = new double[Vsub * 3 * 4 * 2];
    for (int i = 0; i < Vsub * 3 * 4 * 2; i++) {
        source[i] = (double) rand() / RAND_MAX;
    }

    // WALL TIME START;
    double start_t = MPI_Wtime();

    // make source
    lattice_fermion src(subgs, site_vec);
    for (int i = 0; i < Vsub * 3 * 4; i++) {
        src.A[i] = complex<double>(source[2 * i], source[2 * i + 1]);
    }

    // load gauge:
    lattice_gauge U(subgs, site_vec);
    load_gauge(U.A, filename, subgs, site_vec);
    double gauge_t = MPI_Wtime();

    // solve: M^\dagger M * dest = M^\dagger src
    lattice_fermion dest(subgs, site_vec);
    CGinvert(src, dest, U, Mass, MaxCG, Accuracy);

    // WALL TIME END;
    double end_t = MPI_Wtime();
    if (rank == 0) {
        cout << "Source and Gauge, time: " << gauge_t - start_t << endl;
        cout << "CG invert, time: " << end_t - gauge_t << endl;
        cout << "Total time: " << end_t - start_t << endl;
        // check: |M * dest - src|
        cout << "Result checking ..." << endl;
        cout << " [ Accuracy = " << Accuracy << " ]" << endl;
    }

    // Check
    check(src, dest, U, Mass);

    delete[] source;
    MPI_Finalize();

    return 0;
}
