/**
** @file:  load_gauge.cpp
** @brief:
**/

#include "mpi.h"
#include "load_gauge.h"
#include "lattice_gauge.h"
#include "check.h"
#include "operator.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include "sys/stat.h"
using namespace std;

void endian_swap(double *a)
{
    char tmp[8];
    char *p = (char *) a;
    for (int i = 0; i < 8; i++) {
        tmp[i] = p[7 - i];
    }
    for (int i = 0; i < 8; i++) {
        p[i] = tmp[i];
    }
}

int load_gauge(std::complex<double> *gauge[4], std::string &filename, int *subgs, int *site_vec)
{
    int Vsub = 1;
    int N_sub[4] = {0};
    for (int i = 0; i < 4; i++) {
        N_sub[i] = site_vec[i] / subgs[i];
        // V *= site_vec[i];
        Vsub *= subgs[i];
        // Nsub_size *= N_sub[i];
    }

    if (filename == "UNIT") {
        std::complex<double> I(1.0, 0);
        for (int i = 0; i < 4; i++) {
            for (int c = 0; c < Vsub * 9; c = c + 9) {
                gauge[i][c + 0] = I;
                gauge[i][c + 4] = I;
                gauge[i][c + 8] = I;
            }
        }
#ifdef DEBUG
        // check unitary
        check_gauge_unitary(&gauge[0][9 * (rand() % Vsub)]);
#endif
    } else if (filename == "RANDOM") {
        /*
         * random gauge;
         */
    } else {
        // read from conf.file
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int rank_i[4];
        index2site(rank, rank_i, N_sub);
        int xg0[4];
        for (int i = 0; i < 4; i++) {
            xg0[i] = rank_i[i] * subgs[i];
        }

        long endinfo = 0;
        long offset = 0;
#ifdef LIME
        endinfo = 280;
        struct stat statbuff;
        stat(&filename[0], &statbuff);
        bool double_flag = false;
        long size = 9 * 4 * 2 * 4; // single
        long all_size = size;
        for (int i = 0; i < 4; i++)
            all_size *= site_vec[i];
        offset = statbuff.st_size - all_size;

        if (offset > statbuff.st_size / 2) {
            if (rank == 0)
                printf("The configuration is in double precision\n");
            size *= 2;
            all_size *= 2;
            offset = statbuff.st_size - all_size;
            double_flag = true;
        }
        if (rank == 0)
            printf("offset=%15ld,%15ld\n", offset - endinfo, statbuff.st_size);
#endif

        int Offset0 = offset - endinfo;
        int Offset = Offset0;
        double *U4 = new double[Vsub * 72];
        int xg[4] = {0}; // global site;
        int xs[4] = {0}; // local (sub) site;
        int idx_sub = 0;
#ifndef NO_MPI_IO
        MPI_File fh;
        MPI_Status status;
        MPI_File_open(MPI_COMM_WORLD, &filename[0], MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
#else
        FILE *fh = fopen(&filename[0], "rb");
        if (!fh) {
            cerr << "error: can not open file: " << filename << endl;
            exit(EXIT_FAILURE);
        }
#endif
        for (xs[3] = 0; xs[3] < subgs[3]; xs[3]++) {
            xg[3] = xg0[3] + xs[3];
            for (xs[2] = 0; xs[2] < subgs[2]; xs[2]++) {
                xg[2] = xg0[2] + xs[2];
                for (xs[1] = 0; xs[1] < subgs[1]; xs[1]++) {
                    xg[1] = xg0[1] + xs[1];
                    idx_sub = site2index(xs, subgs);
                    Offset = Offset0 + 72 * sizeof(double) * site2index(xg, site_vec);
#ifndef NO_MPI_IO
                    MPI_File_seek(fh, Offset, MPI_SEEK_SET);
                    MPI_File_read(fh, &U4[idx_sub * 72], 72 * subgs[0], MPI_DOUBLE, &status);
                    // MPI_File_read_at_all(fh, Offset, &U4[idx_sub * 72], 72 * subgs[0], MPI_DOUBLE,
                    //  &status);
#else
                    fseek(fh, Offset, SEEK_SET);
                    fread(&U4[idx_sub * 72], 72 * subgs[0] * sizeof(double), 1, fh);
#endif
                }
            }
        }

        for (int i = 0; i < Vsub * 72; i++) {
            endian_swap(&U4[i]);
            // cout << U4[i] << endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        // fh.close();
        // MPI_File_close(&fh);

        // eo-prec
        int count = 0;
        int idx, idxeo;
        for (xs[3] = 0; xs[3] < subgs[3]; xs[3]++) {
            for (xs[2] = 0; xs[2] < subgs[2]; xs[2]++) {
                for (xs[1] = 0; xs[1] < subgs[1]; xs[1]++) {
                    for (xs[0] = 0; xs[0] < subgs[0]; xs[0]++) {
                        for (int i = 0; i < 4; i++) {
                            idx = site2index(xs, subgs) * 72 + i * 18;
                            idxeo = site2eoindex(xs, subgs) * 9;
                            for (int c = 0; c < 9; c++) {
                                // gauge[i][idxeo + c] =
                                // complex<double>(U4[idx + 2 * c], U4[idx + 2 * c + 1]);
                                gauge[i][idxeo + c].real(U4[idx + 2 * c]);
                                gauge[i][idxeo + c].imag(U4[idx + 2 * c + 1]);
                            }
                        }
                    }
                }
            }
        }

#ifdef DEBUG
        // check unitary
        // check_gauge_unitary(&gauge[0][9 * (rand() % Vsub)]);
        int t0 = 59;
        if (rank == t0 / subgs[3]) {
            int sx[4] = {0, 0, 0, t0 % subgs[3]};
            for (int n = 0; n < 2; n++) {
                sx[0] = n;
                int sidx = 9 * site2eoindex(sx, subgs);
                for (int d = 0; d < 4; d++)
                    check_gauge_unitary(&gauge[d][sidx]);
            }
        }
#endif
        delete[] U4;
    }

    return 0;
}
