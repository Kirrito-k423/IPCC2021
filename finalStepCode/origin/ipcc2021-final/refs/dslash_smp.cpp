/**
** @file:  dslash.cpp
** @brief: Dslash and Dlash-dagger operaters.
**/

#include <mpi.h>
#include "dslash.h"
#include "operator.h"
using namespace std;

void DslashEE(lattice_fermion &src, lattice_fermion &dest, const double mass)
{

    dest.clean();
    const double a = 4.0;
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    for (int i = 0; i < subgrid_vol_cb * 3 * 4; i++) {
        dest.A[i] = (a + mass) * src.A[i];
    }
}

void DslashOO(lattice_fermion &src, lattice_fermion &dest, const double mass)
{

    dest.clean();
    const double a = 4.0;
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++) {
        dest.A[i] = (a + mass) * src.A[i];
    }
}

// cb = 0  EO  ;  cb = 1 OE
void Dslashoffd(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag,
                int cb)
{
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

    dest.clean();
    double flag = (dag == true) ? -1 : 1;

    int subgrid[4] = {src.subgs[0], src.subgs[1], src.subgs[2], src.subgs[3]};
    int subgrid_vol = (subgrid[0] * subgrid[1] * subgrid[2] * subgrid[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;
    subgrid[0] >>= 1;
    const double half = 0.5;
    const complex<double> I(0, 1);
    const int x_p = ((rank / N_sub[0]) % N_sub[1]) * subgrid[1] +
                    ((rank / (N_sub[1] * N_sub[0])) % N_sub[2]) * subgrid[2] +
                    (rank / (N_sub[2] * N_sub[1] * N_sub[0])) * subgrid[3];

    MPI_Request reqs[8 * size];
    MPI_Request reqr[8 * size];
    MPI_Status stas[8 * size];
    MPI_Status star[8 * size];

    // x方向：向后（x-u方向）传输 (1-\gamma_u) (\psi[x_begin-\mu)]
    int len_x_f = (subgrid[1] * subgrid[2] * subgrid[3] + cb) >> 1;

    double *resv_x_f = new double[len_x_f * 6 * 2];
    double *send_x_b = new double[len_x_f * 6 * 2];
    if (N_sub[0] != 1) {
        for (int i = 0; i < len_x_f * 6 * 2; i++) {
            send_x_b[i] = 0;
        }

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    // cb = 0, Meo So, xg[4] 得是o的 （1-cb）
                    if ((y + z + t + x_p) % 2 == cb) {
                        continue;
                    }
                    // xg[4] 的奇偶性；
                    int x = 0;
                    complex<double> tmp;
                    // src0 ->> src.A_(1-cb)
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;
                    int b = cont * 6;
                    cont += 1;

                    for (int c2 = 0; c2 < 3; c2++) {
                        tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half;
                        send_x_b[b * 2 + (0 * 3 + c2) * 2 + 0] = tmp.real();
                        send_x_b[b * 2 + (0 * 3 + c2) * 2 + 1] = tmp.imag();
                        tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
                        send_x_b[b * 2 + (1 * 3 + c2) * 2 + 0] = tmp.real();
                        send_x_b[b * 2 + (1 * 3 + c2) * 2 + 1] = tmp.imag();
                    }
                }
            }
        }

        MPI_Isend(send_x_b, len_x_f * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * rank, MPI_COMM_WORLD,
                  &reqs[8 * rank]);
        MPI_Irecv(resv_x_f, len_x_f * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * nodenum_x_f,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_x_f]);
    }
    // x方向：向前（x+u方向）传输 (1+\gamma_u) (U_\mu^\dagger[x_end-\mu]) (\psi[x_end-\mu)]
    int len_x_b = (subgrid[1] * subgrid[2] * subgrid[3] + 1 - cb) >> 1;

    double *resv_x_b = new double[len_x_b * 6 * 2];
    double *send_x_f = new double[len_x_b * 6 * 2];

    if (N_sub[0] != 1) {
        for (int i = 0; i < len_x_b * 6 * 2; i++) {
            send_x_f[i] = 0;
        }

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    if (((y + z + t + x_p) % 2) != cb) {
                        continue;
                    }

                    int x = subgrid[0] - 1;

                    complex<double> *AO = U.A[0] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> tmp;

                    int b = cont * 6;
                    cont += 1;
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);

                            send_x_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                            send_x_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();

                            tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);

                            send_x_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                            send_x_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                        }
                    }
                }
            }
        }

        MPI_Isend(send_x_f, len_x_b * 6 * 2, MPI_DOUBLE, nodenum_x_f, 8 * rank + 1, MPI_COMM_WORLD,
                  &reqs[8 * rank + 1]);
        MPI_Irecv(resv_x_b, len_x_b * 6 * 2, MPI_DOUBLE, nodenum_x_b, 8 * nodenum_x_b + 1,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_x_b + 1]);
    }

    //////////////////////////////////////////////////////// no comunication
    /////////////////////////////////////////////////////////
    // x方向
    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            for (int t = 0; t < subgrid[3]; t++) {
                int x_u =
                    ((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? subgrid[0] : subgrid[0] - 1;

                for (int x = 0; x < x_u; x++) {

                    complex<double> *destE;
                    complex<double> *AE;
                    complex<double> tmp;
                    int f_x;
                    if ((y + z + t + x_p) % 2 == cb) {
                        f_x = x;
                    } else {
                        f_x = (x + 1) % subgrid[0];
                    }

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     f_x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            {
                                tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                                      AE[c1 * 3 + c2];
                                destE[0 * 3 + c1] += tmp;
                                destE[3 * 3 + c1] += flag * (I * tmp);
                                tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
                                      AE[c1 * 3 + c2];
                                destE[1 * 3 + c1] += tmp;
                                destE[2 * 3 + c1] += flag * (I * tmp);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            for (int t = 0; t < subgrid[3]; t++) {
                int x_d = (((y + z + t + x_p) % 2) != cb || N_sub[0] == 1) ? 0 : 1;

                for (int x = x_d; x < subgrid[0]; x++) {
                    complex<double> *destE;
                    complex<double> *AO;
                    complex<double> tmp;

                    int b_x;

                    if ((t + z + y + x_p) % 2 == cb) {
                        b_x = (x - 1 + subgrid[0]) % subgrid[0];
                    } else {
                        b_x = x;
                    }

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     b_x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AO = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + b_x + (1 - cb) * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);

                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (-I * tmp);

                            tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);

                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (-I * tmp);
                        }
                    }
                }
            }
        }
    }

    //    printf(" rank =%i  ghost  \n ", rank);

    //////////////////////////////////////////////////////////////////////////////////////ghost//////////////////////////////////////////////////////////////////
    // x方向：接收x前方向节点向后传输的数据 \chi = (1-\gamma_\mu) \psi_f[x_begin-\mu]，并 U_\mu[x_end] \chi
    if (N_sub[0] != 1) {

        MPI_Wait(&reqr[8 * nodenum_x_f], &star[8 * nodenum_x_f]);

        int cont = 0;
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    if ((y + z + t + x_p) % 2 == cb) {
                        continue;
                    }

                    complex<double> *destE;
                    complex<double> *AE;
                    complex<double> tmp;

                    int x = subgrid[0] - 1;

                    complex<double> *srcO = (complex<double> *) (&resv_x_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (I * tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (I * tmp);
                        }
                    }
                }
            }
        }
    } // if(N_sub[0]!=1)

    //    delete[] send_x_b;
    //    delete[] resv_x_f;

    if (N_sub[0] != 1) {

        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &star[8 * nodenum_x_b + 1]);

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                for (int t = 0; t < subgrid[3]; t++) {
                    if (((y + z + t + x_p) % 2) != cb) {
                        continue;
                    }

                    int x = 0;

                    complex<double> *srcO = (complex<double> *) (&resv_x_b[cont * 6 * 2]);
                    cont += 1;

                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++) {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (-I * srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (-I * srcO[1 * 3 + c1]);
                    }
                }
            }
        }
        MPI_Wait(&reqs[8 * rank + 0], &star[8 * nodenum_x_f + 0]);
        MPI_Wait(&reqs[8 * rank + 1], &star[8 * nodenum_x_f + 1]);

    } // if(N_sub[0]!=1)

    //    delete[] send_x_f;
    //    delete[] resv_x_b;

    MPI_Barrier(MPI_COMM_WORLD);

    delete[] send_x_b;
    delete[] resv_x_f;
    delete[] send_x_f;
    delete[] resv_x_b;
}
