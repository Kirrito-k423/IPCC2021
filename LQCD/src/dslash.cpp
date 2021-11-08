/**
** @file:  dslash.cpp
** @brief: Dslash and Dlash-dagger operaters.
**/

#include <mpi.h>
#include <immintrin.h>
#include "dslash.h"
#include "operator.h"
using namespace std;

void DslashEE(lattice_fermion &src, lattice_fermion &dest, const double mass)
{

    dest.clean();
    const double a = 4.0;
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    for (int i = 0; i < subgrid_vol_cb * 3 * 4; i++)
    {
        dest.A[i] = (a + mass) * src.A[i];
    }
}

void DslashOO(lattice_fermion &src, lattice_fermion &dest, const double mass)
{

    dest.clean();
    const double a = 4.0;
    int subgrid_vol = (src.subgs[0] * src.subgs[1] * src.subgs[2] * src.subgs[3]);
    int subgrid_vol_cb = (subgrid_vol) >> 1;

    for (int i = subgrid_vol_cb * 3 * 4; i < subgrid_vol * 3 * 4; i++)
    {
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

    int len_x_f = (subgrid[1] * subgrid[2] * subgrid[3] + cb) >> 1;

    double *resv_x_f = new double[len_x_f * 6 * 2];
    double *send_x_b = new double[len_x_f * 6 * 2];
    if (N_sub[0] != 1)
    {
        for (int i = 0; i < len_x_f * 6 * 2; i++)
        {
            send_x_b[i] = 0;
        }

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {

                    if ((y + z + t + x_p) % 2 == cb)
                    {
                        continue;
                    }
                    int x = 0;
                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;
                    int b = cont * 6;
                    cont += 1;

                    for (int c2 = 0; c2 < 3; c2++)
                    {
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

    int len_x_b = (subgrid[1] * subgrid[2] * subgrid[3] + 1 - cb) >> 1;

    double *resv_x_b = new double[len_x_b * 6 * 2];
    double *send_x_f = new double[len_x_b * 6 * 2];

    if (N_sub[0] != 1)
    {
        for (int i = 0; i < len_x_b * 6 * 2; i++)
        {
            send_x_f[i] = 0;
        }

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    if (((y + z + t + x_p) % 2) != cb)
                    {
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
                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
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

    int len_y_f = subgrid[0] * subgrid[2] * subgrid[3];

    double *resv_y_f = new double[len_y_f * 6 * 2];
    double *send_y_b = new double[len_y_f * 6 * 2];
    if (N_sub[1] != 1)
    {
        for (int i = 0; i < len_y_f * 6 * 2; i++)
        {
            send_y_b[i] = 0;
        }

        int cont = 0;

        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    int y = 0;
                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    int b = cont * 6;
                    cont += 1;

                    for (int c2 = 0; c2 < 3; c2++)
                    {
                        tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half;
                        send_y_b[b * 2 + (0 * 3 + c2) * 2 + 0] = tmp.real();
                        send_y_b[b * 2 + (0 * 3 + c2) * 2 + 1] = tmp.imag();
                        tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
                        send_y_b[b * 2 + (1 * 3 + c2) * 2 + 0] = tmp.real();
                        send_y_b[b * 2 + (1 * 3 + c2) * 2 + 1] = tmp.imag();
                    }
                }
            }
        }

        MPI_Isend(send_y_b, len_y_f * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * rank + 2, MPI_COMM_WORLD,
                  &reqs[8 * rank + 2]);
        MPI_Irecv(resv_y_f, len_y_f * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * nodenum_y_f + 2,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_y_f + 2]);
    }

    int len_y_b = subgrid[0] * subgrid[2] * subgrid[3];

    double *resv_y_b = new double[len_y_b * 6 * 2];
    double *send_y_f = new double[len_y_b * 6 * 2];

    if (N_sub[1] != 1)
    {

        for (int i = 0; i < len_y_b * 6 * 2; i++)
        {
            send_y_f[i] = 0;
        }

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    complex<double> tmp;

                    int y = subgrid[1] - 1;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> *AO = U.A[1] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    int b = cont * 6;
                    cont += 1;
                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {

                            tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            send_y_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                            send_y_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                            tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            send_y_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                            send_y_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                        }
                    }
                }
            }
        }

        MPI_Isend(send_y_f, len_y_b * 6 * 2, MPI_DOUBLE, nodenum_y_f, 8 * rank + 3, MPI_COMM_WORLD,
                  &reqs[8 * rank + 3]);
        MPI_Irecv(resv_y_b, len_y_b * 6 * 2, MPI_DOUBLE, nodenum_y_b, 8 * nodenum_y_b + 3,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_y_b + 3]);
    }

    int len_z_f = subgrid[0] * subgrid[1] * subgrid[3];

    double *resv_z_f = new double[len_z_f * 6 * 2];
    double *send_z_b = new double[len_z_f * 6 * 2];
    if (N_sub[2] != 1)
    {
        for (int i = 0; i < len_z_f * 6 * 2; i++)
        {
            send_z_b[i] = 0;
        }

        int cont = 0;

        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    int z = 0;

                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    int b = cont * 6;
                    cont += 1;

                    for (int c2 = 0; c2 < 3; c2++)
                    {
                        tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half;
                        send_z_b[b * 2 + (0 * 3 + c2) * 2 + 0] += tmp.real();
                        send_z_b[b * 2 + (0 * 3 + c2) * 2 + 1] += tmp.imag();
                        tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half;
                        send_z_b[b * 2 + (1 * 3 + c2) * 2 + 0] += tmp.real();
                        send_z_b[b * 2 + (1 * 3 + c2) * 2 + 1] += tmp.imag();
                    }
                }
            }
        }

        MPI_Isend(send_z_b, len_z_f * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * rank + 4, MPI_COMM_WORLD,
                  &reqs[8 * rank + 4]);
        MPI_Irecv(resv_z_f, len_z_f * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * nodenum_z_f + 4,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_z_f + 4]);
    }

    int len_z_b = subgrid[0] * subgrid[1] * subgrid[3];

    double *resv_z_b = new double[len_z_b * 6 * 2];
    double *send_z_f = new double[len_z_b * 6 * 2];
    if (N_sub[2] != 1)
    {

        for (int i = 0; i < len_z_b * 6 * 2; i++)
        {
            send_z_f[i] = 0;
        }

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    complex<double> tmp;

                    int z = subgrid[2] - 1;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> *AO = U.A[2] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    int b = cont * 6;
                    cont += 1;
                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {

                            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            send_z_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                            send_z_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                            tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            send_z_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                            send_z_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                        }
                    }
                }
            }
        }

        MPI_Isend(send_z_f, len_z_b * 6 * 2, MPI_DOUBLE, nodenum_z_f, 8 * rank + 5, MPI_COMM_WORLD,
                  &reqs[8 * rank + 5]);
        MPI_Irecv(resv_z_b, len_z_b * 6 * 2, MPI_DOUBLE, nodenum_z_b, 8 * nodenum_z_b + 5,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_z_b + 5]);
    }

    int len_t_f = subgrid[0] * subgrid[1] * subgrid[2];

    double *resv_t_f = new double[len_t_f * 6 * 2];
    double *send_t_b = new double[len_t_f * 6 * 2];
    if (N_sub[3] != 1)
    {
        for (int i = 0; i < len_t_f * 6 * 2; i++)
        {
            send_t_b[i] = 0;
        }

        int cont = 0;

        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int z = 0; z < subgrid[2]; z++)
                {
                    int t = 0;

                    complex<double> tmp;
                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    int b = cont * 6;
                    cont += 1;

                    for (int c2 = 0; c2 < 3; c2++)
                    {
                        tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half;
                        send_t_b[b * 2 + (0 * 3 + c2) * 2 + 0] += tmp.real();
                        send_t_b[b * 2 + (0 * 3 + c2) * 2 + 1] += tmp.imag();
                        tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half;
                        send_t_b[b * 2 + (1 * 3 + c2) * 2 + 0] += tmp.real();
                        send_t_b[b * 2 + (1 * 3 + c2) * 2 + 1] += tmp.imag();
                    }
                }
            }
        }

        MPI_Isend(send_t_b, len_t_f * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * rank + 6, MPI_COMM_WORLD,
                  &reqs[8 * rank + 6]);
        MPI_Irecv(resv_t_f, len_t_f * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * nodenum_t_f + 6,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_t_f + 6]);
    }

    int len_t_b = subgrid[0] * subgrid[1] * subgrid[2];

    double *resv_t_b = new double[len_t_b * 6 * 2];
    double *send_t_f = new double[len_t_b * 6 * 2];
    if (N_sub[3] != 1)
    {

        for (int i = 0; i < len_t_b * 6 * 2; i++)
        {
            send_t_f[i] = 0;
        }

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int z = 0; z < subgrid[2]; z++)
                {
                    complex<double> tmp;

                    int t = subgrid[3] - 1;

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    complex<double> *AO = U.A[3] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                    subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                    x + (1 - cb) * subgrid_vol_cb) *
                                                       9;

                    int b = cont * 6;
                    cont += 1;
                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {

                            tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            send_t_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                            send_t_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                            tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            send_t_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                            send_t_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                        }
                    }
                }
            }
        }

        MPI_Isend(send_t_f, len_t_b * 6 * 2, MPI_DOUBLE, nodenum_t_f, 8 * rank + 7, MPI_COMM_WORLD,
                  &reqs[8 * rank + 7]);
        MPI_Irecv(resv_t_b, len_t_b * 6 * 2, MPI_DOUBLE, nodenum_t_b, 8 * nodenum_t_b + 7,
                  MPI_COMM_WORLD, &reqr[8 * nodenum_t_b + 7]);
    }

    //////////////////////////////////////////////////////// no comunication
    /////////////////////////////////////////////////////////

    const int srcO_scale = subgrid[0] * subgrid[1] * subgrid[2] * 12;
    const int destE_scale = subgrid[0] * subgrid[1] * subgrid[2] * 12;
    const int AO_scale = subgrid[0] * subgrid[1] * subgrid[2] * 9;
    const int AE_scale = subgrid[0] * subgrid[1] * subgrid[2] * 9;

    for (int y = 0; y < subgrid[1]; y++)
    {
        for (int z = 0; z < subgrid[2]; z++)
        {
            for (int t = 0; t < subgrid[3]; t++)
            {
                int x_u =
                    ((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? subgrid[0] : subgrid[0] - 1;

                for (int x = 0; x < x_u; x++)
                {

                    complex<double> *destE;
                    complex<double> *AE;
                    complex<double> tmp;
                    int f_x;
                    if ((y + z + t + x_p) % 2 == cb)
                    {
                        f_x = x;
                    }
                    else
                    {
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

                    __m256d srcStart, srcStart2;
                    __m256d vSrcReal[4], vSrcImag[4], vdestE, vAE;

                    // load srcO
                    __m256i mask = _mm256_set_epi32(0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x80000000, 0x0); //之前反了
                    for (int i = 0; i < 4; i++)
                    {
                        // srcStart = _mm256_loadu_pd(&srcO[3 * i + 0]);
                        srcStart = _mm256_loadu_pd((double *)&srcO[3 * i + 0]);
                        // srcStart2 = _mm256_loadu_pd(&srcO[3 * i + 2]);
                        srcStart2 = _mm256_maskload_pd((double *)&srcO[3 * i + 2], mask);

                        vSrcReal[i] = _mm256_unpacklo_pd(srcStart, srcStart2); // 四个实数，但是只要用3个 [A0r A2r  A1r 0 ]
                        vSrcImag[i] = _mm256_unpackhi_pd(srcStart, srcStart2);
                    }

                    // vtmp = srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]
                    // srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]
                    // flag = (dag == true) ? -1 : 1;
                    __m256d vTmp1Real, vTmp1Imag, vTmp2Real, vTmp2Imag;
                    if (!dag)
                    {
                        vTmp1Real = _mm256_add_pd(vSrcReal[0], vSrcImag[3]); // vSrcReal[0] + vSrcImag[3]
                        vTmp1Imag = _mm256_sub_pd(vSrcImag[0], vSrcReal[3]); // vSrcImag[0] - vSrcReal[3]
                        vTmp2Real = _mm256_add_pd(vSrcReal[1], vSrcImag[2]); // vSrcReal[1] + vSrcImag[2]
                        vTmp2Imag = _mm256_sub_pd(vSrcImag[1], vSrcReal[2]); // vSrcImag[1] - vSrcReal[2]
                    }
                    else
                    {
                        vTmp1Real = _mm256_sub_pd(vSrcReal[0], vSrcImag[3]);
                        vTmp1Imag = _mm256_add_pd(vSrcImag[0], vSrcReal[3]);
                        vTmp2Real = _mm256_sub_pd(vSrcReal[1], vSrcImag[2]);
                        vTmp2Imag = _mm256_add_pd(vSrcImag[1], vSrcReal[2]);
                    }

                    // load AE
                    __m256d vAEReal_c1, vAEImag_c1;
                    __m256d vHalf = _mm256_set1_pd(-0.5);

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        srcStart = _mm256_loadu_pd((double *)&AE[3 * c1 + 0]);
                        srcStart2 = _mm256_maskload_pd((double *)&AE[3 * c1 + 2], mask);
                        vAEReal_c1 = _mm256_unpacklo_pd(srcStart, srcStart2); // 四个实数，但是好像只要用3个
                        vAEImag_c1 = _mm256_unpackhi_pd(srcStart, srcStart2);

                        // 计算第一行的实数 (vTmp1Real * vAEReal - vAEimag * vTmp1imag) / -2
                        __m256d vTmpReal = _mm256_fmsub_pd(vTmp1Real, vAEReal_c1, _mm256_mul_pd(vAEImag_c1, vTmp1Imag));
                        vTmpReal = _mm256_mul_pd(vTmpReal, vHalf);
                        // Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
                        __m256d vTmpSumReal = _mm256_hadd_pd(vTmpReal, vTmpReal);
                        destE[0 * 3 + c1].real(((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]);
                        destE[3 * 3 + c1].imag(flag * (((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]));

                        // 计算第一行的虚部 (vTmp1Real * vAEimag + vAEReal * vTmp1imag) / -2
                        __m256d vTmpImag = _mm256_fmadd_pd(vTmp1Real, vAEImag_c1, _mm256_mul_pd(vAEReal_c1, vTmp1Imag));
                        vTmpImag = _mm256_mul_pd(vTmpImag, vHalf);
                        // Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
                        __m256d vTmpSumImag = _mm256_hadd_pd(vTmpImag, vTmpImag);
                        destE[0 * 3 + c1].imag(((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]);
                        destE[3 * 3 + c1].real(-flag * (((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]));

                        // 计算第二行的实部 (vTmp2Real * vAEReal - vAEimag * vTmp2imag) / -2
                        vTmpReal = _mm256_fmsub_pd(vTmp2Real, vAEReal_c1, _mm256_mul_pd(vAEImag_c1, vTmp2Imag));
                        vTmpReal = _mm256_mul_pd(vTmpReal, vHalf);
                        // Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
                        vTmpSumReal = _mm256_hadd_pd(vTmpReal, vTmpReal);
                        destE[1 * 3 + c1].real(((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]);
                        destE[2 * 3 + c1].imag(flag * (((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]));

                        // 计算第二行的虚部 (vTmp2Real * vAEimag + vAEReal * vTmp2imag) / -2
                        vTmpImag = _mm256_fmadd_pd(vTmp2Real, vAEImag_c1, _mm256_mul_pd(vAEReal_c1, vTmp2Imag));
                        vTmpImag = _mm256_mul_pd(vTmpImag, vHalf);
                        // Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
                        vTmpSumImag = _mm256_hadd_pd(vTmpImag, vTmpImag);
                        destE[1 * 3 + c1].imag(((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]);
                        destE[2 * 3 + c1].real(-flag * (((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]));
                    }
                }
            }
        }
    }

    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            for (int t = 0; t < subgrid[3]; t++) {
                int x_d = (((y + z + t + x_p) % 2) != cb || N_sub[0] == 1) ? 0 : 1;
             
                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AO_vindex = _mm_set_epi32(6 * AO_scale, 4 * AO_scale, 2 * AO_scale, 0 * AO_scale);

                int x = x_d;
                for (; x + 3 < subgrid[0]; x += 4) {
                    int b_x;

                    if ((t + z + y + x_p) % 2 == cb) {
                        b_x = (x - 1 + subgrid[0]) % subgrid[0];
                    } else {
                        b_x = x;
                    }
                    complex<double> *srcO = src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z + subgrid[0] * y + b_x + (1 - cb) * subgrid_vol_cb) * 12;
                    complex<double> *destE = dest.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                    complex<double> *AO = U.A[0] +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z + subgrid[0] * y + b_x + (1 - cb) * subgrid_vol_cb) * 9;

                    /*
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
                    */          

                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAOReal, vAOImag;
                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAOReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]), AO_vindex, gather_scale);
                            vAOImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]) + 1, AO_vindex, gather_scale);
                            vAOReal = _mm256_mul_pd(vAOReal, vNHalf);
                            vAOImag = _mm256_sub_pd(vZero, _mm256_mul_pd(vAOImag, vNHalf));
                            // vAO = conj(-half * AO[c1 * 3 + c2])
                            // don't touch vAO from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmpReal = _mm256_add_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_sub_pd(vtmpImag, vtmp2Real);
                            } else {
                                vtmpReal = _mm256_sub_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_add_pd(vtmpImag, vtmp2Real);
                            }
                            vtmp2Real = vtmpReal, vtmp2Imag = vtmpImag;
                            // tmp2 = srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * conj(-half * AO[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpReal);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpReal);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] += flag * (-I * tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmpReal = _mm256_add_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_sub_pd(vtmpImag, vtmp2Real);
                            } else {
                                vtmpReal = _mm256_sub_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_add_pd(vtmpImag, vtmp2Real);
                            }
                            vtmp2Real = vtmpReal, vtmp2Imag = vtmpImag;
                            // tmp2 = srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * (-half * AO[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[1 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpReal);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpReal);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] += flag * (-I * tmp);
                        }
                    }
                }
                
                if (x == subgrid[0])
                    continue;
                    
                for (; x < subgrid[0]; x++) {
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

    int y_u = (N_sub[1] == 1) ? subgrid[1] : subgrid[1] - 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < y_u; y++) {
            const int f_y = (y + 1) % subgrid[1];
            for (int z = 0; z < subgrid[2]; z++) {
                complex<double> * const srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * f_y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> * const destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> * const AE_base = U.A[1] + (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 9;

                // index multiplied by 2: complex<double> == double[2]
                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AE_vindex = _mm_set_epi32(6 * AE_scale, 4 * AE_scale, 2 * AE_scale, 0 * AE_scale);

                for (int t = 0; t < subgrid[3]; t += 4) {
                    complex<double> * const srcO = srcO_base + t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AE = AE_base + t * AE_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAEReal, vAEImag;
                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAEReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]), AE_vindex, gather_scale);
                            vAEImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]) + 1, AE_vindex, gather_scale);
                            vAEReal = _mm256_mul_pd(vAEReal, vNHalf); // multiply AE by -0.5: saves some instructions later
                            vAEImag = _mm256_mul_pd(vAEImag, vNHalf);
                            // vAE = -half * AE[c1 * 3 + c2]
                            // don't touch vAE from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] += flag * (tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                        }
                    }
                }

                /*
                for (int t = 0; t < subgrid[3]; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    complex<double> *srcO =
                        src.A +
                        (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                         subgrid[0] * f_y + x + (1 - cb) * subgrid_vol_cb) *
                            12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[1] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = -(srcO[0 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
                */
            }
        }
    }

    int y_d = (N_sub[1] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = y_d; y < subgrid[1]; y++) {
            int b_y = (y - 1 + subgrid[1]) % subgrid[1];

            for (int z = 0; z < subgrid[2]; z++) {
                complex<double> * const srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * b_y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> * const destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> * const AO_base = U.A[1] +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * b_y + x + (1 - cb) * subgrid_vol_cb) * 9;

                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AO_vindex = _mm_set_epi32(6 * AO_scale, 4 * AO_scale, 2 * AO_scale, 0 * AO_scale);
                const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                for (int t = 0; t < subgrid[3]; t += 4) {
                    complex<double> * const srcO = srcO_base + t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AO = AO_base + t * AO_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAOReal, vAOImag;

                    /*
                    // PROVEN CORRECT
                    complex<double> tmp;
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] -= flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (tmp);
                        }
                    }
                    */

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAOReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]), AO_vindex, gather_scale);
                            vAOImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]) + 1, AO_vindex, gather_scale);
                            vAOReal = _mm256_mul_pd(vAOReal, vNHalf);
                            vAOImag = _mm256_sub_pd(vZero, _mm256_mul_pd(vAOImag, vNHalf)); // conj(AO)
                            // vAO = conj(-half * AO[c2 * 3 + c1])
                            // don't touch vAO from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] - flag * srcO[3 * 3 + c2]) * conj(-half * AO[c2 * 3 + c1])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] -= flag * (tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            }

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[1 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] += flag * (tmp);
                        }
                    }
                }
            }
        }
    }

    int z_u = (N_sub[2] == 1) ? subgrid[2] : subgrid[2] - 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < z_u; z++) {
                int f_z = (z + 1) % subgrid[2];
                complex<double> *srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * f_z + subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> *destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> *AE_base = U.A[2] +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 9;

                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AE_vindex = _mm_set_epi32(6 * AE_scale, 4 * AE_scale, 2 * AE_scale, 0 * AE_scale);

                for (int t = 0; t < subgrid[3]; t += 4) {
                    /*
                    complex<double> tmp;
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (I * tmp);
                            tmp = -(srcO[1 * 3 + c2] + flag * I * srcO[3 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (-I * tmp);
                        }
                    }
                    */
                    complex<double> * const srcO = srcO_base + t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AE = AE_base + t * AE_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAEReal, vAEImag;
                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAEReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]), AE_vindex, gather_scale);
                            vAEImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]) + 1, AE_vindex, gather_scale);
                            vAEReal = _mm256_mul_pd(vAEReal, vNHalf); // multiply AE by -0.5: saves some instructions later
                            vAEImag = _mm256_mul_pd(vAEImag, vNHalf);
                            // vAE = -half * AE[c1 * 3 + c2]
                            // don't touch vAE from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmpReal = _mm256_sub_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_add_pd(vtmpImag, vtmp2Real);
                            } else {
                                vtmpReal = _mm256_add_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_sub_pd(vtmpImag, vtmp2Real);
                            }
                            vtmp2Real = vtmpReal, vtmp2Imag = vtmpImag;
                            // tmp2 = srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpReal);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpReal);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] += flag * (I * tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmpReal = _mm256_add_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_sub_pd(vtmpImag, vtmp2Real);
                            } else {
                                vtmpReal = _mm256_sub_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_add_pd(vtmpImag, vtmp2Real);
                            }
                            vtmp2Real = vtmpReal, vtmp2Imag = vtmpImag;
                            // tmp2 = srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] - flag * srcO[2 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpReal);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpReal);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    int z_d = (N_sub[2] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = z_d; z < subgrid[2]; z++) {
                int b_z = (z - 1 + subgrid[2]) % subgrid[2];
                complex<double> *srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * b_z + subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> *destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> *AO_base = U.A[2] +
                    (subgrid[0] * subgrid[1] * b_z + subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) * 9;

                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AO_vindex = _mm_set_epi32(6 * AO_scale, 4 * AO_scale, 2 * AO_scale, 0 * AO_scale);

                for (int t = 0; t < subgrid[3]; t += 4) {
                    /*
                    complex<double> tmp;
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (-I * tmp);
                            tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (I * tmp);
                        }
                    }
                    */
                    complex<double> * const srcO = srcO_base + t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AO = AO_base + t * AO_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAOReal, vAOImag;
                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAOReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]), AO_vindex, gather_scale);
                            vAOImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AO[c2 * 3 + c1]) + 1, AO_vindex, gather_scale);
                            vAOReal = _mm256_mul_pd(vAOReal, vNHalf);
                            vAOImag = _mm256_sub_pd(vZero, _mm256_mul_pd(vAOImag, vNHalf));
                            // vAO = conj(-half * AO[c1 * 3 + c2])
                            // don't touch vAO from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmpReal = _mm256_add_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_sub_pd(vtmpImag, vtmp2Real);
                            } else {
                                vtmpReal = _mm256_sub_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_add_pd(vtmpImag, vtmp2Real);
                            }
                            vtmp2Real = vtmpReal, vtmp2Imag = vtmpImag;
                            // tmp2 = srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] + flag * I * srcO[2 * 3 + c2]) * conj(-half * AO[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpReal);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpReal);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] += flag * (-I * tmp);


                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmpReal = _mm256_sub_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_add_pd(vtmpImag, vtmp2Real);
                            } else {
                                vtmpReal = _mm256_add_pd(vtmpReal, vtmp2Imag);
                                vtmpImag = _mm256_sub_pd(vtmpImag, vtmp2Real);
                            }
                            vtmp2Real = vtmpReal, vtmp2Imag = vtmpImag;
                            // tmp2 = srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAOReal), _mm256_mul_pd(vtmp2Imag, vAOImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAOImag), _mm256_mul_pd(vtmp2Imag, vAOReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * (-half * AO[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpReal);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpImag);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpReal);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                        }
                    }
                }
            }
        }
    }

    int t_u = (N_sub[3] == 1) ? subgrid[3] : subgrid[3] - 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                complex<double> *srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> *destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> *AE_base = U.A[2] +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 9;

                // index multiplied by 2: complex<double> == double[2]
                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AE_vindex = _mm_set_epi32(6 * AE_scale, 4 * AE_scale, 2 * AE_scale, 0 * AE_scale);
                
                int t = 0;
                for (; t + 3 < t_u; t += 4) {
                    int f_t = (t + 1) % subgrid[3];
                    complex<double> * const srcO = srcO_base + f_t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AO = AO_base + t * AO_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAEReal, vAEImag;
                    
                    /*
                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] -= flag * (tmp);
                        }
                    }
                    */

                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAEReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]), AE_vindex, gather_scale);
                            vAEImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]) + 1, AE_vindex, gather_scale);
                            vAEReal = _mm256_mul_pd(vAEReal, vNHalf); // multiply AE by -0.5: saves some instructions later
                            vAEImag = _mm256_mul_pd(vAEImag, vNHalf);
                            // vAE = -half * AE[c1 * 3 + c2]
                            // don't touch vAE from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] -= flag * (tmp);

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[1 * 3 + c1] += tmp;
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
                
                for (;t < t_u; t++) {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int f_t = (t + 1) % subgrid[3];

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * f_t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[3] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            tmp = -(srcO[0 * 3 + c2] - flag * srcO[2 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] - flag * srcO[3 * 3 + c2]) * half *
                                  AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
            }
        }
    }

    int t_d = (N_sub[3] == 1) ? 0 : 1;
    for (int x = 0; x < subgrid[0]; x++) {
        for (int y = 0; y < subgrid[1]; y++) {
            for (int z = 0; z < subgrid[2]; z++) {
                complex<double> *srcO_base = src.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) * 12;
                complex<double> *destE_base = dest.A +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 12;
                complex<double> *AE_base = U.A[2] +
                    (subgrid[0] * subgrid[1] * z + subgrid[0] * y + x + cb * subgrid_vol_cb) * 9;

                // index multiplied by 2: complex<double> == double[2]
                const __m128i srcO_vindex = _mm_set_epi32(6 * srcO_scale, 4 * srcO_scale, 2 * srcO_scale, 0 * srcO_scale);
                const __m128i destE_vindex = srcO_vindex;
                const __m128i AE_vindex = _mm_set_epi32(6 * AE_scale, 4 * AE_scale, 2 * AE_scale, 0 * AE_scale);
                
                int t = t_d;
                for (; t + 3 < subgrid[3]; t += 4) {
                    int b_t = (t - 1 + subgrid[3]) % subgrid[3];
                    complex<double> * const srcO = srcO_base + b_t * srcO_scale;
                    complex<double> * const destE = destE_base + t * destE_scale;
                    complex<double> * const AO = AO_base + b_t * AO_scale;
                    __m256d vtmpReal, vtmpImag, vtmp2Real, vtmp2Imag, vAEReal, vAEImag;
                    
                    /*
                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (tmp);
                        }
                    }
                    */

                    const __m256d vNHalf = _mm256_set1_pd(-0.5), vZero = _mm256_set1_pd(0.0);

                    double storeReal[4], storeImag[4];
                    const int gather_scale = 8; // one index = 8 bytes
                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            vAEReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]), AE_vindex, gather_scale);
                            vAEImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&AE[c1 * 3 + c2]) + 1, AE_vindex, gather_scale);
                            vAEReal = _mm256_mul_pd(vAEReal, vNHalf); // multiply AE by -0.5: saves some instructions later
                            vAEImag = _mm256_mul_pd(vAEImag, vNHalf);
                            // vAE = -half * AE[c1 * 3 + c2]
                            // don't touch vAE from now on

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[0 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[2 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            // dag --> flag = -1
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[0 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[0 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[0 * 3 + c1] += tmp;

                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[2 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[3 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[2 * 3 + c1] += flag * (tmp);

                            vtmpReal = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmpImag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[1 * 3 + c2]) + 1, srcO_vindex, gather_scale);
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]), srcO_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&srcO[3 * 3 + c2]) + 1, srcO_vindex, gather_scale);

                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_sub_pd(vtmpImag, vtmp2Imag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmpReal, vtmp2Real);
                                vtmp2Imag = _mm256_add_pd(vtmpImag, vtmp2Imag);
                            }
                            // tmp2 = srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]

                            // (a+bi)(c+di) = (ac-bd) + (ad+bc)i
                            vtmpReal = _mm256_sub_pd(_mm256_mul_pd(vtmp2Real, vAEReal), _mm256_mul_pd(vtmp2Imag, vAEImag));
                            vtmpImag = _mm256_add_pd(_mm256_mul_pd(vtmp2Real, vAEImag), _mm256_mul_pd(vtmp2Imag, vAEReal));
                            // result now in vtmp = (srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * (-half * AE[c1 * 3 + c2])

                            // reuse vtmp2 for destE
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[1 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                            vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[1 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[1 * 3 + c1] += tmp;
                            vtmp2Real = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]), destE_vindex, gather_scale);
                            vtmp2Imag = _mm256_i32gather_pd(reinterpret_cast<double*>(&destE[3 * 3 + c1]) + 1, destE_vindex, gather_scale);
                            if (dag) {
                                vtmp2Real = _mm256_sub_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_sub_pd(vtmp2Imag, vtmpImag);
                            } else {
                                vtmp2Real = _mm256_add_pd(vtmp2Real, vtmpReal);
                                vtmp2Imag = _mm256_add_pd(vtmp2Imag, vtmpImag);
                            }
                            _mm256_storeu_pd(storeReal, vtmp2Real);
                            _mm256_storeu_pd(storeImag, vtmp2Imag);
                            for (int i = 0; i < 4; i++) {
                                destE[2 * 3 + c1 + i * destE_scale] = complex<double>(storeReal[i], storeImag[i]);
                            }
                            // destE[3 * 3 + c1] += flag * (tmp);
                        }
                    }
                }
                
                for (; t < subgrid[3]; t++)
                {

                    complex<double> *destE;
                    complex<double> *AO;

                    int b_t = (t - 1 + subgrid[3]) % subgrid[3];

                    complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * b_t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AO = U.A[3] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * b_t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + (1 - cb) * subgrid_vol_cb) *
                             9;

                    complex<double> tmp;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (tmp);
                            tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                                  conj(AO[c2 * 3 + c1]);
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (tmp);
                        }
                    }
                }
            }
        }
    }

    //    printf(" rank =%i  ghost  \n ", rank);

    //////////////////////////////////////////////////////////////////////////////////////ghost//////////////////////////////////////////////////////////////////

    if (N_sub[0] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_x_f], &star[8 * nodenum_x_f]);

        int cont = 0;
        for (int y = 0; y < subgrid[1]; y++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    if ((y + z + t + x_p) % 2 == cb)
                    {
                        continue;
                    }

                    complex<double> *destE;
                    complex<double> *AE;
                    complex<double> tmp;

                    int x = subgrid[0] - 1;

                    complex<double> *srcO = (complex<double> *)(&resv_x_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
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

        MPI_Wait(&reqs[8 * rank], &stas[8 * rank]);

    } // if(N_sub[0]!=1)

    //    delete[] send_x_b;
    //    delete[] resv_x_f;

    if (N_sub[0] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_x_b + 1], &star[8 * nodenum_x_b + 1]);

        int cont = 0;

        for (int y = 0; y < subgrid[1]; y++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    if (((y + z + t + x_p) % 2) != cb)
                    {
                        continue;
                    }

                    int x = 0;

                    complex<double> *srcO = (complex<double> *)(&resv_x_b[cont * 6 * 2]);
                    cont += 1;

                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (-I * srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (-I * srcO[1 * 3 + c1]);
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 1], &stas[8 * rank + 1]);

    } // if(N_sub[0]!=1)

    //    delete[] send_x_f;
    //    delete[] resv_x_b;

    if (N_sub[1] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_y_f + 2], &star[8 * nodenum_y_f + 2]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int y = subgrid[1] - 1;

                    complex<double> *srcO = (complex<double> *)(&resv_y_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[1] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 2], &stas[8 * rank + 2]);

    } // if(N_sub[1]!=1)

    //    delete[] send_y_b;
    //    delete[] resv_y_f;

    if (N_sub[1] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_y_b + 3], &star[8 * nodenum_y_b + 3]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int z = 0; z < subgrid[2]; z++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    complex<double> *srcO = (complex<double> *)(&resv_y_b[cont * 6 * 2]);

                    cont += 1;

                    int y = 0;
                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[3 * 3 + c1] -= flag * srcO[0 * 3 + c1];
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[2 * 3 + c1] += flag * srcO[1 * 3 + c1];
                    }
                    //                    for (int i = 0; i < 12; i++) {
                    //                        destE[i] += srcO[i];
                    //                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 3], &stas[8 * rank + 3]);
    }

    //    delete[] send_y_f;
    //    delete[] resv_y_b;

    if (N_sub[2] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_z_f + 4], &star[8 * nodenum_z_f + 4]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;

                    int z = subgrid[2] - 1;

                    complex<double> *srcO = (complex<double> *)(&resv_z_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[2] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] += flag * (I * tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] += flag * (-I * tmp);
                        }
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 4], &stas[8 * rank + 4]);

    } // if(N_sub[2]!=1)

    //    delete[] send_z_b;
    //    delete[] resv_z_f;

    if (N_sub[2] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_z_b + 5], &star[8 * nodenum_z_b + 5]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int t = 0; t < subgrid[3]; t++)
                {
                    complex<double> *srcO = (complex<double> *)(&resv_z_b[cont * 6 * 2]);

                    cont += 1;

                    int z = 0;
                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (-I * srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (I * srcO[1 * 3 + c1]);
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 5], &stas[8 * rank + 5]);

    } // if (N_sub[2] != 1)

    // delete[] send_z_f;
    // delete[] resv_z_b;

    if (N_sub[3] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_t_f + 6], &star[8 * nodenum_t_f + 6]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int z = 0; z < subgrid[2]; z++)
                {

                    complex<double> tmp;
                    complex<double> *destE;
                    complex<double> *AE;
                    int t = subgrid[3] - 1;

                    complex<double> *srcO = (complex<double> *)(&resv_t_f[cont * 6 * 2]);

                    cont += 1;

                    destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U.A[3] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        for (int c2 = 0; c2 < 3; c2++)
                        {
                            tmp = srcO[0 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[0 * 3 + c1] += tmp;
                            destE[2 * 3 + c1] -= flag * (tmp);
                            tmp = srcO[1 * 3 + c2] * AE[c1 * 3 + c2];
                            destE[1 * 3 + c1] += tmp;
                            destE[3 * 3 + c1] -= flag * (tmp);
                        }
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 6], &stas[8 * rank + 6]);
    }

    if (N_sub[3] != 1)
    {

        MPI_Wait(&reqr[8 * nodenum_t_b + 7], &star[8 * nodenum_t_b + 7]);

        int cont = 0;
        for (int x = 0; x < subgrid[0]; x++)
        {
            for (int y = 0; y < subgrid[1]; y++)
            {
                for (int z = 0; z < subgrid[2]; z++)
                {
                    complex<double> *srcO = (complex<double> *)(&resv_t_b[cont * 6 * 2]);

                    cont += 1;
                    int t = 0;
                    complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                       subgrid[0] * subgrid[1] * z +
                                                       subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                          12;

                    for (int c1 = 0; c1 < 3; c1++)
                    {
                        destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                        destE[2 * 3 + c1] += flag * (srcO[0 * 3 + c1]);
                        destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                        destE[3 * 3 + c1] += flag * (srcO[1 * 3 + c1]);
                    }
                }
            }
        }

        MPI_Wait(&reqs[8 * rank + 7], &stas[8 * rank + 7]);

    } // if (N_sub[3] != 1)

    MPI_Barrier(MPI_COMM_WORLD);

    delete[] send_x_b;
    delete[] resv_x_f;
    delete[] send_x_f;
    delete[] resv_x_b;

    delete[] send_y_b;
    delete[] resv_y_f;
    delete[] send_y_f;
    delete[] resv_y_b;

    delete[] send_z_b;
    delete[] resv_z_f;
    delete[] send_z_f;
    delete[] resv_z_b;

    delete[] send_t_b;
    delete[] resv_t_f;
    delete[] send_t_f;
    delete[] resv_t_b;
}
