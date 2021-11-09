/**
** @file:  lattice_gauge.cpp
** @brief:
**/

#include <mpi.h>
#include "lattice_gauge.h"
#include "operator.h"
using namespace std;

/*
lattice_gauge::lattice_gauge(multi1d <LatticeColorMatrix> &chroma_gauge, int *subgs1, int *site_vec1) {
    A[0] = (complex<double> *) &(chroma_gauge[0].elem(0).elem());
    A[1] = (complex<double> *) &(chroma_gauge[1].elem(0).elem());
    A[2] = (complex<double> *) &(chroma_gauge[2].elem(0).elem());
    A[3] = (complex<double> *) &(chroma_gauge[3].elem(0).elem());
    subgs = subgs1;
    site_vec = site_vec1;
}
*/

lattice_gauge::lattice_gauge(complex<double> *chroma_gauge[4], int *subgs1, int *site_vec1)
{
    A[0] = chroma_gauge[0];
    A[1] = chroma_gauge[1];
    A[2] = chroma_gauge[2];
    A[3] = chroma_gauge[3];
    subgs = subgs1;
    site_vec = site_vec1;
    size = subgs[0] * subgs[1] * subgs[2] * subgs[3] * 3 * 3;
    mem_flag = false;
}

lattice_gauge::lattice_gauge(int *subgs1, int *site_vec1)
{
    subgs = subgs1;
    site_vec = site_vec1;
    size = subgs[0] * subgs[1] * subgs[2] * subgs[3] * 3 * 3;
    for (int i = 0; i < 4; i++) {
        A[i] = new complex<double>[size]();
    }
    mem_flag = true;
}

lattice_gauge::~lattice_gauge()
{
    if (mem_flag)
        for (int i = 0; i < 4; i++)
            delete[] A[i];
}

// ll=color_row; // mm=color_rank // dd=dir
complex<double> lattice_gauge::peeksite(const int *site, int ll, int mm, int dd)
{

    int coords[4] = {site[0] / subgs[0], site[1] / subgs[1], site[2] / subgs[2],
                     site[3] / subgs[3]};
    int N_sub[4] = {site_vec[0] / subgs[0], site_vec[1] / subgs[1], site_vec[2] / subgs[2],
                    site_vec[3] / subgs[3]};

    //    int nodenum = (coords[3] * N_sub[2] * N_sub[1] * N_sub[0] +
    //                   coords[2] * N_sub[1] * N_sub[0] +
    //                   coords[1] * N_sub[0] + coords[0]);

    //    int rank = QMP_get_node_number(); //当前节点编号
    int rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //    int nodenum = QMP_get_node_number_from(coords); //当前坐标节点编号
    int nodenum = get_nodenum(coords, N_sub, 4);

    double dest[2];

    if (rank == nodenum) {
        int coord[4];
        for (int i = 0; i < 4; i++) {
            coord[i] = site[i];
        }
        int subgrid_vol_cb = (subgs[0] * subgs[1] * subgs[2] * subgs[3]) >> 1;
        int subgrid_cb_nrow[4];
        for (int i = 0; i < 4; i++) {
            subgrid_cb_nrow[i] = subgs[i];
        }

        subgrid_cb_nrow[0] >>= 1;

        int cb = 0;
        for (int m = 0; m < 4; ++m)
            cb += coord[m];
        cb &= 1;

        int subgrid_cb_coord[4];
        subgrid_cb_coord[0] = (coord[0] >> 1) % subgrid_cb_nrow[0];
        for (int i = 1; i < 4; ++i)
            subgrid_cb_coord[i] = coord[i] % subgrid_cb_nrow[i];
        int t = local_site2(subgrid_cb_coord, subgrid_cb_nrow) + cb * subgrid_vol_cb;
        dest[0] = A[dd][t * 9 + ll * 3 + mm].real();
        dest[1] = A[dd][t * 9 + ll * 3 + mm].imag();
    }

    if (nodenum == 0) {
        MPI_Bcast(dest, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        complex<double> dest2(dest[0], dest[1]);
        printf("finished a sendToWait_node=0 %i \n", rank);
        return dest2;
    } else {
        if (rank == nodenum) {
            MPI_Send(dest, 2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        MPI_Recv(dest, 2, MPI_DOUBLE, nodenum, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Bcast(dest, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    printf("finished a sendToWait node=rank  %i \n", rank);

    complex<double> dest2(dest[0], dest[1]);
    return dest2;
}
