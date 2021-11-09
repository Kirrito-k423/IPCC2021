#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <set>
#include <vector>
#include "su3.h"
#include "layout.h"
#include "field_rotate.h"
#include "comm/comm_low.h"
#include "comm/comm_intermediate.h"
#include "io.h"
#include "timer.h"
#include "../io/communicator.h"
#include <sys/stat.h>

namespace qcd {

void read_nersc_lattice(const char *filename, su3_field &lnk, int offset, bool order, bool type,
                        const parallel_io &io, bool dble = true)
{
    FILE *fileOutput = io.fopen(filename, "r");
    if (io.is_io_node())
        fseek(fileOutput, offset, SEEK_CUR);

    timer all(false);
    all.start("read field");
    int nCol = (type == false) ? 2 : 3;
    static lattice_desc *currentdesc = lnk.desc;

    size_t unit = (dble == true) ? sizeof(double) : sizeof(float);
    static generic_communicator<> com(io, *lnk.desc, 24 * nCol * unit,
                                      generic_communicator<>::read);
    /*  if(currentdesc != lnk.desc || com.io.num_io_nodes != io.num_io_nodes)
  {
    currentdesc = lnk.desc;
    com.reset(io, *lnk.desc);
  }*/
    com.execute(fileOutput);
    if (order == true)
        switchend((char *) com.comm_buffer, 24 * nCol * lnk.desc->sites_on_node);

    //  double* buf = (double*) com.comm_buffer;
    double_complex *buf = (double_complex *) com.comm_buffer;
    complex<float> *buf_f = (complex<float> *) com.comm_buffer;

#pragma omp parallel for shared(lnk, buf)
    for (local_idx_t i = 0; i < lnk.desc->sites_on_node; ++i)
        for (int iDir = 0; iDir < 4; iDir++) {
            for (int iColorCol = 0; iColorCol < nCol; iColorCol++)
                for (int iColorRow = 0; iColorRow < 3; iColorRow++) {
                    int icount = iColorRow + 3 * (iColorCol + nCol * (iDir + 4 * i));
                    if (dble == true)
                        lnk.data[iDir * lnk.desc->sites_on_node + i].e[iColorCol][iColorRow] =
                            buf[icount];
                    else
                        lnk.data[iDir * lnk.desc->sites_on_node + i].e[iColorCol][iColorRow] =
                            buf_f[icount];
                }
            if (nCol == 2) {
                su3_matrix &tmp = lnk.data[iDir * lnk.desc->sites_on_node + i];
                lnk.data[iDir * lnk.desc->sites_on_node + i].e[2][0] =
                    conj(tmp.e[0][1] * tmp.e[1][2] - tmp.e[0][2] * tmp.e[1][1]);
                lnk.data[iDir * lnk.desc->sites_on_node + i].e[2][1] =
                    conj(tmp.e[0][2] * tmp.e[1][0] - tmp.e[0][0] * tmp.e[1][2]);
                lnk.data[iDir * lnk.desc->sites_on_node + i].e[2][2] =
                    conj(tmp.e[0][0] * tmp.e[1][1] - tmp.e[0][1] * tmp.e[1][0]);
            }
        }

    all.stop();
    if (io.this_node == 0)
        printf("READ_TIMING: total %.2e s io %.2e s  BW: disk %.3f GB/s eff %.3f GB/s\n",
               all.get_time(), com.io_time,
               lnk.desc->volume * 4 * nCol / 3 * sizeof(su3_matrix) / com.io_time / 1e9,
               lnk.desc->volume * 4 * nCol / 3 * sizeof(su3_matrix) / all.get_time() / 1e9);

    io.fclose(fileOutput);
}

double plaquette(su3_field &links);
void save_nersc_lattice(const char *filename, su3_field &lnk, const parallel_io &io, bool order,
                        bool type)
{

    lattice_desc &desc = lnk.desc[0];
    double sum = 0.0;
    for (int isp = 0; isp < desc.sites_on_node * 4; isp++)
        sum += lnk.data[isp].e[0][0].real + lnk.data[isp].e[1][1].real + lnk.data[isp].e[2][2].real;
    global_sum(sum);
    sum /= (desc.volume * 12.0);
    double plq = plaquette(lnk);
    if (io.io_rank == 0)
        printf("link_trace=%20.10f,plq=%20.10f\n", sum, plq);

    if (io.io_rank == 0) {
        FILE *f = fopen(filename, "w");
        fprintf(f, "BEGIN_HEADER\n");
        fprintf(f, "HDR_VERSION = 1.0\n");
        fprintf(f, "DATATYPE = 4D_SU3_GAUGE\n");
        fprintf(f, "STORAGE_FORMAT = 1.0\n");
        fprintf(f, "DIMENSION_1 = %d\n", lnk.desc[0][0]);
        fprintf(f, "DIMENSION_2 = %d\n", lnk.desc[0][1]);
        fprintf(f, "DIMENSION_3 = %d\n", lnk.desc[0][2]);
        fprintf(f, "DIMENSION_4 = %d\n", lnk.desc[0][3]);

        fprintf(f, "LINK_TRACE = %15.13f\n", sum);
        fprintf(f, "PLAQUETTE = %15.13f\n", plq);
        fprintf(f, "BOUNDARY_1 = PERIODIC\n");
        fprintf(f, "BOUNDARY_2 = PERIODIC\n");
        fprintf(f, "BOUNDARY_3 = PERIODIC\n");
        fprintf(f, "BOUNDARY_4 = ANTIPERIODIC\n");
        fprintf(f, "CHECKSUM = 452c11d8\n");
        fprintf(f, "ENSEMBLE_ID = RBC\n");
        fprintf(f, "ENSEMBLE_LABEL = fake\n");
        fprintf(f, "SEQUENCE_NUMBER = 0\n");
        fprintf(f, "CREATOR = RBC\n");
        fprintf(f, "CREATOR_HARDWARE = CU-NOARCH nan\n");
        fprintf(f, "CREATION_DATE = Sun Wed 1 00:00:00 2020\n");
        fprintf(f, "ARCHIVE_DATE_DATE = Sun Wed 1 00:00:00 2020\n");
        if (order == true)
            fprintf(f, "FLOATING_POINT = IEEE64BIG\n");
        else
            fprintf(f, "FLOATING_POINT = IEEE64LITTLE\n");
        fprintf(f, "END_HEADER\n");
        fclose(f);
    }
    synchronize();
    FILE *fileOutput = io.fopen(filename, "r+b");
    if (io.is_io_node())
        fseek(fileOutput, 0, SEEK_END);

    timer all(false);
    all.start("read field");
    int nCol = (type == false) ? 2 : 3;
    static lattice_desc *currentdesc = lnk.desc;
    static generic_communicator<> com(io, *lnk.desc, 24 * nCol * sizeof(double),
                                      generic_communicator<>::write);
    /*  if(currentdesc != lnk.desc || com.io.num_io_nodes != io.num_io_nodes)
  {
    currentdesc = lnk.desc;
    com.reset(io, *lnk.desc);
  }*/
    //  double* buf = (double*) com.comm_buffer;
    double_complex *buf = (double_complex *) com.comm_buffer;

#pragma omp parallel for shared(lnk, buf)
    for (local_idx_t i = 0; i < lnk.desc->sites_on_node; ++i)
        for (int iDir = 0; iDir < 4; iDir++) {
            for (int iColorCol = 0; iColorCol < nCol; iColorCol++)
                for (int iColorRow = 0; iColorRow < 3; iColorRow++) {
                    int icount = iColorRow + 3 * (iColorCol + nCol * (iDir + 4 * i));
                    buf[icount] =
                        lnk.data[iDir * lnk.desc->sites_on_node + i].e[iColorCol][iColorRow];
                }
        }

    if (order == true)
        switchend((char *) com.comm_buffer, 24 * nCol * lnk.desc->sites_on_node);
    com.execute(fileOutput);

    all.stop();
    if (io.this_node == 0)
        printf("READ_TIMING: total %.2e s io %.2e s  BW: disk %.3f GB/s eff %.3f GB/s\n",
               all.get_time(), com.io_time,
               lnk.desc->volume * 4 * nCol / 3 * sizeof(su3_matrix) / com.io_time / 1e9,
               lnk.desc->volume * 4 * nCol / 3 * sizeof(su3_matrix) / all.get_time() / 1e9);

    io.fclose(fileOutput);
}

void read_twqcd_lattice(const char *filename, su3_field &lnk, const parallel_io &io)
{
    read_nersc_lattice(filename, lnk, 4, false, false, io);
#pragma omp parallel for
    for (local_idx_t id = 0; id < 4 * lnk.desc->sites_on_node; ++id)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < i; j++) {
                double_complex tmp = lnk.data[id].e[i][j];
                lnk.data[id].e[i][j] = lnk.data[id].e[j][i];
                lnk.data[id].e[j][i] = tmp;
            }
}

void read_nersc_lattice(const char *filename, su3_field &lnk, const parallel_io &io, bool order)
{
    struct stat statbuff;
    stat(filename, &statbuff);
    long size = 9 * 4 * 2 * 8;
    lattice_desc &desc = lnk.desc[0];
    unsigned long IntendedSize = desc[0] * desc[1] * desc[2] * desc[3] * size;
    int offset = -1;
    bool type = true;
    if (statbuff.st_size > IntendedSize)
        offset = statbuff.st_size - IntendedSize;
    else if (statbuff.st_size > IntendedSize / 3 * 2) {
        offset = statbuff.st_size - IntendedSize / 3 * 2;
        type = false;
    }
    if (get_node_rank() == 0)
        printf("%20d%20d%20d\n", offset, IntendedSize, statbuff.st_size);
    if (offset == -1) {
        if (get_node_rank() == 0)
            printf("the data size is incorrect\n");
        return;
    }
    read_nersc_lattice(filename, lnk, offset, order, type, io);
}

void read_milc_lattice(const char *filename, su3_field &lnk, const parallel_io &io, bool order)
{
    struct stat statbuff;
    stat(filename, &statbuff);
    long size = 9 * 4 * 2 * 4;
    lattice_desc &desc = lnk.desc[0];
    unsigned long IntendedSize = desc[0] * desc[1] * desc[2] * desc[3] * size;
    int offset = -1;
    if (statbuff.st_size > IntendedSize)
        offset = statbuff.st_size - IntendedSize;
    if (get_node_rank() == 0)
        printf("%20d%20d%20d\n", offset, IntendedSize, statbuff.st_size);
    if (offset == -1) {
        if (get_node_rank() == 0)
            printf("the data size is incorrect\n");
        return;
    }
    read_nersc_lattice(filename, lnk, offset, order, true, io, false);
}

void read_ildg_lattice(const char *filename, su3_field &lnk, const parallel_io &io, int off)
{

    struct stat statbuff;
    stat(filename, &statbuff);
    bool double_flag = false;
    long size = 9 * 4 * 2 * 4;
    long all_size = size;
    lattice_desc &desc = lnk.desc[0];
    for (int i = 0; i < 4; i++)
        all_size *= desc[i];
    long offset = statbuff.st_size - all_size;

    if (offset > statbuff.st_size / 2) {
        if (io.this_node == 0)
            printf("The configuration is in double precision\n");
        size *= 2;
        all_size *= 2;
        offset = statbuff.st_size - all_size;
        double_flag = true;
    }

    if (io.this_node == 0)
        printf("offset=%15ld,%15ld\n", offset, statbuff.st_size);

    FILE *fileOutput = io.fopen(filename, "r");
    if (io.is_io_node())
        fseek(fileOutput, offset - off, SEEK_CUR);

    timer all(false);
    all.start("read field");
    fflush(stdout);
    static lattice_desc *currentdesc = lnk.desc;
    static generic_communicator<> com(io, *lnk.desc, size, generic_communicator<>::read);
    com.execute(fileOutput);

    switchend((char *) com.comm_buffer, 72 * lnk.desc->sites_on_node,
              (double_flag == true) ? 8 : 4);

    //  double* buf = (double*) com.comm_buffer;
    float_complex *buf = (float_complex *) com.comm_buffer;

#pragma omp parallel for shared(lnk, buf)
    for (local_idx_t i = 0; i < lnk.desc->sites_on_node; ++i)
        for (int iDir = 0; iDir < 4; iDir++) {
            for (int iColorCol = 0; iColorCol < 3; iColorCol++)
                for (int iColorRow = 0; iColorRow < 3; iColorRow++) {
                    int icount = iColorRow + 3 * (iColorCol + 3 * (iDir + 4 * i));
                    //      lnk.data[iDir*lnk.desc->sites_on_node+i].e[iColorCol][iColorRow]=buf[icount];
                    if (double_flag == true)
                        lnk.data[iDir * lnk.desc->sites_on_node + i].e[iColorCol][iColorRow] =
                            ((double_complex *) com.comm_buffer)[icount];
                    else
                        lnk.data[iDir * lnk.desc->sites_on_node + i].e[iColorCol][iColorRow] =
                            ((float_complex *) com.comm_buffer)[icount];
                }
        }

    all.stop();
    if (io.this_node == 0)
        printf("READ_TIMING: total %.2e s io %.2e s  BW: disk %.3f GB/s eff %.3f GB/s\n",
               all.get_time(), com.io_time,
               lnk.desc->volume * 2 * sizeof(su3_matrix) / com.io_time / 1e9,
               lnk.desc->volume * 2 * sizeof(su3_matrix) / all.get_time() / 1e9);
    fflush(stdout);

    io.fclose(fileOutput);
}

void read_jlqcd_lattice(const char *filename, su3_field &lnk, const parallel_io &io)
{
    struct stat statbuff;
    stat(filename, &statbuff);
    long size = 9 * 8 * 2;
    long all_size = size * 4;
    lattice_desc &desc = lnk.desc[0];
    for (int i = 0; i < 4; i++)
        all_size *= desc[i];
    long offset = statbuff.st_size - all_size;
    double io_time = 0.0;

    if (io.this_node == 0)
        printf("offset=%15ld,%15ld\n", offset, statbuff.st_size);

    FILE *fileOutput = io.fopen(filename, "r");
    if (io.is_io_node())
        fseek(fileOutput, offset - 4, SEEK_CUR);

    timer all(false);
    all.start("read field");
    fflush(stdout);
    static lattice_desc *currentdesc = lnk.desc;
    static generic_communicator<> com(io, *lnk.desc, size, generic_communicator<>::read);

    double_complex *buf = (double_complex *) com.comm_buffer;

    for (int iDir = 0; iDir < 4; iDir++) {
        com.execute(fileOutput);

        switchend((char *) com.comm_buffer, 18 * lnk.desc->sites_on_node, 8);

#pragma omp parallel for shared(lnk, buf)
        for (local_idx_t i = 0; i < lnk.desc->sites_on_node; ++i) {
            for (int iColorCol = 0; iColorCol < 3; iColorCol++)
                for (int iColorRow = 0; iColorRow < 3; iColorRow++) {
                    int icount = iColorRow + 3 * (iColorCol + 3 * i);
                    lnk.data[iDir * lnk.desc->sites_on_node + i].e[iColorCol][iColorRow] =
                        buf[icount];
                }
        }
        io_time += com.io_time;
    }

    all.stop();
    if (io.this_node == 0)
        printf("READ_TIMING: total %.2e s io %.2e s  BW: disk %.3f GB/s eff %.3f GB/s\n",
               all.get_time(), com.io_time,
               lnk.desc->volume * 2 * sizeof(su3_matrix) / com.io_time / 1e9,
               lnk.desc->volume * 2 * sizeof(su3_matrix) / all.get_time() / 1e9);
    fflush(stdout);

    io.fclose(fileOutput);
}

//end qcd
}
