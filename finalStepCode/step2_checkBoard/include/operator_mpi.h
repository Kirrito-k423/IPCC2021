/**
** @file:  operator_mpi.h
** @brief:
**/

#ifndef LATTICE_OPERATOR_MPI_H
#define LATTICE_OPERATOR_MPI_H

#include <mpi.h>
#include <complex>

template <typename T>
double norm_2(const T &s)
{
    //    int rank;
    //    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //if (rank==0){
    std::complex<double> s1(0.0, 0.0);
    for (int i = 0; i < s.size; i++) {
        s1 += s.A[i] * conj(s.A[i]);
    }
    double sum_n = s1.real();
    double sum;
    MPI_Reduce(&sum_n, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    return sum;

    //    return s1.real();
    //    }else
    //    {return 0;}
}

template <typename T>
std::complex<double> vector_p(const T &r1, const T &r2)
{
    std::complex<double> s1(0.0, 0.0);
    for (int i = 0; i < r1.size; i++) {
        s1 += (conj(r1.A[i]) * r2.A[i]);
    }
    double sum_r = s1.real(); //fix
    double sum_i = s1.imag();
    double sumr;
    double sumi;
    MPI_Reduce(&sum_r, &sumr, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sum_i, &sumi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sumr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sumi, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    std::complex<double> sum(sumr, sumi);
    //sum.real()= sumr;
    //sum.imag()= sumi;
    return sum;
};

#endif //LATTICE_OPERATOR_MPI_H
