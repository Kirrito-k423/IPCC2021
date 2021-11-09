/**
** @file:  lattice_gauge.h
** @brief:
**/

#ifndef LATTICE_LATTICE_GAUGE_H
#define LATTICE_LATTICE_GAUGE_H

#include <complex>

class lattice_gauge {
  public:
    int *site_vec;
    std::complex<double> *A[4];
    int *subgs;
    int size;

    //    lattice_gauge(multi1d <LatticeColorMatrix> &chroma_gauge, int *subgs1, int *site_vec1);
    lattice_gauge(std::complex<double> *chroma_gauge[4], int *subgs1, int *site_vec1);
    lattice_gauge(int *subgs1, int *site_vec1);
    ~lattice_gauge();
    std::complex<double> peeksite(const int *site, int ll = 0, int mm = 0, int dd = 0);

  private:
    bool mem_flag;
};

#endif //LATTICE_LATTICE_GAUGE_H
