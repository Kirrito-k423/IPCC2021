/**
** @file:  check.h
** @brief:
**/

#ifndef _CHECK_H_
#define _CHECK_H_

#include "lattice_fermion.h"
#include "lattice_gauge.h"
#include <complex>

int check(std::complex<double> *src_p, std::complex<double> *chi_p, std::complex<double> *gauge[4],
          const double mass, int *subgs, int *site_vec);

int check(lattice_fermion &src, lattice_fermion &chi, lattice_gauge &U, double mass);

void check_lattice_site(int *subgs, int *site_vec);

void gamma5_lattcefermion(lattice_fermion &src, lattice_fermion &dest);

void check_gauge_unitary(std::complex<double> *U);

void check_plaq(lattice_gauge &U);

#endif