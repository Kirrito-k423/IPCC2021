/**
** @file:  load_gauge.h
** @brief:
**/

#ifndef LOAD_GAUGE_H
#define LOAD_GAUGE_H

#include "lattice_gauge.h"
#include <string>
#include <complex>

int load_gauge(std::complex<double> *gauge[4], std::string &filename, int *subgs, int *site_vec);

#endif