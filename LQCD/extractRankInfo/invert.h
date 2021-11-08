/**
** @file:   invert.h
** @brief:
**/

#ifndef LATTICE_invert_H
#define LATTICE_invert_H

#include <complex>
#include "lattice_fermion.h"
#include "lattice_gauge.h"
#include "dslash.h"

int CGinvert(std::complex<double> *src_p, std::complex<double> *dest_p,
             std::complex<double> *gauge[4], const double mass, const int max,
             const double accuracy, int *sugs, int *site_vec);

int CGinvert(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy, double * tsjtime, int roundNum);

void Dslash(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger);
void Dslash(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger,
            int * N_sub,int rank, int size, 
            int * site_x_f, int * site_x_b, const int nodenum_x_b, const int nodenum_x_f, 
            int * site_y_f, int * site_y_b, const int nodenum_y_b, const int nodenum_y_f, 
            int * site_z_f, int * site_z_b, const int nodenum_z_b, const int nodenum_z_f, 
            int * site_t_f, int * site_t_b, const int nodenum_t_b, const int nodenum_t_f, 
            int subgrid_vol, const int x_p
                );
#endif //LATTICE_invert_H
