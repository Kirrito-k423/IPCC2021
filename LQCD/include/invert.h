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
             const int max, const double accuracy);

int CGinvert2(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy);

void Dslash(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger);

void Dslash_tilde(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger);

void LInv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass);

void UInv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass);

#endif //LATTICE_invert_H
