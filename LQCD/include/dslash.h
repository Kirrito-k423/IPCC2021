/**
** @file:  dslash.h
** @brief: define Dslash and Dlash-dagger operaters, 
           i.e.  dest = M(U) src &  dest = M(U)^\dagger src. 
           Note: even-odd precondition trick to be used.
**/

#ifndef LATTICE_DSLASH_H
#define LATTICE_DSLASH_H

#include "lattice_fermion.h"
#include "lattice_gauge.h"
#include "operator.h"
#include "operator_mpi.h"

void DslashEE(lattice_fermion &src, lattice_fermion &dest, const double mass);
void DslashOO(lattice_fermion &src, lattice_fermion &dest, const double mass);
void DslashEO(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag);
void DslashOE(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag);
void Dslashoffd(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag,
                int cb);
void Dslashoffd(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag,
                int cb,
                int * N_sub,int rank, int size, 
                const int nodenum_x_b, const int nodenum_x_f, 
                const int nodenum_y_b, const int nodenum_y_f, 
                const int nodenum_z_b, const int nodenum_z_f, 
                const int nodenum_t_b, const int nodenum_t_f, 
                int subgrid_vol, const int x_p
                );
#endif
