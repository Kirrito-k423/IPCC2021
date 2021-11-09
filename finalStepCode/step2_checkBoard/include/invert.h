/*
 * @Descripttion: 
 * @version: 
 * @Author: Shaojie Tan
 * @Date: 2021-10-25 22:48:50
 * @LastEditors: Shaojie Tan
 * @LastEditTime: 2021-11-09 09:44:19
 */
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
//add CB
int CGinvert_noDestClean(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy);
int CGinvert2(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
             const int max, const double accuracy);

void Dslash_tilde(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger);

void DslashEEInv(lattice_fermion &src, lattice_fermion &dest, const double mass);

void UpLowChange(lattice_fermion &src, lattice_fermion &dest);

void LInv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass);
void LEE(lattice_fermion &src, lattice_fermion &dest);

void LOO(lattice_fermion &src, lattice_fermion &dest);
void L_OE_EE_Inv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass);
void UInv(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass);
void UEE(lattice_fermion &src, lattice_fermion &dest);
void UOO(lattice_fermion &src, lattice_fermion &dest);
void U_EE_Inv_EO(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass);


void Dslash(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const double mass,
            const bool dagger);

#endif //LATTICE_invert_H
