/**
** @file:  lattice_fermion.cpp
** @brief:
**/

#include "lattice_fermion.h"
#include <iostream>
using namespace std;

//lattice_fermion::lattice_fermion(LatticeFermion &chroma_fermi) {
//    A = (complex<double> *) &(chroma_fermi.elem(0).elem(0).elem(0));
//}

lattice_fermion::lattice_fermion(complex<double> *chroma_fermi, int *subgs1, int *site_vec1)
{
    A = chroma_fermi;
    subgs = subgs1;
    site_vec = site_vec1;
    size = subgs[0] * subgs[1] * subgs[2] * subgs[3] * 3 * 4;
    mem_flag = false;
}

lattice_fermion::lattice_fermion(int *subgs1, int *site_vec1)
{
    subgs = subgs1;
    site_vec = site_vec1;
    size = subgs[0] * subgs[1] * subgs[2] * subgs[3] * 3 * 4;
    A = new std::complex<double>[size]();
    mem_flag = true;
}

lattice_fermion::~lattice_fermion()
{
    if (mem_flag)
        delete[] A;
    A = NULL;
    subgs = NULL;
    site_vec = NULL;
}

void lattice_fermion::clean()
{
    for (int i = 0; i < size; i++) {
        A[i] = 0;
    }
}

lattice_fermion &lattice_fermion::operator-(const lattice_fermion &a)
{
    for (int i = 0; i < size; i++) {
        this->A[i] = this->A[i] - a.A[i];
    }
    return *this;
}

lattice_fermion &lattice_fermion::operator+(const lattice_fermion &a)
{
    for (int i = 0; i < size; i++) {
        this->A[i] = this->A[i] + a.A[i];
    }
    return *this;
}

/*
complex<double> lattice_fermion::peeksite(vector<int> site,
                                          vector<int> site_vec,
                                          int ii,               //ii=spin
                                          int ll) {             //ll=color

    int length = site_vec[0] * site_vec[1] * site_vec[2] * site_vec[3];
    int vol_cb;
    if (length % 2 == 0) {
        vol_cb = (length) / 2;
    }
    if (length % 2 == 1) {
        vol_cb = (length - 1) / 2;
    }

    int nx = site_vec[0];
    int ny = site_vec[1];
    int nz = site_vec[2];
    int nt = site_vec[3];

    int x = site[0];
    int y = site[1];
    int z = site[2];
    int t = site[3];

    int order = 0;
    int cb = x + y + z + t;

    //判断nx的奇偶性
    if (site_vec[0] % 2 == 0) {
        order = t * nz * ny * nx / 2 + z * ny * nx / 2 + y * nx / 2;
    }
    if (site_vec[0] % 2 == 1) {
        order = t * nz * ny * (nx - 1) / 2 + z * ny * (nx - 1) / 2 + y * (nx - 1) / 2;
    }
    //判断x奇偶性
    if (x % 2 == 0) {
        x = (x / 2);
    }
    if (x % 2 == 1) {
        x = (x - 1) / 2;
    }

    order += x;
    //判断x+y+z+t的奇偶性
    cb &= 1;
    printf("vol_cb=%i\n", vol_cb);
    return A[(order + cb * vol_cb) * 12 + ii * 3 + ll];
}
*/

void Minus(lattice_fermion &src1, lattice_fermion &src2, lattice_fermion &a)
{
    a.clean();
    for (int i = 0; i < src1.size; i++) {
        a.A[i] = src1.A[i] - src2.A[i];
    }
}

void Plus(lattice_fermion &src1, lattice_fermion &src2, lattice_fermion &a)
{
    a.clean();
    for (int i = 0; i < src1.size; i++) {
        a.A[i] = src1.A[i] + src2.A[i];
    }
}
