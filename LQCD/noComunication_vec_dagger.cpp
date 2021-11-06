#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <complex>
using namespace std;

int main()
{
	const complex<double> I(0, 1);
	complex<double> destE[12];
	complex<double> AE[9] = {
		{0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52},
		{0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38},
		{-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71}};
		// {0.38, -0.67}, {-0.11, 0.17}, {0.10, -0.59}};
	complex<double> srcO[12] = {
		{0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21},
		{0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71},
		{0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79},
		{0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93}};
	bool dag = true;
	double flag = (dag == true) ? -1 : 1;


	// complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
	// 								 subgrid[0] * subgrid[1] * z + subgrid[0] * y +
	// 								 f_x + (1 - cb) * subgrid_vol_cb) *
	// 									12;

	// destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
	// 				  subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
	// 				  cb * subgrid_vol_cb) *
	// 					 12;

	// AE = U.A[0] +
	// 	 (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
	// 	  subgrid[0] * y + x + cb * subgrid_vol_cb) *
	// 		 9;

	__m256d srcStart, srcStart2;
	__m256d vSrcReal[4], vSrcImag[4], vdestE, vAE;
	double tmpReal[4],tmpImag[4];

	__m256i mask = _mm256_set_epi32(0x0, 0x0, 0x0, 0x0,0x80000000, 0x0, 0x80000000, 0x0); //之前反了
	for (int i = 0; i < 4; i++)
	{
		// srcStart = _mm256_loadu_pd(&srcO[3 * i + 0]);
		srcStart = _mm256_loadu_pd(((double *)&srcO) + 3 * 2 * i + 0);
		// srcStart2 = _mm256_loadu_pd(&srcO[3*i+2]);
		srcStart2 = _mm256_maskload_pd(((double *)&srcO)+3 * 2 * i + 2 * 2, mask); // 后面要遮住两个，
		vSrcReal[i] = _mm256_unpacklo_pd(srcStart, srcStart2); // 四个实数，但是只要用3个 [A0r A2r  A1r 0 ]
		vSrcImag[i] = _mm256_unpackhi_pd(srcStart, srcStart2);
		double saveSrc[4],saveSrc2[4];
		_mm256_store_pd(saveSrc,srcStart);
		_mm256_store_pd(saveSrc2,srcStart2);
		printf("srcStart: \n");
		printf("%.2f %.2f %.2f %.2f \n", saveSrc[0], saveSrc[1], saveSrc[2], saveSrc[3]);
		printf("%.2f %.2f %.2f %.2f \n", saveSrc2[0], saveSrc2[1], saveSrc2[2], saveSrc2[3]);
		
	}

	printf("complex %.2f %.2f %.2f %.2f \n", ((double *)&srcO)[0], ((double *)&srcO)[1],((double *)&srcO)[2], ((double *)&srcO)[3]);
	printf("complex %.2f %.2f %.2f %.2f \n", ((double *)&srcO)[4], ((double *)&srcO)[5],((double *)&srcO)[6], ((double *)&srcO)[7]);
	
	for(int j=0; j< 4; j++){
		_mm256_store_pd(tmpReal,vSrcReal[j]);
		_mm256_store_pd(tmpImag,vSrcImag[j]);
		printf("src%d: 最后一个不用\n",j);
		for (int i=0; i<4; i++){
			printf("%.2f+%.2fi \n", tmpReal[i], tmpImag[i]);
		}
	}
	// vtmp = srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]
	//  srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]
	__m256d vTmp1Real, vTmp1Imag, vTmp2Real, vTmp2Imag;
	if (dag != true)
	{
		vTmp1Real = _mm256_add_pd(vSrcReal[0], vSrcImag[3]); //vSrcReal[0] + vSrcImag[3]
		vTmp1Imag = _mm256_sub_pd(vSrcImag[0], vSrcReal[3]);
		vTmp2Real = _mm256_add_pd(vSrcReal[1], vSrcImag[2]); //vSrcReal[0] + vSrcImag[3]
		vTmp2Imag = _mm256_sub_pd(vSrcImag[1], vSrcReal[2]);
	}
	else
	{
		vTmp1Real = _mm256_sub_pd(vSrcReal[0], vSrcImag[3]);
		vTmp1Imag = _mm256_add_pd(vSrcImag[0], vSrcReal[3]);
		vTmp2Real = _mm256_sub_pd(vSrcReal[1], vSrcImag[2]); //vSrcReal[0] + vSrcImag[3]
		vTmp2Imag = _mm256_add_pd(vSrcImag[1], vSrcReal[2]);
	}

	
	_mm256_store_pd(tmpReal,vTmp1Real);
	_mm256_store_pd(tmpImag,vTmp1Imag);
	printf("coreTmp1: 最后一个不用\n");
	for (int i=0; i<4; i++){
		printf("%.2f+%.2fi \n", tmpReal[i], tmpImag[i]);
	}

	_mm256_store_pd(tmpReal,vTmp2Real);
	_mm256_store_pd(tmpImag,vTmp2Imag);
	printf("coreTmp2: 最后一个不用\n");
	for (int i=0; i<4; i++){
		printf("%.2f+%.2fi \n", tmpReal[i], tmpImag[i]);
	}

	// load AE
	__m256d vAEReal[3], vAEImag[3];
	for (int c1 = 0; c1 < 3; c1++)
	{
		srcStart = _mm256_loadu_pd((double *)&AE[3 * c1 + 0]);
		srcStart2 = _mm256_maskload_pd((double *)&AE[3 * c1 + 2], mask);
		vAEReal[c1] = _mm256_unpacklo_pd(srcStart, srcStart2); // 四个实数，但是好像只要用3个
		vAEImag[c1] = _mm256_unpackhi_pd(srcStart, srcStart2);
		double saveSrc[4],saveSrc2[4];
		_mm256_store_pd(saveSrc,srcStart);
		_mm256_store_pd(saveSrc2,srcStart2);
		printf("srcStart2: \n");
		printf("%.2f %.2f %.2f %.2f \n", saveSrc[0], saveSrc[1], saveSrc[2], saveSrc[3]);
		printf("%.2f %.2f %.2f %.2f \n", saveSrc2[0], saveSrc2[1], saveSrc2[2], saveSrc2[3]);
	}

	__m256d vHalf = _mm256_set1_pd(-0.5);
	for (int c1 = 0; c1 < 3; c1++)
	{
		// 另一种思路 step1Real = _mm256_set_epi64x(vTmp1Real[c2*64:(c2+1)*64],vTmp1Real[c2*64:(c2+1)*64],vTmp1Real[c2*64:(c2+1)*64],0);

		// 计算第一行的实数 (vTmp1Real * vAEReal - vAEimag * vTmp1imag) / -2
		__m256d vTmpC = _mm256_mul_pd(vAEImag[c1], vTmp1Imag);
		__m256d vTmp3Real = _mm256_fmsub_pd(vTmp1Real, vAEReal[c1], vTmpC);
		vTmp3Real = _mm256_mul_pd(vTmp3Real, vHalf);				// (vTmp1Real * AEReal - vAEimag * tmp1imag) / 2
		__m256d vTmpSumReal = _mm256_hadd_pd(vTmp3Real, vTmp3Real); // vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		destE[0 * 3 + c1].real(((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]);
		destE[3 * 3 + c1].imag(flag * (((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]));

		// 计算第一行的虚部 (vTmp1Real * vAEimag + vAEReal * vTmp1imag) / -2
		vTmpC = _mm256_mul_pd(vAEReal[c1], vTmp1Imag);
		__m256d vTmp3Imag = _mm256_fmadd_pd(vTmp1Real, vAEImag[c1], vTmpC);
		vTmp3Imag = _mm256_mul_pd(vTmp3Imag, vHalf);				// (vTmp1Real * AEReal - vAEimag * tmp1imag) / 2
		__m256d vTmpSumImag = _mm256_hadd_pd(vTmp3Imag, vTmp3Imag); // vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		destE[0 * 3 + c1].imag(((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]);
		destE[3 * 3 + c1].real(-flag * (((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]));

		// (vTmp2Real * vAEReal - vAEimag * vTmp2imag) / -2
		vTmpC = _mm256_mul_pd(vAEImag[c1], vTmp2Imag);
		vTmp3Real = _mm256_fmsub_pd(vTmp2Real, vAEReal[c1], vTmpC);
		vTmp3Real = _mm256_mul_pd(vTmp3Real, vHalf);		// (vTmp1Real * AEReal - vAEimag * tmp1imag) / 2
		vTmpSumReal = _mm256_hadd_pd(vTmp3Real, vTmp3Real); // vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		destE[1 * 3 + c1].real(((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]);
		destE[2 * 3 + c1].imag(flag * (((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]));

		// (vTmp2Real * vAEimag + vAEReal * vTmp2imag) / -2
		vTmpC = _mm256_mul_pd(vAEReal[c1], vTmp2Imag);
		vTmp3Imag = _mm256_fmadd_pd(vTmp2Real, vAEImag[c1], vTmpC);
		vTmp3Imag = _mm256_mul_pd(vTmp3Imag, vHalf);		// (vTmp1Real * AEReal - vAEimag * tmp1imag) / 2
		vTmpSumImag = _mm256_hadd_pd(vTmp3Imag, vTmp3Imag); // vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		destE[1 * 3 + c1].imag(((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]);
		destE[2 * 3 + c1].real(-flag * (((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]));
	}

	printf("srcO:\n");
	for (int c1 = 0; c1 < 4; c1++)
	{
		for (int c2 = 0; c2 < 3; c2++)
		{
			printf("%.2f+%.2fi ", srcO[c1 * 3 + c2].real(), srcO[c1 * 3 + c2].imag());
		}
		printf("\n");
	}

	printf("AE:\n");
	for (int c1 = 0; c1 < 3; c1++)
	{
		for (int c2 = 0; c2 < 3; c2++)
		{
			printf("%.2f+%.2fi ", AE[c1 * 3 + c2].real(), AE[c1 * 3 + c2].imag());
		}
		printf("\n");
	}

	printf("destE:\n");
	for (int c1 = 0; c1 < 4; c1++)
	{
		for (int c2 = 0; c2 < 3; c2++)
		{
			printf("%.2f+%.2fi ", destE[c1 * 3 + c2].real(), destE[c1 * 3 + c2].imag());
		}
		printf("\n");
	}
}