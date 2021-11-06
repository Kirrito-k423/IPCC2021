#include <stdio.h>
#include <iostream>
#include <immintrin.h>
#include <complex>
using namespace std;

int main()
{
	const complex<double> I(0, 1);
	complex<double> destE[12];
	complex<double> AO[9] = {
		{0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52}, {0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38}, {-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71}};
	// {0.38, -0.67}, {-0.11, 0.17}, {0.10, -0.59}};
	complex<double> srcO[12] = {
		{0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21}, {0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71}, {0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79}, {0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93}};
	bool dag = true;
	double flag = (dag == true) ? -1 : 1;

	__m256d srcStart, srcStart2;
	__m256d vSrcReal[4], vSrcImag[4], vdestE, vAO;
	double tmpReal[4], tmpImag[4];

	__m256i mask = _mm256_set_epi32(0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x80000000, 0x0); //之前反了
	for (int i = 0; i < 4; i++)
	{
		// srcStart = _mm256_loadu_pd(&srcO[3 * i + 0]);
		srcStart = _mm256_loadu_pd(((double *)&srcO) + 3 * 2 * i + 0);
		// srcStart2 = _mm256_loadu_pd(&srcO[3*i+2]);
		srcStart2 = _mm256_maskload_pd(((double *)&srcO) + 3 * 2 * i + 2 * 2, mask); // 后面要遮住两个，

		vSrcReal[i] = _mm256_unpacklo_pd(srcStart, srcStart2); // 四个实数，但是只要用3个 [A0r A2r  A1r 0 ]
		vSrcImag[i] = _mm256_unpackhi_pd(srcStart, srcStart2);
	}

	// vtmp = srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]
	//  srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]
	__m256d vTmp1Real, vTmp1Imag, vTmp2Real, vTmp2Imag;
	if (dag != true)
	{
		vTmp1Real = _mm256_sub_pd(vSrcReal[0], vSrcImag[3]);
		vTmp1Imag = _mm256_add_pd(vSrcImag[0], vSrcReal[3]);
		vTmp2Real = _mm256_sub_pd(vSrcReal[1], vSrcImag[2]); //vSrcReal[0] + vSrcImag[3]
		vTmp2Imag = _mm256_add_pd(vSrcImag[1], vSrcReal[2]);
	}
	else
	{
		vTmp1Real = _mm256_add_pd(vSrcReal[0], vSrcImag[3]); //vSrcReal[0] + vSrcImag[3]
		vTmp1Imag = _mm256_sub_pd(vSrcImag[0], vSrcReal[3]);
		vTmp2Real = _mm256_add_pd(vSrcReal[1], vSrcImag[2]); //vSrcReal[0] + vSrcImag[3]
		vTmp2Imag = _mm256_sub_pd(vSrcImag[1], vSrcReal[2]);
	}

	// load AO
	__m256d vAOReal_c2, vAOImag_c2;
	__m256d vHalf = _mm256_set1_pd(-0.5);

	for (int c1 = 0; c1 < 3; c1++)
	{
		vAOReal_c2 = _mm256_set_pd(0, AO[1 * 3 + c1].real(), AO[2 * 3 + c1].real(), AO[0 * 3 + c1].real());
		vAOImag_c2 = _mm256_set_pd(0, AO[1 * 3 + c1].imag(), AO[2 * 3 + c1].imag(), AO[0 * 3 + c1].imag());

		// 计算第一行的实数 (vTmp1Real * vAOReal + vAOimag * vTmp1imag) / -2
		__m256d vTmpReal = _mm256_fmadd_pd(vTmp1Real, vAOReal_c2, _mm256_mul_pd(vAOImag_c2, vTmp1Imag));
		vTmpReal = _mm256_mul_pd(vTmpReal, vHalf);
		// Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		__m256d vTmpSumReal = _mm256_hadd_pd(vTmpReal, vTmpReal);
		destE[0 * 3 + c1].real(((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]);
		destE[3 * 3 + c1].imag(-flag * (((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]));

		// 计算第一行的虚部 (vTmp1Real * vAOimag - vAOReal * vTmp1imag) / -2
		__m256d vTmpImag = _mm256_fmsub_pd(vTmp1Real, vAOImag_c2, _mm256_mul_pd(vAOReal_c2, vTmp1Imag));
		vTmpImag = _mm256_mul_pd(vTmpImag, vHalf);
		// Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		__m256d vTmpSumImag = _mm256_hadd_pd(vTmpImag, vTmpImag);
		destE[0 * 3 + c1].imag(((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]);
		destE[3 * 3 + c1].real(flag * (((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]));

		// 计算第二行的实部 (vTmp2Real * vAOReal + vAOimag * vTmp2imag) / -2
		vTmpReal = _mm256_fmadd_pd(vTmp2Real, vAOReal_c2, _mm256_mul_pd(vAOImag_c2, vTmp2Imag));
		vTmpReal = _mm256_mul_pd(vTmpReal, vHalf);
		// Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		vTmpSumReal = _mm256_hadd_pd(vTmpReal, vTmpReal);
		destE[1 * 3 + c1].real(((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]);
		destE[2 * 3 + c1].imag(-flag * (((double *)&vTmpSumReal)[0] + ((double *)&vTmpSumReal)[2]));

		// 计算第二行的虚部 (vTmp2Real * vAOimag - vAOReal * vTmp2imag) / -2
		vTmpImag = _mm256_fmsub_pd(vTmp2Real, vAOImag_c2, _mm256_mul_pd(vAOReal_c2, vTmp2Imag));
		vTmpImag = _mm256_mul_pd(vTmpImag, vHalf);
		// Compute vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
		vTmpSumImag = _mm256_hadd_pd(vTmpImag, vTmpImag);
		destE[1 * 3 + c1].imag(((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]);
		destE[2 * 3 + c1].real(flag * (((double *)&vTmpSumImag)[0] + ((double *)&vTmpSumImag)[2]));
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

	printf("AO:\n");
	for (int c1 = 0; c1 < 3; c1++)
	{
		for (int c2 = 0; c2 < 3; c2++)
		{
			printf("%.2f+%.2fi ", AO[c1 * 3 + c2].real(), AO[c1 * 3 + c2].imag());
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