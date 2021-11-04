/*
 * @Descripttion: 
 * @version: 
 * @Author: Shaojie Tan
 * @Date: 2021-11-03 18:57:36
 * @LastEditors: Shaojie Tan
 * @LastEditTime: 2021-11-03 20:51:30
 */

 #include <immintrin.h>

 
for (int y = 0; y < subgrid[1]; y++) {
	for (int z = 0; z < subgrid[2]; z++) {
		for (int t = 0; t < subgrid[3]; t++) {
			int x_u =
				((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? subgrid[0] : subgrid[0] - 1;

			for (int x = 0; x < x_u; x++) {

				complex<double> *destE;
				complex<double> *AE;
				complex<double> tmp;
				int f_x;
				if ((y + z + t + x_p) % 2 == cb) {
				f_x = x;
				} else {
				f_x = (x + 1) % subgrid[0];
				}

				complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
								subgrid[0] * subgrid[1] * z + subgrid[0] * y +
								f_x + (1 - cb) * subgrid_vol_cb) *
								12;

				destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
						subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
						cb * subgrid_vol_cb) *
							12;

				AE = U.A[0] +
					(subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
					subgrid[0] * y + x + cb * subgrid_vol_cb) *
					9;

				__m256d	srcStart, srcStart2;
				__m256d  vSrcReal[4], vSrcImag[4], vdestE, vAE;
				
				__m256i mask = _mm256_set_epi32(0x80000000,0x0,0x80000000,0x0,0x80000000,0x0,0x0,0x0);
				for (int i=0; i<4; i++){
					srcStart = _mm256_loadu_pd(&srcO[3*i+0]);
					// srcStart2 = _mm256_loadu_pd(&srcO[3*i+2]);
					srcStart2 = _mm256_maskload_pd(&srcO[3*i+2],mask);
					vSrcReal[i] = _mm256_unpacklo_pd(srcStart,srcStart2); // 四个实数，但是好像只要用3个
					vSrcImag[i] = _mm256_unpackhi_pd(srcStart,srcStart2); 
				}
				
				// vtmp = srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]
				//  srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]
				if(flag){
					vTmp1Real = _mm256_add_pd(vSrcReal[0], vSrcImag[3]); //vSrcReal[0] + vSrcImag[3] 
					vTmp1Imag = _mm256_sub_pd(vSrcImag[0], vSrcReal[3]);
					vTmp2Real = _mm256_add_pd(vSrcReal[1], vSrcImag[2]); //vSrcReal[0] + vSrcImag[3] 
					vTmp2Imag = _mm256_sub_pd(vSrcImag[1], vSrcReal[2]);
				}else{
					vTmp1Real = _mm256_sub_pd(vSrcReal[0], vSrcImag[3]); 
					vTmp1Imag = _mm256_add_pd(vSrcImag[0], vSrcReal[3]);
					vTmp2Real = _mm256_sub_pd(vSrcReal[1], vSrcImag[2]); //vSrcReal[0] + vSrcImag[3] 
					vTmp2Imag = _mm256_add_pd(vSrcImag[1], vSrcReal[2]);
				}
				
				__m256d vAEReal[3], vAEImage[3];
				for (int c1 = 0; c1 < 3; c1++) {
					srcStart = _mm256_loadu_pd(&AE[3 * c1 + 0]);
					srcStart2 = _mm256_maskload_pd(&AE[3 * c1 + 2], mask);
					vAEReal[i] = _mm256_unpacklo_pd(srcStart, srcStart2); // 四个实数，但是好像只要用3个
					vAEImag[i] = _mm256_unpackhi_pd(srcStart, srcStart2); 					
				}

				__m256d vHalf = _mm256_set1_pd(-0.5);
				for (int c1 = 0; c1 < 3; c1++) {
					// 另一种思路 step1Real = _mm256_set_epi64x(vTmp1Real[c2*64:(c2+1)*64],vTmp1Real[c2*64:(c2+1)*64],vTmp1Real[c2*64:(c2+1)*64],0);


					// 计算第一行的实数 (vTmp1Real * vAEReal - vAEImage * vTmp1Image) / -2
					__m256d vTmpC = _mm256_mul_pd(vAEImag, vTmp1Imag);
					__m256d vTmp3Real = _mm256_fmsub_pd(vTmp1Real, vAEReal, vTmpC);
					vTmp3Real = _mm256_mul_pd(vTmp3Real, vHalf);			// (vTmp1Real * AEReal - vAEimage * tmp1Image) / 2
					__m256d vTmpSumReal = _mm256_hadd_pd(vTmp3Real, vTmp3Real);		// vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
					destE[0 * 3 + c1].real = ((double*)&vTmpSumReal)[0] + ((double*)&vTmpSumReal)[2];
					destE[3 * 3 + c1].image = flag * (((double*)&vTmpSumReal)[0] + ((double*)&vTmpSumReal)[2]);

					// 计算第一行的虚部 (vTmp1Real * vAEImage + vAEReal * vTmp1Image) / -2
					vTmpC = _mm256_mul_pd(vAEReal, vTmp1Imag);
					__m256d vTmp3Imag = _mm256_fmsub_pd(vTmp1Real, vAEImag, vTmpC);
					vTmp3Imag = _mm256_mul_pd(vTmp3Imag, vHalf);			// (vTmp1Real * AEReal - vAEimage * tmp1Image) / 2
					__m256d vTmpSumImag = _mm256_hadd_pd(vTmp3Imag, vTmp3Imag);		// vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
					destE[0 * 3 + c1].image = ((double*)&vTmpSumImag)[0] + ((double*)&vTmpSumImag)[2];
					destE[3 * 3 + c1].real = - flag * (((double*)&vTmpSumImag)[0] + ((double*)&vTmpSumImag)[2]);
					 
					// (vTmp2Real * vAEReal - vAEImage * vTmp2Image) / -2
					__m256d vTmpC = _mm256_mul_pd(vAEImag, vTmp2Imag);
					__m256d vTmp3Real = _mm256_fmsub_pd(vTmp2Real, vAEReal, vTmpC);
					vTmp3Real = _mm256_mul_pd(vTmp3Real, vHalf);			// (vTmp1Real * AEReal - vAEimage * tmp1Image) / 2
					__m256d vTmpSumReal = _mm256_hadd_pd(vTmp3Real, vTmp3Real);		// vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
					destE[1 * 3 + c1].real = ((double*)&vTmpSumReal)[0] + ((double*)&vTmpSumReal)[2];
					destE[2 * 3 + c1].image = flag * (((double*)&vTmpSumReal)[0] + ((double*)&vTmpSumReal)[2]);

					// (vTmp2Real * vAEImage + vAEReal * vTmp2Image) / -2
					vTmpC = _mm256_mul_pd(vAEReal, vTmp2Imag);
					__m256d vTmp3Imag = _mm256_fmsub_pd(vTmp2Real, vAEImag, vTmpC);
					vTmp3Imag = _mm256_mul_pd(vTmp3Imag, vHalf);			// (vTmp1Real * AEReal - vAEimage * tmp1Image) / 2
					__m256d vTmpSumImag = _mm256_hadd_pd(vTmp3Imag, vTmp3Imag);		// vtmp3[2] + vtmp3[3], vtmp3[0] + vtmp3[1]
					destE[1 * 3 + c1].image = ((double*)&vTmpSumImag)[0] + ((double*)&vTmpSumImag)[2];
					destE[2 * 3 + c1].real = - flag * (((double*)&vTmpSumImag)[0] + ((double*)&vTmpSumImag)[2]);
		
				}
				

				// for (int c1 = 0; c1 < 3; c1++) {
				// 	for (int c2 = 0; c2 < 3; c2++) {
				// 		{
				// 			tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
				// 				AE[c1 * 3 + c2];
				// 			destE[0 * 3 + c1] += tmp;
				// 			destE[3 * 3 + c1] += flag * (I * tmp);
				// 			tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
				// 				AE[c1 * 3 + c2];
				// 			destE[1 * 3 + c1] += tmp;
				// 			destE[2 * 3 + c1] += flag * (I * tmp);
				// 		}
				// 	}
				// }
			}
		}
	}
}