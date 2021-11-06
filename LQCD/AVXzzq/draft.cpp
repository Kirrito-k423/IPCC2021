#include <stdio.h>
#include <iostream>
// #include <immintrin.h>
#include <complex>
using namespace std;

#define tsj(name) #name, (name)
void Print(char * name,int value){
    printf("%s : %d\n",name,value);
}
// void printAVX256d(char * name, __m256d saveSrc){
//         double saveSrc[4];
// 		_mm256_store_pd(saveSrc,saveSrc);
// 		printf("%s : \n",name);
// 		printf("%.2f %.2f %.2f %.2f \n", saveSrc[0], saveSrc[1], saveSrc[2], saveSrc[3]);
// }

void printArray(char * name,complex<double> * src,int i,int j){
    printf("%s : \n",name);
	for (int c1 = 0; c1 < i; c1++)
	{
		for (int c2 = 0; c2 < j; c2++)
		{
			printf("%.2f+%.2fi ", src[c1 * j + c2].real(), src[c1 * j + c2].imag());
		}
		printf("\n");
	}
}
//https://stackoverflow.com/questions/3386861/converting-a-variable-name-to-a-string-in-c
int main(){
    int subgrid[4] = {1, 1, 1, 4};
    int x_p = 1;
    int cb = 0;
    int N_sub[4] = {1, 1, 1, 4};
    int subgrid_vol_cb=0;
    int flag = 1;
    float half = 0.5;
    const complex<double> I(0, 1);
    complex<double> dest[12*4];
	complex<double> U[9*4] = {
		{0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52},{0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38},{-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71},
        {0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52},{0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38},{-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71},
        {0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52},{0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38},{-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71},
        {0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52},{0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38},{-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71}
        };
		// {0.38, -0.67}, {-0.11, 0.17}, {0.10, -0.59}};
	complex<double> src[12*4] = {
	        {0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21},{0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71},
		{0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79},{0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93},
        	{0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21},{0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71},
		{0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79},{0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93},
        	{0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21},{0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71},
		{0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79},{0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93},
        	{0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21},{0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71},
		{0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79},{0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93}
        };
    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            for (int t = 0; t < subgrid[3]; t++) {
                int x_u =
                    ((y + z + t + x_p) % 2 == cb || N_sub[0] == 1) ? subgrid[0] : subgrid[0] - 1;
                Print(tsj(x_u));
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

                    complex<double> *srcO = src + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                     subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                     f_x + (1 - cb) * subgrid_vol_cb) *
                                                        12;

                    destE = dest + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                      subgrid[0] * subgrid[1] * z + subgrid[0] * y + x +
                                      cb * subgrid_vol_cb) *
                                         12;

                    AE = U[0] +
                         (subgrid[0] * subgrid[1] * subgrid[2] * t + subgrid[0] * subgrid[1] * z +
                          subgrid[0] * y + x + cb * subgrid_vol_cb) *
                             9;

                    for (int c1 = 0; c1 < 3; c1++) {
                        for (int c2 = 0; c2 < 3; c2++) {
                            {
                                tmp = -(srcO[0 * 3 + c2] - flag * I * srcO[3 * 3 + c2]) * half *
                                      AE[c1 * 3 + c2];
                                destE[0 * 3 + c1] += tmp;
                                destE[3 * 3 + c1] += flag * (I * tmp);
                                tmp = -(srcO[1 * 3 + c2] - flag * I * srcO[2 * 3 + c2]) * half *
                                      AE[c1 * 3 + c2];
                                destE[1 * 3 + c1] += tmp;
                                destE[2 * 3 + c1] += flag * (I * tmp);
                            }
                        }
                    }
                }
            }
        }
    }
    printArray(tsj(dest),12,4);
}

