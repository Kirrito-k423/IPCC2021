
#include <iostream>
#include <complex>
using namespace std;
int main(){
for (int x = 0; x < subgrid[0]; x++) {
    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            complex<double> tmp;

            int t = subgrid[3] - 1;

            complex<double> *srcO = src.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                                x + (1 - cb) * subgrid_vol_cb) *
                                                12;

            complex<double> *AO = U.A[3] + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                            subgrid[0] * subgrid[1] * z + subgrid[0] * y +
                                            x + (1 - cb) * subgrid_vol_cb) *
                                                9;

            int b = cont * 6;
            cont += 1;
            for (int c1 = 0; c1 < 3; c1++) {
                for (int c2 = 0; c2 < 3; c2++) {

                    tmp = -(srcO[0 * 3 + c2] + flag * srcO[2 * 3 + c2]) * half *
                            conj(AO[c2 * 3 + c1]);
                    send_t_f[b * 2 + (0 * 3 + c1) * 2 + 0] += tmp.real();
                    send_t_f[b * 2 + (0 * 3 + c1) * 2 + 1] += tmp.imag();
                    tmp = -(srcO[1 * 3 + c2] + flag * srcO[3 * 3 + c2]) * half *
                            conj(AO[c2 * 3 + c1]);
                    send_t_f[b * 2 + (1 * 3 + c1) * 2 + 0] += tmp.real();
                    send_t_f[b * 2 + (1 * 3 + c1) * 2 + 1] += tmp.imag();
                }
            }
        }
    }
}

for (int x = 0; x < subgrid[0]; x++) {
    for (int y = 0; y < subgrid[1]; y++) {
        for (int z = 0; z < subgrid[2]; z++) {
            complex<double> *srcO = (complex<double> *) (&resv_t_b[cont * 6 * 2]);

            cont += 1;
            int t = 0;
            complex<double> *destE = dest.A + (subgrid[0] * subgrid[1] * subgrid[2] * t +
                                                subgrid[0] * subgrid[1] * z +
                                                subgrid[0] * y + x + cb * subgrid_vol_cb) *
                                                    12;

            for (int c1 = 0; c1 < 3; c1++) {
                destE[0 * 3 + c1] += srcO[0 * 3 + c1];
                destE[2 * 3 + c1] += flag * (srcO[0 * 3 + c1]);
                destE[1 * 3 + c1] += srcO[1 * 3 + c1];
                destE[3 * 3 + c1] += flag * (srcO[1 * 3 + c1]);
            }
        }
    }
}
printAVX256d(tsj(yahaha))
}


#define tsj(name) #name, (name)
void printAVX256d(char * name,__m256d saveSrc){
        double saveSrc[4];
		_mm256_store_pd(saveSrc,saveSrc);
		printf("%s : \n",name);
		printf("%.2f %.2f %.2f %.2f \n", saveSrc[0], saveSrc[1], saveSrc[2], saveSrc[3]);
}
//https://stackoverflow.com/questions/3386861/converting-a-variable-name-to-a-string-in-c