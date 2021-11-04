#include <iostream>
#include <complex>
using namespace std;

int main()
{
    const complex<double> I(0, 1);
	complex<double> destE[12];
	complex<double> AE[12] = {
		{0.40, 0.16}, {0.23, -0.66}, {0.23, 0.52},
		{0.75, -0.27}, {0.42, 0.23}, {0.02, -0.38},
		{-0.22, 0.37}, {0.24, -0.48}, {0.14, -0.71},
		{0.38, -0.67}, {-0.11, 0.17}, {0.10, -0.59}};
	complex<double> srcO[12] = {
		{0.68, 0.27}, {0.44, 0.76}, {0.27, 0.21},
		{0.47, 0.61}, {0.67, 0.91}, {0.72, 0.71},
		{0.28, 0.84}, {0.78, 0.93}, {0.56, 0.79},
		{0.98, 0.38}, {0.78, 0.64}, {0.70, 0.93}};
    complex<double> tmp;
	double flag = -1;
    const double half = 0.5;

    for (int c1 = 0; c1 < 3; c1++)
    {
        for (int c2 = 0; c2 < 3; c2++)
        {
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
	for (int c1 = 0; c1 < 4; c1++)
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
    return 0;
}