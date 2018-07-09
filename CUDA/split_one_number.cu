#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define EPS 0.0000001192f

float UPPER_BOUND = 1000.0f;

int main(int argc, char **argv)
{
    srand(time(NULL));
    float X = (float)rand() / (float)(RAND_MAX) * UPPER_BOUND;
    
    printf("The input is %.10f\n", X);

    float scale1 = (float)fabs(X);
    // Restrict scale range
    if (scale1 < EPS){
        scale1 = EPS;
    }
    if (scale1 > 1.0f/EPS){
        scale1 = 1.0f/EPS;
    }

    // Get the normalized Xhi
    float Xtemp = X / scale1;
    half Xhi = (half) Xtemp;

    // Using Xtemp to store the residual
    Xtemp = X - scale1 * (float)Xhi;

    // Normalize Xlo
    float scale2 = (float)fabs(Xtemp);
    if (scale2 < EPS){
        scale2 = EPS;
    }
    if (scale2 > 1.0f/EPS){
        scale2 = 1.0f/EPS;
    }
    Xtemp = Xtemp / scale2;
    half Xlo = (half) Xtemp;

    printf("S1=%.10f, High=%.10f,\n"
           "S2=%.10f, Low=%.10f\n",
           scale1, (float)Xhi, scale2, (float)Xlo);
}
