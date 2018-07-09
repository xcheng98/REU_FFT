#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define EPS 0.0000001192f

#include "helper/my_vector.h"

float UPPER_BOUND = 1000.0f;
int SIZE = 10;

__host__ int split_32_to_16(fft::VectorF X, fft::VectorH Xhi, fft::VectorH Xlo, float* s1, float* s2, int N)
{
    float scale1 = 0.0f;

    for (int i = 1; i <= N; i++){
        float norm = (float) fabs(X.element(i));
        if (norm > scale1) scale1 = norm;
    }
    
    // Restrict scale range
    if (scale1 < EPS){
        scale1 = EPS;
    }
    if (scale1 > 1.0f/EPS){
        scale1 = 1.0f/EPS;
    }

    fft::VectorF Xtemp;
    Xtemp.size = N;
    Xtemp.array = (float*)malloc(Xtemp.size * sizeof(float));

    // Get the normalized Xhi
    for (int i = 1; i <= N; i++) {
        Xtemp.element(i) = X.element(i) / scale1;
        Xhi.element(i) = (half)(Xtemp.element(i));
        // Using Xtemp to store the residual
        Xtemp.element(i) = X.element(i) - scale1 * (float)Xhi.element(i);
    }

    // Normalize Xlo
    float scale2 = 0.0f;
    for (int i = 1; i <= N; i++){
        float norm = (float)fabs(Xtemp.element(i));
        if (norm > scale2) scale2 = norm;
    }
    if (scale2 < EPS){
        scale2 = EPS;
    }
    if (scale2 > 1.0f/EPS){
        scale2 = 1.0f/EPS;
    }
    for (int i = 1; i <= N; i++){
        Xtemp.element(i) = Xtemp.element(i) / scale2;
        Xlo.element(i) = (half) (Xtemp.element(i));
    }

    *s1 = scale1;
    *s2 = scale2;
    free(Xtemp.array);    
    return 0;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    fft::VectorF X;
    X.size = SIZE;
    X.array = (float*)malloc(X.size * sizeof(float));

    printf("The input is: \n");
    for (int i = 1; i <= SIZE; i++){
        X.element(i) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
        printf("X[%d] = %.10f\n", i, X.element(i));
    }
    
    fft::VectorH Xhi;
    Xhi.size = SIZE;
    Xhi.array = (half*)malloc(Xhi.size * sizeof(half));

    fft::VectorH Xlo;
    Xlo.size = SIZE;
    Xlo.array = (half*)malloc(Xlo.size * sizeof(half));

    float scale1, scale2;

    split_32_to_16(X, Xhi, Xlo, &scale1, &scale2, SIZE);

    printf("Result: \n S1=%.10f, S2=%.10f, \n", scale1, scale2);
    for (int i = 1; i <= SIZE; i++){
        printf("Xhi[%d] = %.10f, Xlo[%d] = %.10f\n", i, (float)Xhi.element(i), i, (float)Xlo.element(i));
    }
   
    
    free(X.array);    
    free(Xhi.array);    
    free(Xlo.array);    
}
