#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define EPS 0.0000001192f

float UPPER_BOUND = 1000.0f;
int SIZE = 10;

__host__ int split_32_to_16(float* X, half* Xhi, half* Xlo, float* s1, float* s2, int N)
{
    float scale1 = 0.0f;

    for (int i = 0; i < N; i++){
        float norm = (float) fabs(X[i]);
        if (norm > scale1) scale1 = norm;
    }
    
    // Restrict scale range
    if (scale1 < EPS){
        scale1 = EPS;
    }
    if (scale1 > 1.0f/EPS){
        scale1 = 1.0f/EPS;
    }

    float Xtemp[N];

    // Get the normalized Xhi
    for (int i = 0; i < N; i++) {
        Xtemp[i] = X[i] / scale1;
        Xhi[i] = (half)(Xtemp[i]);
        // Using Xtemp to store the residual
        Xtemp[i] = X[i] - scale1 * (float)Xhi[i];
    }

    // Normalize Xlo
    float scale2 = 0.0f;
    for (int i = 0; i < N; i++){
        float norm = (float)fabs(Xtemp[i]);
        if (norm > scale2) scale2 = norm;
    }
    if (scale2 < EPS){
        scale2 = EPS;
    }
    if (scale2 > 1.0f/EPS){
        scale2 = 1.0f/EPS;
    }
    for (int i = 0; i < N; i++){
        Xtemp[i] = Xtemp[i] / scale2;
        Xlo[i] = (half) (Xtemp[i]);
    }

    *s1 = scale1;
    *s2 = scale2;
    
    return 0;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    float X[SIZE];
    printf("The input is: \n");
    for (int i = 0; i < SIZE; i++){
        X[i] = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
        printf("X[%d] = %.10f\n", i, X[i]);
    }
    
    half Xhi[SIZE], Xlo[SIZE];
    float scale1, scale2;

    split_32_to_16(X, Xhi, Xlo, &scale1, &scale2, SIZE);

    printf("Result: \n S1=%.10f, S2=%.10f, \n", scale1, scale2);
    for (int i = 0; i < SIZE; i++){
        printf("Xhi[%d] = %.10f, Xlo[%d] = %.10f\n", i, (float)Xhi[i], i, (float)Xlo[i]);
    }

}
