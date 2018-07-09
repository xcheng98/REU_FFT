#include <stdio.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define EPS 0.0000001192f

#include "helper/my_vector.h"
#include "helper/my_matrix.h"

float UPPER_BOUND = 1000.0f;
int SIZE = 10;
int BATCH = 4;

__host__ int split_32_to_16(fft::MatrixF X, fft::MatrixH Xhi, fft::MatrixH Xlo, fft::VectorF s1, fft::VectorF s2, int N, int B)
{
    // Calculate scaling factor 1
    for (int j = 1; j <= B; j++){
        float scale1 = 0.0f;
        for (int i = 1; i <= N; i++){
            float norm = (float) fabs(X.element(i, j));
            if (norm > scale1) scale1 = norm;
        }
        
        // Restrict scale range
        if (scale1 < EPS){
            scale1 = EPS;
        }
        if (scale1 > 1.0f/EPS){
            scale1 = 1.0f/EPS;
        }

        s1.element(j) = scale1;
    }

    // Initialize temporary matrix
    fft::MatrixF Xtemp;
    Xtemp.width = B;
    Xtemp.height = N;
    Xtemp.array = (float*)malloc(Xtemp.width * Xtemp.height * sizeof(float));

    // Get the normalized Xhi
    for (int j = 1; j <= B; j++){ 
        for (int i = 1; i <= N; i++) {
            Xtemp.element(i, j) = X.element(i, j) / s1.element(j);
            Xhi.element(i, j) = (half)(Xtemp.element(i, j));
            // Using Xtemp to store the residual
            Xtemp.element(i, j) = X.element(i, j) - s1.element(j) * (float)Xhi.element(i, j);
        }
    }

    // Calculate Xhi
    for (int j = 1; j <= B; j++){
        float scale2 = 0.0f;
        for (int i = 1; i <= N; i++){
            float norm = (float)fabs(Xtemp.element(i, j));
            if (norm > scale2) scale2 = norm;
        }
        if (scale2 < EPS){
            scale2 = EPS;
        }
        if (scale2 > 1.0f/EPS){
            scale2 = 1.0f/EPS;
        }
        s2.element(j) = scale2;
    }

    // Normalize lower part
    for (int j = 1; j <= B; j++){
        for (int i = 1; i <= N; i++){
            Xtemp.element(i, j) = Xtemp.element(i, j) / s2.element(j);
            Xlo.element(i, j) = (half) (Xtemp.element(i, j));
        }
    }

    free(Xtemp.array);    
    return 0;
}

int main(int argc, char **argv)
{
    srand(time(NULL));
    fft::MatrixF X;
    X.width = BATCH;
    X.height = SIZE;
    X.array = (float*)malloc(X.width * X.height * sizeof(float));

    printf("The input is: \n");
    for (int i = 1; i <= SIZE; i++){
        for (int j = 1; j <= BATCH; j++){
            X.element(i, j) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            printf("X[%d, %d] = %.10f\t", i, j, X.element(i, j));
        }
        printf("\n");
    }
    
    fft::MatrixH Xhi;
    Xhi.width = BATCH;
    Xhi.height = SIZE;
    Xhi.array = (half*)malloc(Xhi.width * Xhi.height * sizeof(half));

    fft::MatrixH Xlo;
    Xlo.width = BATCH;
    Xlo.height = SIZE;
    Xlo.array = (half*)malloc(Xlo.width * Xlo.height * sizeof(half));

    fft::VectorF scale1;
    scale1.size = BATCH;
    scale1.array = (float*)malloc(scale1.size * sizeof(float));

    fft::VectorF scale2;
    scale2.size = BATCH;
    scale2.array = (float*)malloc(scale2.size * sizeof(float));
    
    split_32_to_16(X, Xhi, Xlo, scale1, scale2, SIZE, BATCH);

    printf("Result: \n ");
    for (int i = 1; i <= BATCH; i++){
        printf("S1[%d]=%.10f, S2[%d]=%.10f, \n", i, scale1.element(i), i, scale2.element(i));
    }

    for (int i = 1; i <= SIZE; i++){
        for (int j = 1; j <= BATCH; j++){
            printf("[%d, %d]=%.10f, %.10f\t", i, j, (float)Xhi.element(i, j), (float)Xlo.element(i, j));
        }
        printf("\n");
    }
   
    
    free(X.array);    
    free(Xhi.array);    
    free(Xlo.array); 
    free(scale1.array);
    free(scale2.array);   
}
