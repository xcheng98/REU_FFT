/*
 * Implementing fft4 algorithm
 * It's not a complete FFT
 * To be used by other program
 */

// C includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublasXt.h>

// Matrix and vector
#include <helper/my_vector.h>
#include <helper/my_matrix.h>

int fft4()
{
    fft::MatrixF F4_re;
    F4_re.width = 4;
    F4_re.height = 4;
    F4_re.array = (float*)malloc(F4_re.width * F4_re.height * sizeof(float));

    F4_re(1, 1) = 1.0f;
}

int main()
{
      

}
/*
 * Define matrix:
 ** Matrix A;
 ** A.width = w;
 ** A.height = h;
 ** A.elements = (float*)malloc(A.width * A.height * sizeof(float));
 * Access elements:
 ** A.elements[getIndex(i, j, A.width, A.height)] = x;
 */

int getIndex(int i, int j, int width, int height){
    return i-1 + (j-1) * height;
}    

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


