/*
 * Implementing fft4 algorithm
 * Input is one float32 vector
 * No spliting
 * It's not a complete FFT
 * To be used recursively by gfft
 */

// C includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas.h>

// Matrix and vector
#include <helper/my_vector.h>
#include <helper/my_matrix.h>
#include <helper/my_const.h>


fft::MatrixF F4_re;
fft::MatrixF F4_im;

FFT_S init_F4()
{
    F4_re.width = 4;
    F4_re.height = 4;
    F4_re.array = (float*)malloc(F4_re.width * F4_re.height * sizeof(float));

    F4_re.element(1, 1) = 1.0f;
    F4_re.element(2, 1) = 1.0f;
    F4_re.element(3, 1) = 1.0f;
    F4_re.element(4, 1) = 1.0f;
    F4_re.element(1, 2) = 1.0f;
    F4_re.element(2, 2) = 0.0f;
    F4_re.element(3, 2) =-1.0f;
    F4_re.element(4, 2) = 0.0f;
    F4_re.element(1, 3) = 1.0f;
    F4_re.element(2, 3) =-1.0f;
    F4_re.element(3, 3) = 1.0f;
    F4_re.element(4, 3) =-1.0f;
    F4_re.element(1, 4) = 1.0f;
    F4_re.element(2, 4) = 0.0f;
    F4_re.element(3, 4) =-1.0f;
    F4_re.element(4, 4) = 0.0f;
    
    F4_im.width = 4;
    F4_im.height = 4;
    F4_im.array = (float*)malloc(F4_re.width * F4_re.height * sizeof(float));

    F4_im.element(1, 1) = 0.0f;
    F4_im.element(2, 1) = 0.0f;
    F4_im.element(3, 1) = 0.0f;
    F4_im.element(4, 1) = 0.0f;
    F4_im.element(1, 2) = 0.0f;
    F4_im.element(2, 2) =-1.0f;
    F4_im.element(3, 2) = 0.0f;
    F4_im.element(4, 2) = 1.0f;
    F4_im.element(1, 3) = 0.0f;
    F4_im.element(2, 3) = 0.0f;
    F4_im.element(3, 3) = 0.0f;
    F4_im.element(4, 3) = 0.0f;
    F4_im.element(1, 4) = 0.0f;
    F4_im.element(2, 4) = 1.0f;
    F4_im.element(3, 4) = 0.0f;
    F4_im.element(4, 4) =-1.0f;

    return FFT_SUCCESS;
}

int fft4(fft::VectorF X_re, fft::VectorF X_im, fft::VectorF FX_re, fft::VectorF FX_im) 
{
    // Initialize and plan cublas
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;
    stat = cublasCreate(&handle);




}

int main()
{
    FFT_S status;
    status = init_f4();
    if (status != FFT_SUCCESS){
        printf("Error in Fourier matrix initialization\n");
        exit(1);
    }

    
    return 0;
}
