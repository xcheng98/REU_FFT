/*
 * Implementing fft4 algorithm
 * Input is multiple fp32 vector, number given by B
 * First split every input vector to two fp16 vectors
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
#include <cublas_v2.h>

// Matrix and vector
#include "helper/my_vector.h"
#include "helper/my_matrix.h"
#include "helper/my_const.h"
#include "helper/fp32_to_fp16.h"

float UPPER_BOUND = 1000.0f;
int BATCH = 4;


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

FFT_S fft4(int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im) 
{
    cublasStatus_t status;
    cublasHandle_t handle;
    float *dev_FM, *dev_input, *dev_result1, *dev_result2;
    float alpha = 1.0f, beta = 0.0f;

    // Initialize cublas and allocate device memory
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return FFT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_FM), 4 * 4 * sizeof(dev_FM[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate Fourier Matrix)\n");
        return FFT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_input), 4 * B * 2 * sizeof(dev_input[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate Input Matrix)\n");
        return FFT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_result1), 4 * B * 2 * sizeof(dev_result1[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate result 1 Matrix)\n");
        return FFT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_result2), 4 * B * 2 * sizeof(dev_result2[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate result 2 Matrix)\n");
        return FFT_FAILURE;
    }

    // F4_re * (X_re, X_im)
    //// Copy host data to device
    status = cublasSetVector(4 * 4, sizeof(F4_re.array[0]), F4_re.array, 1, dev_FM, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write F4_re)\n");
        return FFT_FAILURE;
    }
    status = cublasSetVector(4 * B, sizeof(X_re.array[0]), X_re.array, 1, dev_input, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write X_re)\n");
        return FFT_FAILURE;
    }
    status = cublasSetVector(4 * B, sizeof(X_im.array[0]), X_im.array, 1, dev_input + 4 * B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write X_im)\n");
        return FFT_FAILURE;
    }
    //// Call cublas gemm
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 2, 4, &alpha, dev_FM,
                       4, dev_input, 4, &beta, dev_result1, 4);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (a * (c, d)).\n");
        return FFT_FAILURE;
    }

    // F4_im * (X_re, X_im)
    //// Copy host data to device
    status = cublasSetVector(4 * 4, sizeof(F4_im.array[0]), F4_im.array, 1, dev_FM, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write F4_im)\n");
        return FFT_FAILURE;
    }
    //// Call cublas gemm
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 2, 4, &alpha, dev_FM,
                       4, dev_input, 4, &beta, dev_result2, 4);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (b * (c, d)).\n");
        return FFT_FAILURE;
    }

    // Combine and get result, store in result1
    alpha = -1.0f;
    status = cublasSaxpy(handle, 4 * B, &alpha, dev_result2 + 4 * B, 1, dev_result1, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (ac - bd).\n");
        return FFT_FAILURE;
    }
    alpha = 1.0f;
    status = cublasSaxpy(handle, 4 * B, &alpha, dev_result2, 1, dev_result1 + 4 * B, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (ad + bc).\n");
        return FFT_FAILURE;
    }

    // Copy device memory to host
    status = cublasGetVector(4 * B, sizeof(FX_re.array[0]), dev_result1, 1, FX_re.array, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read FX_re)\n");
        return FFT_FAILURE;
    }
    status = cublasGetVector(4 * B, sizeof(FX_im.array[0]), dev_result1 + 4 * B, 1, FX_im.array, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read FX_im)\n");
        return FFT_FAILURE;
    }

    // Deallocate device memory and shutdown
    if (cudaFree(dev_FM) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free Fourier Matrix)\n");
        return FFT_FAILURE;
    }
    if (cudaFree(dev_input) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free Input Matrix)\n");
        return FFT_FAILURE;
    }
    if (cudaFree(dev_result1) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free result 1 Matrix)\n");
        return FFT_FAILURE;
    }
    if (cudaFree(dev_result2) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free result 2 Matrix)\n");
        return FFT_FAILURE;
    }
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

int main()
{
    FFT_S status;
    status = init_F4();
    if (status != FFT_SUCCESS){
        printf("Error in Fourier matrix initialization\n");
        exit(1);
    }

    fft::MatrixF X_re;
    X_re.height = 4;
    X_re.width = BATCH;
    X_re.array = (float*)malloc(X_re.height * X_re.width * sizeof(float));

    fft::MatrixF X_im;
    X_im.height = 4;
    X_im.width = BATCH;
    X_im.array = (float*)malloc(X_im.height * X_im.width * sizeof(float));

    fft::MatrixF FX_re;
    FX_re.height = 4;
    FX_re.width = BATCH;
    FX_re.array = (float*)malloc(FX_re.height * FX_re.width * sizeof(float));

    fft::MatrixF FX_im;
    FX_im.height = 4;
    FX_im.width = BATCH;
    FX_im.array = (float*)malloc(FX_im.height * FX_im.width * sizeof(float));

    // Setting input value
    srand(time(NULL));
    printf("The input is: \n");
    for (int j = 1; j <= BATCH; j++){
        printf("Vector %d: \n", j);
        for (int i = 1; i <= 4; i++){
            X_re.element(i, j) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            X_im.element(i, j) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            printf("X[%d] = (%.10f, %.10f) \n", i, X_re.element(i, j), X_im.element(i, j));
        }
    }

    status = fft4(BATCH, X_re,X_im, FX_re, FX_im);
    if (status != FFT_SUCCESS){
        printf("Error in running fft calculation\n");
        exit(1);
    }

    printf("Result: \n");
    for (int j = 1; j <= BATCH; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 1; i <= 4; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.element(i, j), FX_im.element(i, j));
        }
    }


    free(X_re.array);
    free(X_im.array);
    free(FX_re.array);
    free(FX_im.array);
    return 0;
}
