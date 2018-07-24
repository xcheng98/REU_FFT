// C includes
#include <stdio.h>
#include <stdlib.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "nvidia_helper/checkCudaErrors.h"

int UPPER_BOUND = 4096;

int main(){
    half* F4_re;
    half* X_split;
    float* result1;

    int M = 16;
    int B = 256*64;


    checkCudaErrors(cudaMallocManaged((void **) &F4_re, 4 * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &X_split, M * 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &result1, M * 4 * B * 4 * sizeof(float));

    F4_re(1, 1) = 1.0f;
    F4_re(2, 1) = 1.0f;
    F4_re(3, 1) = 1.0f;
    F4_re(4, 1) = 1.0f;
    F4_re(1, 2) = 1.0f;
    F4_re(2, 2) = 0.0f;
    F4_re(3, 2) =-1.0f;
    F4_re(4, 2) = 0.0f;
    F4_re(1, 3) = 1.0f;
    F4_re(2, 3) =-1.0f;
    F4_re(3, 3) = 1.0f;
    F4_re(4, 3) =-1.0f;
    F4_re(1, 4) = 1.0f;
    F4_re(2, 4) = 0.0f;
    F4_re(3, 4) =-1.0f;
    F4_re(4, 4) = 0.0f;

    srand(time(NULL));
    for (int i = 0; i < M * 4 * B * 4; i++){
        X_split = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
    }

    cublasStatus_t status;
    cublasHandle_t handle;
    float alpha = 1.0f, beta = 0.0f; 

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return exit(1);
    }
    status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // allow Tensor Core
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS setting math mode error\n");
        return exit(1);
    }

    long long int stride = M * 4;

    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, X_split,
                        CUDA_R_16F, M, stride, F4_re, CUDA_R_16F, 4, 0, &beta, result1, CUDA_R_32F, M, stride, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error: %d .\n", status);
        exit(1);
    }

    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return FFT_FAILURE;
    }

    checkCudaErrors(cudaFree(F4_re));
    checkCudaErrors(cudaFree(X_split));
    checkCudaErrors(cudaFree(result1));

    return 0;
}