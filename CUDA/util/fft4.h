/*
 * fft4 algorithm header
 * Using fp16
 * Using unified memory
 * To be used recursively by gfft
 */

#ifndef FFT_FFT4_H
#define FFT_FFT4_H

// C includes
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Matrix and vector
#include "../helper/my_vector.h"
#include "../helper/my_matrix.h"
#include "../helper/my_const.h"

// Other utilities
#include "fp32_to_fp16.h"
#include "fourier_matrix_4.h"

// CUDA helper: to check error
#include "../nvidia_helper/checkCudaErrors.h"

extern fft::MatrixH F4_re;
extern fft::MatrixH F4_im;

FFT_S fft4(int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im) 
{
    // Variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;

    //// Unified variables
    float *scales; // = *re_s1, *re_s2, *im_s1, *im_s2;
    half *X_split; // = *X_re_hi, *X_re_lo, *X_im_hi, *X_im_lo;
    float *result1, *result2; // Store the intermediate result
    //// Scaling variables
    float alpha = 1.0f, beta = 0.0f; 

    // Initialize cublas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return FFT_FAILURE;
    }

    status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // allow Tensor Core
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS setting math mode error\n");
        return FFT_FAILURE;
    }

    //  Allocate unified memory with 0 initialization
    checkCudaErrors(cudaMallocManaged((void **) &scales, B * 4 * sizeof(float)));
    checkCudaErrors(cudaMemset(scales, 0.0f, B * 4 * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **) &X_split, 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMemset(X_split, 0.0f, 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &result1, 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMemset(result1, 0.0f, 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMallocManaged((void **) &result2, 4 * 4 * sizeof(result2[0])));
    checkCudaErrors(cudaMemset(result2, 0.0f, 4 * 4 * sizeof(result2[0])));

    // Split input
    //// Initialize Matrix and Vector data structure to store split result
    fft::MatrixH X_re_hi;
    X_re_hi.width = B;
    X_re_hi.height = 4;
    X_re_hi.array = X_split + 4 * B * 0;

    fft::MatrixH X_re_lo;
    X_re_lo.width = B;
    X_re_lo.height = 4;
    X_re_lo.array = X_split + 4 * B * 1;

    fft::MatrixH X_im_hi;
    X_im_hi.width = B;
    X_im_hi.height = 4;
    X_im_hi.array = X_split + 4 * B * 2;

    fft::MatrixH X_im_lo;
    X_im_lo.width = B;
    X_im_lo.height = 4;
    X_im_lo.array = X_split + 4 * B * 3;

    fft::VectorF re_s1;
    re_s1.size = B;
    re_s1.array = scales + B * 0;

    fft::VectorF re_s2;
    re_s2.size = B;
    re_s2.array = scales + B * 1;

    fft::VectorF im_s1;
    im_s1.size = B;
    im_s1.array = scales + B * 2;

    fft::VectorF im_s2;
    im_s2.size = B;
    im_s2.array = scales + B * 3;

    //// Call splitting function
    FFT_S fft_status;

    fft_status = split_32_to_16(X_re, X_re_hi, X_re_lo, re_s1, re_s2, 4, B);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Data splitting error (split X_re).\n");
        return FFT_FAILURE;
    }

    fft_status = split_32_to_16(X_im, X_im_hi, X_im_lo, im_s1, im_s2, 4, B);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Data splitting error (split X_im).\n");
        return FFT_FAILURE;
    }

    
    // Call cublas function and finish Matrix multiplication calculation
    //// Call cublas gemm on F4_re
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 4, 4, &alpha, F4_re.array,
                        CUDA_R_16F, 4, X_split, CUDA_R_16F, 4, &beta, result1, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (a * (c, d)).\n");
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 4, 4, &alpha, F4_im.array,
                        CUDA_R_16F, 4, X_split, CUDA_R_16F, 4, &beta, result2, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (b * (c, d)).\n");
        return FFT_FAILURE;
    }


    // Make sure output is clean to write (0 initialization)
    checkCudaErrors(cudaMemset(FX_re.array, 0.0f, 4 * B * sizeof(float)));
    checkCudaErrors(cudaMemset(FX_im.array, 0.0f, 4 * B * sizeof(float)));


    // Scale, combine and get result, add to output
    for (int j = 1; j <= B; j++)
    {
        //// Scale FM_re * X_re_h and accumulate
        alpha = re_s1.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 0 + 4 * (j - 1), 1, (float*)&(FX_re.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_re_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_re * X_re_l and accumulate
        alpha = re_s2.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 1 + 4 * (j - 1), 1, (float*)&(FX_re.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_re_l and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_im_h and accumulate
        alpha = -1.0f * im_s1.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 2 + 4 * (j - 1), 1, (float*)&(FX_re.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_im_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_im_l and accumulate
        alpha = -1.0f * im_s2.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 3 + 4 * (j - 1), 1, (float*)&(FX_re.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_im_l and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_re * X_im_h and accumulate
        alpha = im_s1.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 2 + 4 * (j - 1), 1, (float*)&(FX_im.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_im_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_re * X_im_l and accumulate
        alpha = im_s2.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 3 + 4 * (j - 1), 1, (float*)&(FX_im.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_im_l and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_re_h and accumulate
        alpha = re_s1.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 0 + 4 * (j - 1), 1, (float*)&(FX_im.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_re_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_re_l and accumulate
        alpha = re_s2.element(j);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 1 + 4 * (j - 1), 1, (float*)&(FX_im.element(1, j)), 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_re_l and accumulate).\n");
            return FFT_FAILURE;
        }
    }


    // Deallocate unified memory
    if (cudaFree(scales) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free scales vector)\n");
        return FFT_FAILURE;
    }

    if (cudaFree(X_split) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free split result matrix)\n");
        return FFT_FAILURE;
    }

    if (cudaFree(result1) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free result 1 Matrix)\n");
        return FFT_FAILURE;
    }

    if (cudaFree(result2) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free result 2 Matrix)\n");
        return FFT_FAILURE;
    }

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return FFT_FAILURE;
    }

    cudaDeviceSynchronize();

    return FFT_SUCCESS;
}

#endif /* FFT_FFT4_H */
