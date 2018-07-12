/*
 * Implementing fft4 algorithm
 * Input is multiple fp32 vector, number given by B
 * First split every input vector to two fp16 vectors (integrate in the same file)
 * It's not a complete FFT
 * To be used recursively by gfft
 * Using unified memory
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
#include <cublas_v2.h>

// Matrix and vector
#include "helper/my_vector.h"
#include "helper/my_matrix.h"
#include "helper/my_const.h"
#include "util/fp32_to_fp16.h"

// CUDA helper: to check error
#include "nvidia_helper/checkCudaErrors.h"

const float UPPER_BOUND = 1.0f;
const int BATCH = 4;
// const int BLOCKSIZE = 16;

fft::MatrixH F4_re;
fft::MatrixH F4_im;

FFT_S init_F4()
{
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

namespace fft{
    template <int BLOCK_SIZE> __global__ void _half_to_single(int size, half* input, float* output)
    {
        /* 
         * Convert the input half-precision vector to single-precision
         * Block and thread layout should be 1D
         * Block size need to be specified
         * */
        int bx = blockIdx.x;
        int tx = threadIdx.x;
        int index = bx * BLOCK_SIZE + tx;

        if (index < size) {
            output[index] = __half2float(input[index]);
        }
    }
}


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


    // Scale, combine and get result, store in the first quarter and third quarter of result1
    for (int j = 0; j < B; j++)
    {
        //// Scale FM_re * X_re_h and accumulate
        alpha = re_s1.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 0 + 4 * j, 1, FX_re.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_re_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_re * X_re_l and accumulate
        alpha = re_s2.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 1 + 4 * j, 1, FX_re.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_re_l and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_im_h and accumulate
        alpha = -1.0f * im_s1.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 2 + 4 * j, 1, FX_re.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_im_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_im_l and accumulate
        alpha = -1.0f * im_s2.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 3 + 4 * j, 1, FX_re.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_im_l and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_re * X_im_h and accumulate
        alpha = im_s1.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 2 + 4 * j, 1, FX_im.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_im_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_re * X_im_l and accumulate
        alpha = im_s2.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result1 + 4 * B * 3 + 4 * j, 1, FX_im.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_re * X_im_l and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_re_h and accumulate
        alpha = re_s1.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 0 + 4 * j, 1, FX_im.array + 4 * j, 1);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (Scale FM_im * X_re_h and accumulate).\n");
            return FFT_FAILURE;
        }

        //// Scale FM_im * X_re_l and accumulate
        alpha = re_s2.element(j + 1);
        status = cublasSaxpy(handle, 4, &alpha, result2 + 4 * B * 1 + 4 * j, 1, FX_im.array + 4 * j, 1);
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

    return FFT_SUCCESS;
}

int main()
{
    int mem_size;

    // Allocate unified memory for Fourier Matrix
    F4_re.width = 4;
    F4_re.height = 4;
    mem_size = F4_re.width * F4_re.height * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &(F4_re.array), mem_size));
    F4_im.width = 4;
    F4_im.height = 4;
    mem_size = F4_im.width * F4_im.height * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &(F4_im.array), mem_size));

    FFT_S status;
    status = init_F4();
    if (status != FFT_SUCCESS){
        printf("Error in Fourier matrix initialization\n");
        exit(1);
    }

    fft::MatrixF X_re;
    X_re.height = 4;
    X_re.width = BATCH;
    mem_size = X_re.height * X_re.width * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(X_re.array), mem_size));

    fft::MatrixF X_im;
    X_im.height = 4;
    X_im.width = BATCH;
    mem_size = X_im.height * X_im.width * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(X_im.array), mem_size));

    fft::MatrixF FX_re;
    FX_re.height = 4;
    FX_re.width = BATCH;
    mem_size = FX_re.height * FX_re.width * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(FX_re.array), mem_size));

    fft::MatrixF FX_im;
    FX_im.height = 4;
    FX_im.width = BATCH;
    mem_size = FX_im.height * FX_im.width * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(FX_im.array), mem_size));

    cudaDeviceSynchronize();

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

    status = fft4(BATCH, X_re, X_im, FX_re, FX_im);
    if (status != FFT_SUCCESS){
        printf("Error in running fft calculation\n");
        exit(1);
    }

    cudaDeviceSynchronize();

    printf("Result: \n");
    for (int j = 1; j <= BATCH; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 1; i <= 4; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.element(i, j), FX_im.element(i, j));
        }
    }

    checkCudaErrors(cudaFree(F4_re.array));
    checkCudaErrors(cudaFree(F4_im.array));

    checkCudaErrors(cudaFree(X_re.array));
    checkCudaErrors(cudaFree(X_im.array));
    checkCudaErrors(cudaFree(FX_re.array));
    checkCudaErrors(cudaFree(FX_im.array));

    return 0;
}
