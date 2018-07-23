/*
 * A program testing batch stride function of cuBLAS gemm
 * Can batch stride be zero to make it essentially the same operand?
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <vector>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Helper
/* To process command line input */
#include "nvidia_helper/helper_string.h"
/* To check cuda state */
#include "nvidia_helper/checkCudaErrors.h"

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&duration, start, stop); _V.push_back(duration); cudaEventDestroy(start); cudaEventDestroy(stop);


const int B = 4;


float show_mean(std::vector<float> v)
{
    float sum = 0;
    for (int i = 0; i < v.size(); i++)
        sum += v[i];
    return sum / v.size(); 
}

int main(){
    half* fixed_op;
    half* batched_op;
    float* result;

    float alpha = 1.0f;
    float beta = 0.0f;
    //float* alpha_list;
    //float* beta_list; 

    /* Allocate memory for operand and result */
    checkCudaErrors(cudaMallocManaged((void **) &fixed_op, 4 * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &batched_op, 4 * 8 * B * sizeof (half)));
    checkCudaErrors(cudaMallocManaged((void **) &result, 4 * 8 * B * sizeof(float)));
    //checkCudaErrors(cudaMallocManaged((void **) &alpha_list, B * sizeof(float)));
    //checkCudaErrors(cudaMallocManaged((void **) &beta_list,  B * sizeof(float)));

    // Initialize input
    for (int i = 0; i < 4 * 4; i++){
        fixed_op[i] = 1.0f;
    }

    for (int j = 0; j < B; j++){
        for (int i = 0; i < 4 * 8; i++){
            batched_op[i + j * 4 * 8] = (float)i;
        }
        //alpha_list[j] = 0.0f;
        //beta_list[j] = 0.0f;
    }

    printf("Input: \n");
    for (int i = 0; i < 4 * 8; i++){
        printf("%f\t", (float)batched_op[i]);
    }
    printf("\n");

    // cublas variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;

    // Initialize cublas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        exit(-1);
    }

    // Define event, result data structure
    cudaEvent_t start, stop;
    std::vector<float> normal, batch;
    float duration;

    __START__
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 8, 4, &alpha, fixed_op,
                        CUDA_R_16F, 4, batched_op, CUDA_R_16F, 4, &beta, result, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (a * (c, d)).\n");
        exit(-1);
    }
    __STOP__(normal)

    printf("Normal result: \n");
    for (int i = 0; i < 4 * 8; i++){
        printf("%f\t", result[i]);
    }
    printf("\n");

    printf("Time of normal: %f milliseconds\n", show_mean(normal)); 

    long long int offset1 = 8 * 4 * sizeof(half);
    long long int offset2 = 8 * 4 * sizeof(float);
    long long int zero_offset = 0;

    __START__
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 8, 4, 4, &alpha, batched_op,
                        CUDA_R_16F, 8, offset1, fixed_op, CUDA_R_16F, 4, zero_offset, &beta, result, CUDA_R_32F, 8, offset2, B, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (a * (c, d)).\n");
        exit(-1);
    }
    __STOP__(batch)

    printf("Batch result: \n");
    for (int j = 0; j < B; j++){ 
        for (int i = 0; i < 4 * 8; i++){
            printf("%f\t", result[i]);
        }
        printf("\n");
    }

    printf("Batch size = %d, time of batch: %f milliseconds\n", B, show_mean(batch)); 

    __START__

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        exit(-1);
    }


    checkCudaErrors(cudaFree(fixed_op));
    checkCudaErrors(cudaFree(batched_op));
    checkCudaErrors(cudaFree(result));

    return 0;
}

