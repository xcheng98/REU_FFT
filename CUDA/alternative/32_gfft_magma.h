/*
 * This version is WITHOUT splitting, all data are in FP32 type
 * Implementing the FFT algorithm for general input
 * To be called by testing program
 * Input should be fp32 vectors with size equals to the power of 4
 * Number of vectors is given by BATCH (B)
 * Recursive algorithm, base case is fft4
 * Combine all components in one file
 * Version after multiple optimizations
 * This implementation is without matrix and vector
 * This implementation uses global cublas handle
 */

#ifndef FFT_32_GFFT_H
#define FFT_32_GFFT_H

#include "improved_gfft_magma.h"

FFT_S gfft_32_recursion(int N, float* X_re, float* X_im, float*& FX_re, float*& FX_im, int B);

FFT_S init_F4_32();

FFT_S fft4_32(float* X_re, float* X_im, float* FX_re, float* FX_im, int B);

__global__ void myAccumulate_32(int N, float* X1, float* X2, float* X3, float* X4, float* R1, float* R2, int B);

FFT_S fft4_transposed_32(int M, float* X_re, float* X_im, float* FX_re, float* FX_im, int B);

__global__ void myAccumulate_transposed_32(int n, int M, float* X1, float* X2, float* X3, float* X4, float* R1, float* R2, int B);

cublasStatus_t status_32;
cublasHandle_t handle_32;

float* F4_re_32;
float* F4_im_32;
float* buffer_32;
float *result1_32, *result2_32, *result3_32, *result4_32; // F4_re * X_split, F4_im * X_split


int gfft_32(int SIZE, float* X_re, float* X_im, float* FX_re, float* FX_im, int BATCH)
{
    int mem_size;
    FFT_S fft_status;

    // Allocate memory for input and output matrix
    float *input_re, *input_im, *output_re, *output_im;
    mem_size = BATCH * SIZE * sizeof(float);
    checkCudaErrors(cudaMalloc((void **) &input_re, mem_size));
    checkCudaErrors(cudaMalloc((void **) &input_im, mem_size));
    checkCudaErrors(cudaMalloc((void **) &output_re, mem_size));
    checkCudaErrors(cudaMalloc((void **) &output_im, mem_size));

    // Copy input matrix to device memory
    checkCudaErrors(cudaMemcpy(input_re, X_re, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input_im, X_im, mem_size, cudaMemcpyHostToDevice));
    
    // Allocate unified memory for the buffer (global)
    mem_size = SIZE * BATCH * sizeof(float);
    checkCudaErrors(cudaMalloc((void **) &buffer_32, mem_size));

    // Allocate unified memory for temporary result (global)
    mem_size = SIZE * BATCH * sizeof(float); // re_hi, re_lo, im_hi, im_lo
    checkCudaErrors(cudaMalloc((void **) &result1_32, mem_size));
    checkCudaErrors(cudaMalloc((void **) &result2_32, mem_size));
    checkCudaErrors(cudaMalloc((void **) &result3_32, mem_size));
    checkCudaErrors(cudaMalloc((void **) &result4_32, mem_size));

    // Allocate mamory for and initialize Fourier matrix
    mem_size = 16 * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &F4_re_32, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &F4_im_32, mem_size));
    fft_status = init_F4_32();
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Matrix initialization error (Fourier matrix).\n");
        exit(1);
    }

    // Initialize cublas with global cublas handle and status
    status_32 = cublasCreate(&handle_32);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS initialization error.\n");
        exit(1);
    }

    // Allow cublas to use Tensor Core
    status_32 = cublasSetMathMode(handle_32, CUBLAS_TENSOR_OP_MATH); 
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS setting math mode error.\n");
        exit(1);
    }

    // Call gfft function
    fft_status = gfft_32_recursion(SIZE, input_re, input_im, output_re, output_im, BATCH);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! gFFT execution error.\n");
        exit(1);
    }

    // Shutdown cublas
    status_32 = cublasDestroy(handle_32);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS shutdown error.\n");
        exit(1);
    }

    // Deallocate unified memory for buffer and temporary result
    checkCudaErrors(cudaFree(F4_re_32));
    checkCudaErrors(cudaFree(F4_im_32));
    checkCudaErrors(cudaFree(buffer_32));
    checkCudaErrors(cudaFree(result1_32));
    checkCudaErrors(cudaFree(result2_32));
    checkCudaErrors(cudaFree(result3_32));
    checkCudaErrors(cudaFree(result4_32));

    // Copy result from device to host
    mem_size = BATCH * SIZE * sizeof(float);
    checkCudaErrors(cudaMemcpy(FX_re, output_re, mem_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(FX_im, output_im, mem_size, cudaMemcpyDeviceToHost));

    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Deallocate unified memory
    checkCudaErrors(cudaFree(input_re));
    checkCudaErrors(cudaFree(input_im));
    checkCudaErrors(cudaFree(output_re));
    checkCudaErrors(cudaFree(output_im));

    return 0;
}


FFT_S gfft_32_recursion(int N, float* X_re, float* X_im, float*& FX_re, float*& FX_im, int B) 
{
    // Base case
    if (N == 4) {
        return fft4_32(X_re, X_im, FX_re, FX_im, B);
    }

    // Status and error variable declaration
    FFT_S fft_status;
    cudaError_t cerror;

    // Declare temp variable for buffer swapping
    float* temp;

    // Transpose input matrix: 4 * (N/4*B) --> (N/4) * (4*B)
    // First store the result in buffer to avoid racing condition
    magmablas_stranspose_batched_stride(4, N / 4, N, X_re, 4, buffer, N / 4, B, NULL);
    temp = FX_re; FX_re = buffer; buffer = temp;

    magmablas_stranspose_batched_stride(4, N / 4, N, X_im, 4, buffer, N / 4, B, NULL);
    temp = FX_im; FX_im = buffer; buffer = temp;
    
    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Recursively call gfft function, NOT using buffer matrix
    fft_status = gfft_32_recursion(N / 4, FX_re, FX_im, FX_re, FX_im, 4 * B);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Function execution error (recursively call gfft).\n");
        return FFT_FAILURE;
    }

    // Multiplicate each element with twiddle factor
    //// Set grid and block size
    dim3 threadsPerBlock2(4, 16);
    dim3 blockPerGrid2(B, (N / 4 + 15)/16); // Make sure blocks are enough

    //// Call kernel function
    multiply_twiddle<<<blockPerGrid2, threadsPerBlock2>>>(N, N/4, 4, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during twiddle factor multiplication.\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Call the optimized fft4 function to avoid transposition
    fft_status = fft4_transposed_32(N / 4, FX_re, FX_im, FX_re, FX_im, B);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Function execution error (calling fft4_transposed).\n");
        return FFT_FAILURE;
    }
     
    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    return FFT_SUCCESS;
}

/* 
 * Initialize Fourier matrix
 * Allocate unified memory and set value for handle_32 and F4_im
 * */
FFT_S init_F4_32()
{
    F4_re_32[0] = 1.0f;
    F4_re_32[1] = 1.0f;
    F4_re_32[2] = 1.0f;
    F4_re_32[3] = 1.0f;
    F4_re_32[4] = 1.0f;
    F4_re_32[5] = 0.0f;
    F4_re_32[6] =-1.0f;
    F4_re_32[7] = 0.0f;
    F4_re_32[8] = 1.0f;
    F4_re_32[9] =-1.0f;
    F4_re_32[10] = 1.0f;
    F4_re_32[11] =-1.0f;
    F4_re_32[12] = 1.0f;
    F4_re_32[13] = 0.0f;
    F4_re_32[14] =-1.0f;
    F4_re_32[15] = 0.0f;

    F4_im_32[0] = 0.0f;
    F4_im_32[1] = 0.0f;
    F4_im_32[2] = 0.0f;
    F4_im_32[3] = 0.0f;
    F4_im_32[4] = 0.0f;
    F4_im_32[5] =-1.0f;
    F4_im_32[6] = 0.0f;
    F4_im_32[7] = 1.0f;
    F4_im_32[8] = 0.0f;
    F4_im_32[9] = 0.0f;
    F4_im_32[10] = 0.0f;
    F4_im_32[11] = 0.0f;
    F4_im_32[12] = 0.0f;
    F4_im_32[13] = 1.0f;
    F4_im_32[14] = 0.0f;
    F4_im_32[15] =-1.0f;

    return FFT_SUCCESS;
}


/* 
 * Perform fft on every length-4 vector
 * Batch size is given by B
 * Internally split every FP32 input into two FP16 vectors
 * Combine them together after FFT
 * */
FFT_S fft4_32(float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{
    // Variable declaration
    cudaError_t cerror;
    float alpha = 1.0f, beta = 0.0f; 
    // Temporary results are global variables

    // Split input
    //// Define segmentation pointers for convenience
    //float* X_re = X_split + 4 * B * 0;
    //float* X_im = X_split + 4 * B * 2;
    // float* re_s1 = scales + B * 0;
    // float* re_s2 = scales + B * 1;
    // float* im_s1 = scales + B * 2;
    // float* im_s2 = scales + B * 3;

    //// Call the splitting kernel
    //int numThreads = 64;
    //int numBlocks = (B + 63) / 64;
    //mySplit<<<numBlocks, numThreads>>>(4, X_re, X_re_hi, X_re_lo, re_s1, re_s2, B, buffer);
    //mySplit<<<numBlocks, numThreads>>>(4, X_im, X_im_hi, X_im_lo, im_s1, im_s2, B, buffer);
    //cerror = cudaGetLastError();
    //if (cerror != cudaSuccess)
    //{
       // fprintf(stderr, "!!!!! CUDA error: %s during fft4 splitting\n", cudaGetErrorString(cerror));
        //return FFT_FAILURE;
    //}
  
    // Matrix multiplication with Fourier matrix
    //// Call cublas gemm on F4_re * X_re
    status_32 = cublasGemmEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, 4, B, 4, &alpha, 
        F4_re_32, CUDA_R_32F, 4, X_re, CUDA_R_32F, 4, &beta,
        result1_32, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_re * X_re): %d.\n", status_32);
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_re * X_im
    status_32 = cublasGemmEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, 4, B, 4, &alpha, 
        F4_re_32, CUDA_R_32F, 4, X_im, CUDA_R_32F, 4, &beta,
        result2_32, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_re * X_im).\n");
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im * X_re
    status_32 = cublasGemmEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, 4, B, 4, &alpha, 
        F4_im_32, CUDA_R_32F, 4, X_re, CUDA_R_32F, 4, &beta,
        result3_32, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_im * X_re).\n");
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im * X_im
    status_32 = cublasGemmEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, 4, B, 4, &alpha, 
        F4_im_32, CUDA_R_32F, 4, X_im, CUDA_R_32F, 4, &beta,
        result4_32, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_im * X_im).\n");
        return FFT_FAILURE;
    }

    // Rescale the result and combine them together
    //// Set grid and block size
    dim3 threadsPerBlock(16, 4);
    dim3 blocksPerGrid((B+15)/16, 1);

    //// call kernel function (FX_re and FX_im will be zero-initialized)
    myAccumulate_32<<<blocksPerGrid, threadsPerBlock>>>(4, result1_32, result2_32, result3_32, result4_32, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during fft4 accumulation\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}



/* 
 * For (a + bi) * (c + di), re = ac - bd, im = ad + bc
 * Need to rescale the result before accumulation
 * N is number of elements in one vector (expected to be 4)
 * The number of vectors is given by B
 * X1, X2 are 4 * (B * 4) column-major matrix. Inner order is by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
 * alpha is B * 4 array. Inner order is by batch. Outer order is re_s1, re_s2, im_s1, im_s2
 * R1, R2 are resulting matrix of size 4 * B
 * */
__global__ void myAccumulate_32(int N, float* X1, float* X2, float* X3, float* X4, float* R1, float* R2, int B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < N && j < B){
        R1[i + j * N] = R2[i + j * N] = 0.0f;
        R1[i + j * N] += X1[i + j * N];
        R1[i + j * N] += -1.0f * X4[i + j * N];
        R2[i + j * N] += X2[i + j * N];
        R2[i + j * N] += X3[i + j * N];
    }
}

/* 
 * Perform fft4 assuming the input is in the transposed layout
 * The number of vectors is M * B
 * The number of rows of the input matrix is M
 * The number of columns of the input matrix is 4 * B (4 for radix 4)
 * Note that the fourier matrix is symmetric
 */
FFT_S fft4_transposed_32(int M, float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{
    // Variable declaration
    cudaError_t cerror;
    float alpha = 1.0f, beta = 0.0f; 
    // Temporary results are global variables

    // Split input
    //// Define segmentation pointers for convenience
    //float* X_re = X_split + M * 4 * B * 0;
    //float* X_im = X_split + M * 4 * B * 2;
    //float* re_s1 = scales + M * B * 0;
    //float* re_s2 = scales + M * B * 1;
    //float* im_s1 = scales + M * B * 2;
    //float* im_s2 = scales + M * B * 3;

    //// Call splitting function
    //dim3 threadsPerBlock1(4, 16);
    //dim3 blocksPerGrid1((B + 3)/4, (M + 15)/16);
    //mySplit_transposed<<<blocksPerGrid1, threadsPerBlock1>>>(4, M, X_re, X_re_hi, X_re_lo, re_s1, re_s2, B, buffer);
    //mySplit_transposed<<<blocksPerGrid1, threadsPerBlock1>>>(4, M, X_im, X_im_hi, X_im_lo, im_s1, im_s2, B, buffer);
    //cerror = cudaGetLastError();
    //if (cerror != cudaSuccess)
    //{
       // fprintf(stderr, "!!!!! CUDA error: %s during splitting in fft4_transposed\n", cudaGetErrorString(cerror));
        //return FFT_FAILURE;
    //}
   
    // Matrix multiplication with F4_re and F4_im
    // Note that the order of multiplicands are reversed
    //// Call batched gemm on X_re * F4_re
    status_32 = cublasGemmStridedBatchedEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, 
        X_re, CUDA_R_32F, M, M * 4, F4_re_32, CUDA_R_32F, 4, 0, 
        &beta, result1_32, CUDA_R_32F, M, M * 4, B, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error in fft4_transposed X_re*F4_re multiplication.\n");
        return FFT_FAILURE;
    }

    //// Call batched gemm on X_im * F4_re
    status_32 = cublasGemmStridedBatchedEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, 
        X_im, CUDA_R_32F, M, M * 4, F4_re_32, CUDA_R_32F, 4, 0, 
        &beta, result2_32, CUDA_R_32F, M, M * 4, B, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error in fft4_transposed X_im*F4_re multiplication.\n");
        return FFT_FAILURE;
    }

    //// Call batched gemm on X_re * F4_im
    status_32 = cublasGemmStridedBatchedEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, 
        X_re, CUDA_R_32F, M, M * 4, F4_im_32, CUDA_R_32F, 4, 0, 
        &beta, result3_32, CUDA_R_32F, M, M * 4, B, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error in fft4_transposed X_re*F4_im multiplication.\n");
        return FFT_FAILURE;
    }

    //// Call batched gemm on X_im * F4_im
    status_32 = cublasGemmStridedBatchedEx(handle_32, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, 
        X_im, CUDA_R_32F, M, M * 4, F4_im_32, CUDA_R_32F, 4, 0, 
        &beta, result4_32, CUDA_R_32F, M, M * 4, B, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status_32 != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error in fft4_transposed X_im*F4_im multiplication.\n");
        return FFT_FAILURE;
    }

    // Rescale the result and combine real and imaginary part
    //// Set grid and block size
    dim3 threadsPerBlock2(16, 16);
    dim3 blocksPerGrid2((4 * B + 15)/16, (M + 15)/16);

    //// call the accumulation kernel function (FX_re and FX_im will be zero-initialized inside)
    myAccumulate_transposed_32<<<blocksPerGrid2, threadsPerBlock2>>>(4, M, result1_32, result2_32, result3_32, result4_32, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during accumulation in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

/* 
 * The kernel rescales the multiplication result and accumulates them
 * Each thread works on one element (instead of one vector) in the resulting matrix
 * The length of one vector (unit) is given by n, expected to be 4
 * The total number of vectors is M * B
 * M is the vertical dimension, B is the horizontal dimension
 * X1, X2 are M * (4 * B * 4) matrices. The inner-most column order is by element in a unit. Then by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
 * alpha is a M * B * 4 arrays. Inner most order is by horizontal index. Then by batch. Outer order is re_s1, re_s2, im_s1, im_s2
 * R1, R2 are M * (4 * B) matrices
 * */
__global__ void myAccumulate_transposed_32(int n, int M, float* X1, float* X2, float* X3, float* X4, float* R1, float* R2, int B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // vertical index of the element, max M
    int j = blockIdx.x * blockDim.x + threadIdx.x; // horizontal index of the element, max 4 * B

    if (i < M && j < 4 * B){
        int result_idx = i + j * M;
        R1[result_idx] = R2[result_idx] = 0.0f;

        R1[result_idx] += X1[result_idx];
        R1[result_idx] += -1.0f * X4[result_idx];
        R2[result_idx] += X2[result_idx];
        R2[result_idx] += X3[result_idx];
    }
}

#endif /* FFT_32_GFFT_H */
