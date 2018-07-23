/*
 * Implementing the FFT algorithm for general input
 * Input should be fp32 vectors with size equals to the power of 4
 * Number of vectors is given by BATCH (B)
 * Recursive algorithm
 * Base case is fft4
 * Combine all components in one file
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
#include <cuda_fp16.h>

#include "nvidia_helper/checkCudaErrors.h"

// Matrix and vector
#include "helper/my_vector.h"
#include "helper/my_matrix.h"
#include "helper/my_const.h"

#define PI 3.14159265
#define EPS 0.0000001192f

const float UPPER_BOUND = 1.0f;
const int BATCH = 16;
const int SIZE = 256;


// Utility function declaration
FFT_S split_32_to_16(fft::MatrixF X, fft::MatrixH Xhi, fft::MatrixH Xlo, fft::VectorF s1, fft::VectorF s2, int N, int B);

FFT_S init_F4();

__global__ void myAccumulate(int N, float* X1, float* X2, float* alpha, float* R1, float* R2, int B);

FFT_S fft4(int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im);

__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im);

FFT_S gfft(int N, int B, fft::MatrixF& X_re, fft::MatrixF& X_im, fft::MatrixF& FX_re, fft::MatrixF& FX_im);


// Global variables
fft::MatrixH F4_re;
fft::MatrixH F4_im;
float* buffer;


int main()
{
    int mem_size;

    // allocate unified memory for input matrix
    fft::MatrixF input_re;
    input_re.width = BATCH;
    input_re.height = SIZE;
    mem_size = input_re.width * input_re.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(input_re.array), mem_size));
    fft::MatrixF input_im;
    input_im.width = BATCH;
    input_im.height = SIZE;
    mem_size = input_im.width * input_im.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(input_im.array), mem_size));

    // Initialize the input matrix
    srand(time(NULL));
    printf("The input is: \n");
    for (int j = 1; j <= BATCH; j++){
        printf("Vector %d: \n", j);
        for (int i = 1; i <= SIZE; i++){
            input_re.element(i, j) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            input_im.element(i, j) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            input_re.element(i, j) = (float)i;
            input_im.element(i, j) = 0.0f;
            printf("X[%d] = (%.10f, %.10f) \n", i, input_re.element(i, j), input_im.element(i, j));
        }
        printf("\n");
    }
    
    // allocate unified memory for output matrix
    fft::MatrixF output_re;
    output_re.width = BATCH;
    output_re.height = SIZE;
    mem_size = output_re.width * output_re.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(output_re.array), mem_size));
    fft::MatrixF output_im;
    output_im.width = BATCH;
    output_im.height = SIZE;
    mem_size = output_im.width * output_im.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(output_im.array), mem_size));

    // allocate unified memory for the buffer (array of float)
    mem_size = SIZE * BATCH * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &buffer, mem_size));


    FFT_S status;
    // Initialize Fourier matrix
    status = init_F4();
    if (status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Matrix initialization error (init Fourier matrix).\n");
        return FFT_FAILURE;
    }
    // Call gfft function
    status = gfft(SIZE, BATCH, input_re, input_im, output_re, output_im);
    if (status != FFT_SUCCESS){
        printf("Error in running fft algorithm\n");
        exit(1);
    }

    printf("Result: \n");
    for (int j = 1; j <= BATCH; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 1; i <= SIZE; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, output_re.element(i, j), output_im.element(i, j));
        }
    }

    checkCudaErrors(cudaFree(input_re.array));
    checkCudaErrors(cudaFree(input_im.array));
    checkCudaErrors(cudaFree(output_re.array));
    checkCudaErrors(cudaFree(output_im.array));

    return 0;
}

FFT_S split_32_to_16(fft::MatrixF X, fft::MatrixH Xhi, fft::MatrixH Xlo, fft::VectorF s1, fft::VectorF s2, int N, int B)
{
    // Calculate scaling factor 1
    for (int j = 1; j <= B; j++){
        float scale1 = 0.0f;
        for (int i = 1; i <= N; i++){
            float norm = (float) fabs(X.element(i, j));
            if (norm > scale1) scale1 = norm;
        }
        
        // If all number are zero, skip
       if (scale1 == 0.0f){
            s1.element(j) = 0.0f;
            continue;
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
        // If all number are zero, skip
        if (s1.element(j) == 0.0f){
            continue;
        }

        for (int i = 1; i <= N; i++) {
            Xtemp.element(i, j) = X.element(i, j) / s1.element(j);
            Xhi.element(i, j) = (half)(Xtemp.element(i, j));
            // Using Xtemp to store the residual
            Xtemp.element(i, j) = X.element(i, j) - s1.element(j) * (float)Xhi.element(i, j);
        }
    }

    // Calculate lower scaling factor
    for (int j = 1; j <= B; j++){
        // If all number are zero, skip
        if (s1.element(j) == 0.0f){
            continue;
        }

        float scale2 = 0.0f;
        for (int i = 1; i <= N; i++){
            float norm = (float)fabs(Xtemp.element(i, j));
            if (norm > scale2) scale2 = norm;
        }

        // If all remainders are zero, skip
    if (scale2 == 0.0f){
            s2.element(j) = 0.0f;
            continue;
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
        // If all number are zero, skip
        if (s1.element(j) == 0.0f){
            continue;
        }

        // If all remainders are zero, set X_lo to zero
    if (s2.element(j) == 0.0f){
            for (int i = 1; i <= N; i++){
                Xlo.element(i, j) = (half) 0.0f;
            }
            continue;
    }

        for (int i = 1; i <= N; i++){
            Xtemp.element(i, j) = Xtemp.element(i, j) / s2.element(j);
            Xlo.element(i, j) = (half) (Xtemp.element(i, j));
        }
    }

    free(Xtemp.array);   
    
    // Deal with zero case
    for (int j = 1; j <= B; j++){
        if (s1.element(j) == 0.0f){
            s2.element(j) == 0.0f;
            for (int i = 1; i <= N; i++){
                Xhi.element(i, j) = (half) 0.0f;
                Xlo.element(i, j) = (half) 0.0f;
            }
        }
    }

    return FFT_SUCCESS;
}


FFT_S init_F4()
{
    // Allocate unified memory for Fourier Matrix
    int mem_size;

    F4_re.width = 4;
    F4_re.height = 4;
    mem_size = F4_re.width * F4_re.height * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &(F4_re.array), mem_size));

    F4_im.width = 4;
    F4_im.height = 4;
    mem_size = F4_im.width * F4_im.height * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &(F4_im.array), mem_size));

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


__global__ void myAccumulate(int N, float* X1, float* X2, float* alpha, float* R1, float* R2, int B)
{
    /* 
     * N is number of elements (always 4)
     * X1, X2 are 4 * (B * 4) column-major matrix. Inner order is by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
     * alpha is B * 4 array. Inner order is by batch. Outer order is re_s1, re_s2, im_s1, im_s2
     * R1, R2 are 4 * B matrix
     * B is batch size
     * */
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < N && j < B){
        R1[i + j * N] += alpha[j] * X1[i + j * N];
        R1[i + j * N] += alpha[j + B] * X1[i + j * N + N * B];
        R1[i + j * N] += -1.0f * alpha[j + 2*B] * X2[i + j * N + N * 2 * B];
        R1[i + j * N] += -1.0f * alpha[j + 3*B] * X2[i + j * N + N * 3 * B];
        R2[i + j * N] += alpha[j] * X2[i + j * N];
        R2[i + j * N] += alpha[j + B] * X2[i + j * N + N * B];
        R2[i + j * N] += alpha[j + 2*B] * X1[i + j * N + N * 2 * B];
        R2[i + j * N] += alpha[j + 3*B] * X1[i + j * N + N * 3 * B];
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
    checkCudaErrors(cudaMallocManaged((void **) &result2, 4 * B * 4 * sizeof(result2[0])));
    checkCudaErrors(cudaMemset(result2, 0.0f, 4 * B * 4 * sizeof(result2[0])));

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
    //// Set grid and block size
    dim3 threadsPerBlock(16, 4);
    dim3 numBlocks((B+15)/16, 1);

    //// Call kernel function
    myAccumulate<<<numBlocks, threadsPerBlock>>>(4, result1, result2, scales, FX_re.array, FX_im.array, B);


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


__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im)
{
    /* 
     * Multifly every element of the input matrix with twiddle factor
     * Block and thread layout should be 2D
     * Re.element(i, j) [0 based] = xre * cos(2pi/N * i * j) + xim * sin(2pi/N * i * j)
     * Im.element(i, j) [0 based] = -xre * sin(2pi/N * i * j) + xim * cos(2pi/N * i * j)
     * */

    // Calculate position (0 based)
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < n){
        // Per-thread local variables
        int index = j * m + i;
        float tw_re = cos(2 * PI / N * i * j);
        float tw_im = sin(2 * PI / N * i * j);
        float result_re = matrix_re[index] * tw_re + matrix_im[index] * tw_im;
        float result_im = -1.0f * matrix_re[index] * tw_im + matrix_im[index] * tw_re;

        matrix_re[index] = result_re;
        matrix_im[index] = result_im;
    }
}


FFT_S gfft(int N, int B, fft::MatrixF& X_re, fft::MatrixF& X_im, fft::MatrixF& FX_re, fft::MatrixF& FX_im) 
{
    FFT_S fft_status;

    if (N == 4) {
        return fft4(B, X_re, X_im, FX_re, FX_im);
    }

    // cublas variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    // Scaling variables
    float alpha = 1.0f, beta = 0.0f; 
    // Temporary variables for intermediate result swapping
    float* temp;

    // Initialize cublas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return FFT_FAILURE;
    }


    // Reshape the output matrix: (N -(Reshape)->4*(N/4)) * B
    FX_re.width = N / 4 * B; FX_re.height = 4;
    FX_im.width = N / 4 * B; FX_im.height = 4;


    // Transpose input matrix: (4*(N/4) -(Transpose)-> (N/4)*4) * B
    // Store temporary result first in buffer, then in FX_re.array and FX_im.array
    //// Real matrix
    for (int j = 0; j < B; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N/4, 4, &alpha, X_re.array + j * N, 4, 
                            &beta, X_re.array + j * N, 4, buffer + j * N, N/4);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (transpose real input).\n");
            return FFT_FAILURE;
        }
    }
    ////// Swap FX_re.array and buffer to store the transposition result in FX_re.array
    temp = FX_re.array; FX_re.array = buffer; buffer = temp;
    ////// Set dimension (Note that the transpose happens batch-wisely)
    FX_re.height = N / 4; FX_re.width = B * 4;

    //// Imaginary 
    for (int j = 0; j < B; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N/4, 4, &alpha, X_im.array + j * N, 4, 
                            &beta, X_im.array + j * N, 4, buffer + j * N, N/4);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (transpose imaginary input).\n");
            return FFT_FAILURE;
        }
    }
    ////// Swap FX_im.array and buffer to store the transposition result in FX_im.array
    temp = FX_im.array; FX_im.array = buffer; buffer = temp;
    ////// Set dimension
    FX_im.height = N / 4; FX_im.width = B * 4;

    cudaDeviceSynchronize();


    // Recursively call gfft function, not! using buffer matrix
    //// Call gfft, store result in buffer matrix
    fft_status = gfft(N / 4, 4 * B, FX_re, FX_im, FX_re, FX_im);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Execution error (recursively call gfft).\n");
        return FFT_FAILURE;
    }

    // Multiplication with twiddle factors
    //// Set grid and block size
    dim3 threadsPerBlock(4, 16);
    dim3 numBlocks(1, (N + 63)/64); // Make sure blocks are enough

    //// Call kernel function
    for (int j = 0; j < B; j++){
        multiply_twiddle<<<numBlocks, threadsPerBlock>>>(N, N/4, 4, FX_re.array + j * N, FX_im.array + j * N);
    }

    cudaDeviceSynchronize();


    // Transpose the matrix again
    // Store temporary result first in buffer, then in FX_re.array and FX_im.array
    //// Real matrix
    for (int j = 0; j < B; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, 4, N/4, &alpha, FX_re.array + j * N, N/4, 
                            &beta, FX_re.array + j * N, N/4, buffer + j * N, 4);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (intermediate transpose real).\n");
            return FFT_FAILURE;
        }
    }
    ////// Swap FX_re.array and buffer to store the transposition result in FX_re.array
    temp = FX_re.array; FX_re.array = buffer; buffer = temp;
    ////// Set dimension, note that the transpose happens per batch
    FX_re.height = 4; FX_re.width = N / 4 * B;

    //// Imaginary matrix
    for (int j = 0; j < B; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, 4, N/4, &alpha, FX_im.array + j * N, N/4, 
                            &beta, FX_im.array + j * N, N/4, buffer + j * N, 4);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (intermediate transpose imaginary).\n");
            return FFT_FAILURE;
        }
    }
    ////// Swap FX_im.array and buffer to store the transposition result in FX_im.array
    temp = FX_im.array; FX_im.array = buffer; buffer = temp;
    ////// Set dimension
    FX_im.height = 4; FX_im.width = N / 4 * B;

    cudaDeviceSynchronize();


    // Call fft4, not! using buffer matrix
    //// Call fft4, store result in buffer matrix
    fft_status = fft4(N / 4 * B, FX_re, FX_im, FX_re, FX_im);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Execution error (combine step calling fft4).\n");
        return FFT_FAILURE;
    }

    // Do the final transpose to get the output
    //// Real matrix
    for (int j = 0; j < B; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N/4, 4, &alpha, FX_re.array + j * N, 4, 
                            &beta, FX_re.array + j * N, 4, buffer + j * N, N/4);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (final transpose real).\n");
            return FFT_FAILURE;
        }
    }
    ////// Swap FX_re.array and buffer to store the transposition result in FX_re.array
    temp = FX_re.array; FX_re.array = buffer; buffer = temp;
    ////// Set dimension
    FX_re.height = N / 4; FX_re.width = 4 * B;

    //// Imaginary matrix
    for (int j = 0; j < B; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N/4, 4, &alpha, FX_im.array + j * N, 4, 
                            &beta, FX_im.array + j * N, 4, buffer + j * N, N/4);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (final transpose imaginary).\n");
            return FFT_FAILURE;
        }
    }
    ////// Swap FX_im.array and buffer to store the transposition result in FX_im.array
    temp = FX_im.array; FX_im.array = buffer; buffer = temp;
    ////// Set dimension
    FX_re.height = N / 4; FX_re.width = 4 * B;

    cudaDeviceSynchronize();


    // Reshape back input and output matrix: (4*(N/4) --Reshape--> N) * B
    FX_re.width = B; FX_re.height = N;
    FX_im.width = B; FX_im.height = N;


    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}
