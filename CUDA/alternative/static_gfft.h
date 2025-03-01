/*
 * Implementing the FFT algorithm for general input
 * To be called by testing program
 * This version split FP32 to FP16 with static factor
 * Input should be fp32 vectors with size equals to the power of 4
 * Number of vectors is given by BATCH (B)
 * Recursive algorithm, base case is fft4
 * Combine all components in one file
 * Version after multiple optimizations
 * This implementation is without matrix and vector
 * This implementation uses global cublas handle
 * This implementation makes buffer memory like X_split, scales, result1/2 global
 */

#ifndef FFT_IMPROVED_GFFT_H
#define FFT_IMPROVED_GFFT_H

#include "../util/my_include_combined.h"

#define PI 3.14159265
#define EPS 0.0000001192f

FFT_S gfft_recursion(int N, float* X_re, float* X_im, float*& FX_re, float*& FX_im, int B);
 
__global__ void myTranspose(int m, int n, float* input, float* output, int B);

__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im, int B);

FFT_S init_F4();

FFT_S fft4(float* X_re, float* X_im, float* FX_re, float* FX_im, int B);

__global__ void mySplit(int N, float* X, half* Xhi, half* Xlo, int B);

__global__ void myAccumulate(int N, float* X1, float* X2, float* R1, float* R2, int B);

FFT_S fft4_transposed(int M, float* X_re, float* X_im, float* FX_re, float* FX_im, int B);

__global__ void mySplit_transposed(int n, int M, float* X, half* Xhi, half* Xlo, int B);

__global__ void myAccumulate_transposed(int n, int M, float* X1, float* X2, float* R1, float* R2, int B);


// Static scaling factors
#define s1 1.0f
#define s2 0.001f

cublasStatus_t status;
cublasHandle_t handle;

half* F4_re;
half* F4_im;
float* buffer;
//float* scales; // = re_s1, re_s2, im_s1, im_s2;
half* X_split; // = X_re_hi, X_re_lo, X_im_hi, X_im_lo;
float *result1, *result2; // F4_re * X_split, F4_im * X_split


int gfft(int SIZE, float* X_re, float* X_im, float* FX_re, float* FX_im, int BATCH)
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
    checkCudaErrors(cudaMalloc((void **) &buffer, mem_size));

    // Allocate unified memory for temporary result (global)
    //mem_size = SIZE / 4 * BATCH * 4 * sizeof(float); // Unit length = 4, re_s1, re_s2, im_s1, im_s2
    //checkCudaErrors(cudaMalloc((void **) &scales, mem_size));
    mem_size = SIZE * BATCH * 4 * sizeof(half); // re_hi, re_lo, im_hi, im_lo
    checkCudaErrors(cudaMalloc((void **) &X_split, mem_size));
    mem_size = SIZE * BATCH * 4 * sizeof(float); // re_hi, re_lo, im_hi, im_lo
    checkCudaErrors(cudaMalloc((void **) &result1, mem_size));
    checkCudaErrors(cudaMalloc((void **) &result2, mem_size));

    // Allocate mamory for and initialize Fourier matrix
    mem_size = 16 * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &F4_re, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &F4_im, mem_size));
    fft_status = init_F4();
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Matrix initialization error (Fourier matrix).\n");
        exit(1);
    }

    // Initialize cublas with global cublas handle and status
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS initialization error.\n");
        exit(1);
    }

    // Allow cublas to use Tensor Core
    status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); 
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS setting math mode error.\n");
        exit(1);
    }

    // Call gfft function
    fft_status = gfft_recursion(SIZE, input_re, input_im, output_re, output_im, BATCH);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! gFFT execution error.\n");
        exit(1);
    }

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS shutdown error.\n");
        exit(1);
    }

    // Deallocate unified memory for buffer and temporary result
    checkCudaErrors(cudaFree(F4_re));
    checkCudaErrors(cudaFree(F4_im));
    checkCudaErrors(cudaFree(buffer));
    //checkCudaErrors(cudaFree(scales));
    checkCudaErrors(cudaFree(X_split));
    checkCudaErrors(cudaFree(result1));
    checkCudaErrors(cudaFree(result2));

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


FFT_S gfft_recursion(int N, float* X_re, float* X_im, float*& FX_re, float*& FX_im, int B) 
{
    // Base case
    if (N == 4) {
        return fft4(X_re, X_im, FX_re, FX_im, B);
    }

    // Status and error variable declaration
    FFT_S fft_status;
    cudaError_t cerror;

    // Declare temp variable for buffer swapping
    float* temp;

    // Transpose input matrix: 4 * (N/4*B) --> (N/4) * (4*B)
    // First store the result in buffer to avoid racing condition
    //// Set grid and block size
    dim3 threadsPerBlock1(4, 16);
    dim3 blockPerGrid1(B, (N / 4 + 15)/16); // Make sure blocks are enough

    //// Transpose real matrix
    myTranspose<<<blockPerGrid1, threadsPerBlock1>>>(4, N / 4, X_re, buffer, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during transposition of real matrix.\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    //// Swap FX_re and buffer to store the transposition result in FX_re
    temp = FX_re; FX_re = buffer; buffer = temp;

    //// Transpose imaginary matrix
    myTranspose<<<blockPerGrid1, threadsPerBlock1>>>(4, N / 4, X_im, buffer, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during transposition of imaginary matrix.\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
    ////// Swap FX_im and buffer to store the transposition result in FX_im
    temp = FX_im; FX_im = buffer; buffer = temp;

    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Recursively call gfft function, NOT using buffer matrix
    fft_status = gfft_recursion(N / 4, FX_re, FX_im, FX_re, FX_im, 4 * B);
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
    fft_status = fft4_transposed(N / 4, FX_re, FX_im, FX_re, FX_im, B);
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
 * Allocate unified memory and set value for F4_re and F4_im
 * */
FFT_S init_F4()
{
    F4_re[0] = 1.0f;
    F4_re[1] = 1.0f;
    F4_re[2] = 1.0f;
    F4_re[3] = 1.0f;
    F4_re[4] = 1.0f;
    F4_re[5] = 0.0f;
    F4_re[6] =-1.0f;
    F4_re[7] = 0.0f;
    F4_re[8] = 1.0f;
    F4_re[9] =-1.0f;
    F4_re[10] = 1.0f;
    F4_re[11] =-1.0f;
    F4_re[12] = 1.0f;
    F4_re[13] = 0.0f;
    F4_re[14] =-1.0f;
    F4_re[15] = 0.0f;

    F4_im[0] = 0.0f;
    F4_im[1] = 0.0f;
    F4_im[2] = 0.0f;
    F4_im[3] = 0.0f;
    F4_im[4] = 0.0f;
    F4_im[5] =-1.0f;
    F4_im[6] = 0.0f;
    F4_im[7] = 1.0f;
    F4_im[8] = 0.0f;
    F4_im[9] = 0.0f;
    F4_im[10] = 0.0f;
    F4_im[11] = 0.0f;
    F4_im[12] = 0.0f;
    F4_im[13] = 1.0f;
    F4_im[14] = 0.0f;
    F4_im[15] =-1.0f;

    return FFT_SUCCESS;
}


/* 
 * Transpose every input matrix of size m * n
 * Number of matrices is given by B 
 * Every matrix in a batch is transposed independently
 * Input is expected to be matrix of size m * (n * B)
 * Output is expected to be matrix of size n * (m * B)
 * The grid size is expected to be B in horizontal dimension
 * Usage: transpose a matrix of size 4 * (N/4 * B) to (N/4) * (4 * B)
 * */
__global__ void myTranspose(int m, int n, float* input, float* output, int B)
{
    // Calculate position in the OUTPUT matrix (0 based)
    int j = threadIdx.x; // Column number within a matrix, expected to be 0, 1, 2, 3
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row number
    int matrix_id = blockIdx.x; // The index of matrix in the batch

    if (i < n && j < m && matrix_id < B){
        output[matrix_id * m * n + j * n + i] = input[matrix_id * m * n + i * m + j];
    }
}

/* 
 * Multifly every element of the input matrix with the twiddle factor
 * Every matrix in a batch is processed independently
 * Block and thread layout should be 2D, and the total dimension is expected to be (m, n * B)
 * n is expected to be 4
 * result.re(i, j) [0 based] = xre(i, j) * cos(2pi/N * i * j) + xim(i, j) * sin(2pi/N * i * j)
 * result.im(i, j) [0 based] = -xre(i, j) * sin(2pi/N * i * j) + xim(i, j) * cos(2pi/N * i * j)
 * ONLY that thread will access the particular matrix_re and matrix_im, so buffer is not needed
 * */
__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im, int B)
{
    // Calculate position
    int j = threadIdx.x; // Column number within a matrix, 0 to 3 in radix 4
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row number within a matrix
    int matrix_id = blockIdx.x; // Index of matrix in the batch

    if (i < m && j < n && matrix_id < B){
        // Per-thread local variables
        int index = matrix_id * N + j * m + i;
        float tw_re = cos(2 * PI / N * i * j);
        float tw_im = sin(2 * PI / N * i * j);
        float result_re = matrix_re[index] * tw_re + matrix_im[index] * tw_im;
        float result_im = -1.0f * matrix_re[index] * tw_im + matrix_im[index] * tw_re;

        matrix_re[index] = result_re;
        matrix_im[index] = result_im;
    }
}


/* 
 * Perform fft on every length-4 vector
 * Batch size is given by B
 * Internally split every FP32 input into two FP16 vectors
 * Combine them together after FFT
 * */
FFT_S fft4(float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{
    // Variable declaration
    cudaError_t cerror;
    float alpha = 1.0f, beta = 0.0f; 
    // Temporary results are global variables

    // Split input
    //// Define segmentation pointers for convenience
    half* X_re_hi = X_split + 4 * B * 0;
    half* X_re_lo = X_split + 4 * B * 1;
    half* X_im_hi = X_split + 4 * B * 2;
    half* X_im_lo = X_split + 4 * B * 3;

    //// Call the splitting kernel
    int numThreads = 64;
    int numBlocks = (B + 63) / 64;
    mySplit<<<numBlocks, numThreads>>>(4, X_re, X_re_hi, X_re_lo, B);
    mySplit<<<numBlocks, numThreads>>>(4, X_im, X_im_hi, X_im_lo, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during fft4 splitting\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
  
    // Matrix multiplication with Fourier matrix
    //// Call cublas gemm on F4_re
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 4, 4, &alpha, 
        F4_re, CUDA_R_16F, 4, X_split, CUDA_R_16F, 4, &beta,
        result1, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_re * X_split).\n");
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 4, 4, &alpha,
        F4_im, CUDA_R_16F, 4, X_split, CUDA_R_16F, 4, &beta,
        result2, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_im * X_split).\n");
        return FFT_FAILURE;
    }

    // Rescale the result and combine them together
    //// Set grid and block size
    dim3 threadsPerBlock(16, 4);
    dim3 blocksPerGrid((B+15)/16, 1);

    //// call kernel function (FX_re and FX_im will be zero-initialized)
    myAccumulate<<<blocksPerGrid, threadsPerBlock>>>(4, result1, result2, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during fft4 accumulation\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}


/* 
 * N is expected to be 4
 * Split each FP32 vector to two FP16 vectors
 * Store the scaling factor in s1 and s2
 * The number of vectors is specified by B
 * The grid and block size are expected to be 1D, total size = B
 * Xtemp is of the same size as X, it is passed in to store temporary result
 * All data should be in unified memory or device memory
 * */
__global__ void mySplit(int N, float* X, half* Xhi, half* Xlo, int B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B){
        for (int i = 0; i < N; i++){
            Xhi[i + idx * N] = X[i + idx * N]/s1;
            Xlo[i + idx * N] = (X[i + idx * N] - (float)Xhi[i + idx * N] * s1) / s2;
        }
    }
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
__global__ void myAccumulate(int N, float* X1, float* X2, float* R1, float* R2, int B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < N && j < B){
        R1[i + j * N] = R2[i + j * N] = 0.0f;
        R1[i + j * N] += s1 * X1[i + j * N];
        R1[i + j * N] += s2 * X1[i + j * N + N * B];
        R1[i + j * N] += -1.0f * s1 * X2[i + j * N + N * 2 * B];
        R1[i + j * N] += -1.0f * s2 * X2[i + j * N + N * 3 * B];
        R2[i + j * N] += s1 * X2[i + j * N];
        R2[i + j * N] += s2 * X2[i + j * N + N * B];
        R2[i + j * N] += s1 * X1[i + j * N + N * 2 * B];
        R2[i + j * N] += s2 * X1[i + j * N + N * 3 * B];
    }
}

/* 
 * Perform fft4 assuming the input is in the transposed layout
 * The number of vectors is M * B
 * The number of rows of the input matrix is M
 * The number of columns of the input matrix is 4 * B (4 for radix 4)
 * Note that the fourier matrix is symmetric
 */
FFT_S fft4_transposed(int M, float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{
    // Variable declaration
    cudaError_t cerror;
    float alpha = 1.0f, beta = 0.0f; 
    // Temporary results are global variables

    // Split input
    //// Define segmentation pointers for convenience
    half* X_re_hi = X_split + M * 4 * B * 0;
    half* X_re_lo = X_split + M * 4 * B * 1;
    half* X_im_hi = X_split + M * 4 * B * 2;
    half* X_im_lo = X_split + M * 4 * B * 3;

    //// Call splitting function
    dim3 threadsPerBlock1(4, 16);
    dim3 blocksPerGrid1((B + 3)/4, (M + 15)/16);
    mySplit_transposed<<<blocksPerGrid1, threadsPerBlock1>>>(4, M, X_re, X_re_hi, X_re_lo, B);
    mySplit_transposed<<<blocksPerGrid1, threadsPerBlock1>>>(4, M, X_im, X_im_hi, X_im_lo, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during splitting in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
   
    // Matrix multiplication with F4_re and F4_im
    // Note that the order of multiplicands are reversed
    //// Call batched gemm on F4_re
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, 
        X_split, CUDA_R_16F, M, M * 4, F4_re, CUDA_R_16F, 4, 0, 
        &beta, result1, CUDA_R_32F, M, M * 4, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error in fft4_transposed F4_re multiplication.\n");
        return FFT_FAILURE;
    }

    //// Call batched gemm on F4_im
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, 
        X_split, CUDA_R_16F, M, M * 4, F4_im, CUDA_R_16F, 4, 0, 
        &beta, result2, CUDA_R_32F, M, M * 4, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error in fft4_transposed F4_im multiplication.\n");
        return FFT_FAILURE;
    }

    // Rescale the result and combine real and imaginary part
    //// Set grid and block size
    dim3 threadsPerBlock2(16, 16);
    dim3 blocksPerGrid2((4 * B + 15)/16, (M + 15)/16);

    //// call the accumulation kernel function (FX_re and FX_im will be zero-initialized inside)
    myAccumulate_transposed<<<blocksPerGrid2, threadsPerBlock2>>>(4, M, result1, result2, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during accumulation in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

/* 
 * Split every FP32 vector (unit) to two FP16 vectors
 * The size of each vector is given by n (expected to be 4)
 * The total number of vectors is M * B
 * M is in the vertical dimension, while B is in the horizontal dimension
 * X, Xhi, and Xlo is of size M * (n * B)
 * s1 and s2 is of size M * B
 * Grid and dim size should be 2D, total size is expected to be (B, M)
 * All data should be in unified memory or device memory
 * */
__global__ void mySplit_transposed(int n, int M, float* X, half* Xhi, half* Xlo, int B)
{
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // Vertical index of unit (max M)
    int blockNum = blockIdx.x * blockDim.x + threadIdx.x; // Horizontal index of unit (max B)

    if (rowIdx < M && blockNum < B){
        /* Data index to be manipulated:
         * X, Xhi, Xlo (rowIdx, blockIdx * n +0/1/2/3) = X, Xhi, Xlo[rowIdx + blockIdx * n * M + 0/1/2/3 * M]
         * s1, s2 (rowIdx, blockIdx) = s1, s2[rowIdx + blockIdx * M]
         */
        int offset = rowIdx + blockNum * n * M;
        int stride = M;

        for (int i = 0; i < n; i++){
            Xhi[offset + i * stride] = X[offset + i * stride]/s1;
            Xlo[offset + i * stride] = (X[offset + i * stride] - (float)Xhi[offset + i * stride] * s1) / s2;
        }
    }
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
__global__ void myAccumulate_transposed(int n, int M, float* X1, float* X2, float* R1, float* R2, int B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // vertical index of the element, max M
    int j = blockIdx.x * blockDim.x + threadIdx.x; // horizontal index of the element, max 4 * B

    if (i < M && j < 4 * B){
        int result_idx = i + j * M;
        int e_stride = M * 4 * B; // Stride for elements, e.g. from Re_hi to Re_lo
        R1[result_idx] = R2[result_idx] = 0.0f;

        R1[result_idx] += s1 * X1[result_idx];
        R1[result_idx] += s2 * X1[result_idx + e_stride];
        R1[result_idx] += -1.0f * s1 * X2[result_idx + 2*e_stride];
        R1[result_idx] += -1.0f * s2 * X2[result_idx + 3*e_stride];
        R2[result_idx] += s1 * X2[result_idx];
        R2[result_idx] += s2 * X2[result_idx + e_stride];
        R2[result_idx] += s1 * X1[result_idx + 2*e_stride];
        R2[result_idx] += s2 * X1[result_idx + 3*e_stride];
    }
}

#endif /* FFT_IMPROVED_GFFT_H */
