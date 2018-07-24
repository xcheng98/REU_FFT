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
const int BATCH = 8;
const int SIZE = 1024;


// Utility function declaration
FFT_S init_F4();

__global__ void mySplit(float* X, half* Xhi, half* Xlo, float* s1, float* s2, int N, int B, float* Xtemp);

__global__ void myAccumulate(int N, float* X1, float* X2, float* alpha, float* R1, float* R2, int B);

FFT_S fft4(int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im);

__global__ void mySplit_transposed(float* X, half* Xhi, half* Xlo, float* s1, float* s2, int n, int M, int B, float* Xtemp);

__global__ void myAccumulate_transposed(float* X1, float* X2, float* alpha, float* R1, float* R2, int n, int M, int B);

FFT_S fft4_transposed(int M, int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im);

__global__ void myTranspose(int m, int n, float* input, float* output, int B);

__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im, int B);

FFT_S gfft(int N, int B, fft::MatrixF& X_re, fft::MatrixF& X_im, fft::MatrixF& FX_re, fft::MatrixF& FX_im);


// Global variables
fft::MatrixH F4_re;
fft::MatrixH F4_im;
float* buffer;
float* X_temp;


int main()
{
    int mem_size;

    // Set device heap size
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024 * 1024 * 64);

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
    checkCudaErrors(cudaMallocManaged((void **) &X_temp, mem_size));

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

FFT_S gfft(int N, int B, fft::MatrixF& X_re, fft::MatrixF& X_im, fft::MatrixF& FX_re, fft::MatrixF& FX_im) 
{
    if (N == 4) {
        return fft4(B, X_re, X_im, FX_re, FX_im);
    }

    // Status variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    FFT_S fft_status;
    cudaError_t cerror;

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
    // Store result directly in FX_re.array and FX_im.array
    //// Set grid and block size
    dim3 threadsPerBlock1(4, 16);
    dim3 blockPerGrid1(B, (N / 4 + 15)/16); // Make sure blocks are enough

    //// Real matrix
    myTranspose<<<blockPerGrid1, threadsPerBlock1>>>(4, N / 4, X_re.array, FX_re.array, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during first transposition of real matrix\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
    ////// Set dimension (Note that the transpose happens batch-wisely)
    FX_re.height = N / 4; FX_re.width = 4 * B;

    //// Imaginary matrix
    myTranspose<<<blockPerGrid1, threadsPerBlock1>>>(4, N / 4, X_im.array, FX_im.array, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during first transposition of imaginary matrix\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
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
    dim3 threadsPerBlock2(4, 16);
    dim3 blockPerGrid2(B, (N / 4 + 15)/16); // Make sure blocks are enough

    //// Call kernel function
    multiply_twiddle<<<blockPerGrid2, threadsPerBlock2>>>(N, N/4, 4, FX_re.array, FX_im.array, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during twiddle multiplication\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    cudaDeviceSynchronize();

    // Using the improved algorithm without transposition
    fft_status = fft4_transposed(N / 4, B, FX_re, FX_im, FX_re, FX_im);
    cudaDeviceSynchronize();

    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Execution error (calling fft4_transposed).\n");
        return FFT_FAILURE;
    } 

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

__global__ void myTranspose(int m, int n, float* input, float* output, int B)
{
    /* 
     * Transpose the B input matrices with size m * n
     * Every matrix in a batch is transposed independently
     * Input should be matrix of size m * (n * B)
     * Output should be matrix of size n * (m * B)
     * The grid size is expected to be B * 1
     * Used case: first transpose, from 4 * (N / 4) to (N / 4) * 4
     * */

    // Calculate position in the OUTPUT matrix (0 based)
    int j = threadIdx.x; // Column number within a matrix, expected to be 0, 1, 2, 3
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row number within a matrix
    int matrix_id = blockIdx.x;

    if (i < n && j < m && matrix_id < B){
        output[matrix_id * m * n + j * n + i] = input[matrix_id * m * n + i * m + j];
    }
}



FFT_S fft4(int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im) 
{
    // Variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaError_t cerror;

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
    int numThreads = 64;
    int numBlocks = (B + 63) / 64;
    mySplit<<<numBlocks, numThreads>>>(X_re.array, X_re_hi.array, X_re_lo.array, re_s1.array, re_s2.array, 4, B, X_temp);
    mySplit<<<numBlocks, numThreads>>>(X_im.array, X_im_hi.array, X_im_lo.array, im_s1.array, im_s2.array, 4, B, X_temp);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during splitting\n", cudaGetErrorString(cerror));
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


    // Scale, combine and get result, add to output
    //// Set grid and block size
    dim3 threadsPerBlock(16, 4);
    dim3 BlocksPerGrid((B+15)/16, 1);

    //// call kernel function (buffer is zero-initialized inside)
    myAccumulate<<<BlocksPerGrid, threadsPerBlock>>>(4, result1, result2, scales, FX_re.array, FX_im.array, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during accumulation\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
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


__global__ void mySplit(float* X, half* Xhi, half* Xlo, float* s1, float* s2, int N, int B, float* Xtemp)
{
 /* 
  * fft::MatrixF X (N*B), fft::MatrixH Xhi (N*B), fft::MatrixH Xlo (N*B)
  * fft::VectorF s1, fft::VectorF s2
  * int N, int B. N is always 4
  * Grid and dim size should be 1D, total size = B
  * All data should be in unified memory or device memory
  * */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B){
        // Calculate scaling factor 1
        float scale1 = 0.0f;
        for (int i = 0; i < N; i++){
            float norm = (float) fabs(X[i + idx * N]);
            if (norm > scale1) scale1 = norm;
        }
        
        // If all number are zero, skip
        if (scale1 == 0.0f){
            s1[idx] = 0.0f;
            s2[idx] = 0.0f;
            for (int i = 0; i < N; i++){
                Xhi[i + idx * N] = Xlo[i + idx * N] = 0.0f;
            }
        }
        else
        {
            // Restrict scale range
            if (scale1 < EPS) scale1 = EPS;
            if (scale1 > 1.0f/EPS) scale1 = 1.0f/EPS;
            s1[idx] = scale1;

            // Scale the high half
            for (int i = 0; i < N; i++){
                Xtemp[i + idx * N] = X[i + idx * N]/scale1;
                Xhi[i + idx * N] = (half)(Xtemp[i + idx * N]);
                // Use Xtemp to store the residual
                Xtemp[i + idx * N] = X[i + idx * N] - scale1 * (float)(Xhi[i + idx * N]);
            }

           // Calculate the lower scaling factor
            float scale2 = 0.0f;
            for (int i = 0; i < N; i++){
                float norm = (float) fabs(Xtemp[i + idx * N]);
                if (norm > scale2) scale2 = norm;
            }
        
            // If all number are zero, skip
            if (scale2 == 0.0f){
                s2[idx] = 0.0f;
                for (int i = 0; i < N; i++){
                    Xlo[i + idx * N] = 0.0f;
                }
            }
            else
            {
                // Restrict scale range
                if (scale2 < EPS) scale2 = EPS;
                if (scale2 > 1.0f/EPS) scale2 = 1.0f/EPS;
                s2[idx] = scale2;

                for (int i = 0; i < N; i++){
                Xlo[i + idx * N] = (half) (Xtemp[i + idx * N] / scale2);
                }
            }
        }
    }
}


__global__ void myAccumulate(int N, float* X1, float* X2, float* alpha, float* R1, float* R2, int B)
{
    /* 
     * N is number of elements in one column (expected to be 4)
     * X1, X2 are 4 * (B * 4) column-major matrix. Inner order is by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
     * alpha is B * 4 array. Inner order is by batch. Outer order is re_s1, re_s2, im_s1, im_s2
     * R1, R2 are 4 * B matrix
     * B is batch size
     * */
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < N && j < B){
        R1[i + j * N] = R2[i + j * N] = 0.0f;
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


__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im, int B)
{
    /* 
     * Multifly every element of the input matrix with twiddle factor
     * Every matrix in a batch is scaled independently
     * Block and thread layout should be 2D
     * Re.element(i, j) [0 based] = xre * cos(2pi/N * i * j) + xim * sin(2pi/N * i * j)
     * Im.element(i, j) [0 based] = -xre * sin(2pi/N * i * j) + xim * cos(2pi/N * i * j)
     * */

    // Calculate position (0 based)
    int j = threadIdx.x; // Column number within a matrix, 0 to 3 in radix 4
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row number within a matrix
    int matrix_id = blockIdx.x;

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


FFT_S fft4_transposed(int M, int B, fft::MatrixF X_re, fft::MatrixF X_im, fft::MatrixF FX_re, fft::MatrixF FX_im) 
{
    /* 
     * Perform fft4 assuming the input is in the transposed layout
     * M is the number of rows
     * 4 * B is the number of columns
     * Note that the fourier matrix is symmetric
     */

    // Variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaError_t cerror;

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
    checkCudaErrors(cudaMallocManaged((void **) &scales, M * B * 4 * sizeof(float)));
    checkCudaErrors(cudaMemset(scales, 0.0f, M * B * 4 * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **) &X_split, M * 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMemset(X_split, 0.0f, M * 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &result1, M * 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMemset(result1, 0.0f, M * 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMallocManaged((void **) &result2, M * 4 * B * 4 * sizeof(result2[0])));
    checkCudaErrors(cudaMemset(result2, 0.0f, M * 4 * B * 4 * sizeof(result2[0])));

    // Split input
    //// Initialize Matrix and Vector data structure to store split result
    fft::MatrixH X_re_hi;
    X_re_hi.width = 4 * B;
    X_re_hi.height = M;
    X_re_hi.array = X_split + M * 4 * B * 0;

    fft::MatrixH X_re_lo;
    X_re_lo.width = 4 * B;
    X_re_lo.height = M;
    X_re_lo.array = X_split + M * 4 * B * 1;

    fft::MatrixH X_im_hi;
    X_im_hi.width = 4 * B;
    X_im_hi.height = M;
    X_im_hi.array = X_split + M * 4 * B * 2;

    fft::MatrixH X_im_lo;
    X_im_lo.width = 4 * B;
    X_im_lo.height = M;
    X_im_lo.array = X_split + M * 4 * B * 3;

    fft::VectorF re_s1;
    re_s1.size = M * B;
    re_s1.array = scales + M * B * 0;

    fft::VectorF re_s2;
    re_s2.size = M * B;
    re_s2.array = scales + M * B * 1;

    fft::VectorF im_s1;
    im_s1.size = M * B;
    im_s1.array = scales + M * B * 2;

    fft::VectorF im_s2;
    im_s2.size = M * B;
    im_s2.array = scales + M * B * 3;

    //// Call splitting function
    dim3 threadsPerBlock1(4, 16);
    dim3 BlocksPerGrid1((B + 3)/4, (M + 15)/16);
    mySplit_transposed<<<BlocksPerGrid1, threadsPerBlock1>>>(X_re.array, X_re_hi.array, X_re_lo.array, re_s1.array, re_s2.array, 4, M, B, X_temp);
    mySplit_transposed<<<BlocksPerGrid1, threadsPerBlock1>>>(X_im.array, X_im_hi.array, X_im_lo.array, im_s1.array, im_s2.array, 4, M, B, X_temp);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during splitting in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
  
 
    // Call cublas function and finish Matrix multiplication calculation
    // The order of multiplicands are reversed
    //// Define batched offset
    long long int stride = M * 4;

    //// Call cublas batched gemm on F4_re

    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, X_split,
                        CUDA_R_16F, M, stride, F4_re.array, CUDA_R_16F, 4, 0, &beta, result1, CUDA_R_32F, M, stride, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error in fft4_transposed ((c, d) * a).\n");
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, X_split,
                        CUDA_R_16F, M, stride, F4_im.array, CUDA_R_16F, 4, 0, &beta, result2, CUDA_R_32F, M, stride, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error in fft4_transposed ((c, d) * b).\n");
        return FFT_FAILURE;
    }


    // Scale, combine and get result, add to output
    //// Set grid and block size
    dim3 threadsPerBlock2(16, 16);
    dim3 BlocksPerGrid2((4 * B + 15)/16, (M + 15)/16);

    //// call kernel function (buffer is zero-initialized inside)
    myAccumulate_transposed<<<BlocksPerGrid2, threadsPerBlock2>>>(result1, result2, scales, FX_re.array, FX_im.array, 4, M, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during accumulation in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
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

    return FFT_SUCCESS;
}


__global__ void mySplit_transposed(float* X, half* Xhi, half* Xlo, float* s1, float* s2, int n, int M, int B, float* Xtemp)
{
 /* 
  * fft::MatrixF X (M * (n * B)), fft::MatrixH Xhi (M * (n * B)), fft::MatrixH Xlo (M * (n * B))
  * fft::VectorF s1 of size M * B, fft::VectorF s2 of size M * B
  * int n, int M, int B. n is expected to be 4, M = N / 4
  * Grid and dim size should be 2D, total size = M * B
  * All data should be in unified memory or device memory
  * */
    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // Row number (max M)
    int blockNum = blockIdx.x * blockDim.x + threadIdx.x; // 'Column number' (max B)

    if (rowIdx < M && blockNum < B){
        /* Data to be manipulated:
         *  X, Xhi, Xlo (rowIdx, blockIdx * n +0+1+2+3) = X, Xhi, Xlo[rowIdx + blockIdx * n * M + 0/1/2/3 * M]
         * s1, s2 (rowIdx, blockIdx) = s1, s2[rowIdx + blockIdx * M]
         */
        int offset = rowIdx + blockNum * n * M;
        int stride = M;
        int factor_idx = rowIdx + blockNum * M;

        // Calculate scaling factor 1
        float scale1 = 0.0f;
        for (int i = 0; i < n; i++){
            float norm = (float) fabs(X[offset + i * stride]);
            if (norm > scale1) scale1 = norm;
        }
        
        // If all number are zero, skip
        if (scale1 == 0.0f){
            s1[factor_idx] = 0.0f;
            s2[factor_idx] = 0.0f;
            for (int i = 0; i < n; i++){
                Xhi[offset + i * stride] = Xlo[offset + i * stride] = 0.0f;
            }
        }
        else
        {
            // Restrict scale range
            if (scale1 < EPS) scale1 = EPS;
            if (scale1 > 1.0f/EPS) scale1 = 1.0f/EPS;
            s1[factor_idx] = scale1;

            // Scale the high half
            for (int i = 0; i < n; i++){
                Xtemp[offset + i * stride] = X[offset + i * stride]/scale1;
                Xhi[offset + i * stride] = (half)(Xtemp[offset + i * stride]);
                // Use Xtemp to store the residual
                Xtemp[offset + i * stride] = X[offset + i * stride] - scale1 * (float)(Xhi[offset + i * stride]);
            }

           // Calculate the lower scaling factor
            float scale2 = 0.0f;
            for (int i = 0; i < n; i++){
                float norm = (float) fabs(Xtemp[offset + i * stride]);
                if (norm > scale2) scale2 = norm;
            }
        
            // If all number are zero, skip
            if (scale2 == 0.0f){
                s2[factor_idx] = 0.0f;
                for (int i = 0; i < n; i++){
                    Xlo[offset + i * stride] = 0.0f;
                }
            }
            else
            {
                // Restrict scale range
                if (scale2 < EPS) scale2 = EPS;
                if (scale2 > 1.0f/EPS) scale2 = 1.0f/EPS;
                s2[factor_idx] = scale2;

                for (int i = 0; i < n; i++){
                Xlo[offset + i * stride] = (half) (Xtemp[offset + i * stride] / scale2);
                }
            }
        }
    }
}


__global__ void myAccumulate_transposed(float* X1, float* X2, float* alpha, float* R1, float* R2, int n, int M, int B)
{
    /* 
     * X1, X2 are M * (4 * B * 4) matrix. The inner-most column order is by element in a unit. Then by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
     * alpha is a M * B * 4 array. Inner most order is by rows. Then by batch. Outer order is re_s1, re_s2, im_s1, im_s2
     * R1, R2 are M * (4 * B) matrix
     * n is number of elements in one unit (expected to be 4)
     * M is number of rows, B is batch size
     * */
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < M && j < 4 * B){
        int result_idx = i + j * M;
        int e_stride = M * 4 * B; // Stride for elements, e.g. from Re_hi to Re_lo
        int factor_idx = i + j / 4 * M;
        int f_stride = M * B; // Stride for factors, e.g. from re_s1 to re_s2
        R1[result_idx] = R2[result_idx] = 0.0f;

        R1[result_idx] += alpha[factor_idx] * X1[result_idx];
        R1[result_idx] += alpha[factor_idx + f_stride] * X1[result_idx + e_stride];
        R1[result_idx] += -1.0f * alpha[factor_idx + 2*f_stride] * X2[result_idx + 2*e_stride];
        R1[result_idx] += -1.0f * alpha[factor_idx + 3*f_stride] * X2[result_idx + 3*e_stride];
        R2[result_idx] += alpha[factor_idx] * X2[result_idx];
        R2[result_idx] += alpha[factor_idx + f_stride] * X2[result_idx + e_stride];
        R2[result_idx] += alpha[factor_idx + 2*f_stride] * X1[result_idx + 2*e_stride];
        R2[result_idx] += alpha[factor_idx + 3*f_stride] * X1[result_idx + 3*e_stride];
    }
}
