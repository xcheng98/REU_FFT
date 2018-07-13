/*
 * Implementing the FFT algorithm for general input
 * Input should be fp32 vectors with size equals to the power of 4
 * Number of vectors is given by BATCH (B)
 * Recursive algorithm
 * Base case is fft4
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

// Utility programs
#include "util/debug_fp32_to_fp16.h"
#include "util/fourier_matrix_4.h"
#include "util/debug_fft4.h"

#define PI 3.14159265

const float UPPER_BOUND = 1.0f;
const int BATCH = 1;
const int SIZE = 16;

extern fft::MatrixH F4_re;
extern fft::MatrixH F4_im;
fft::MatrixF buffer_m1;
fft::MatrixF buffer_m2;

float* buffer;

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

    printf("_____calling gfft______: \n N=%d, B=%d\n", N, B);
    for (int j = 1; j <= B; j++){
        printf("Input vector %d: \n", j);
        for (int i = 1; i <= N; i++){
            printf("X[%d] = (%.10f, %.10f) \n", i, X_re.element(i, j), X_im.element(i, j));
        }
    }

    if (N == 4) {
        fft_status = fft4(B, X_re, X_im, FX_re, FX_im);
        return fft_status;
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

    // Reshape input and output matrix: (N -(Reshape)->4*(N/4)) * B
    X_re.width = X_re.width * N / 4; X_re.height = 4;
    X_im.width = X_im.width * N / 4; X_im.height = 4;
    FX_re.width = FX_re.width * N / 4; FX_re.height = 4;
    FX_im.width = FX_im.width * N / 4; FX_im.height = 4;

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

    //// Imaginary matrix
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

    cudaDeviceSynchronize();


    // Recursively call gfft function
    buffer_m1.width =4; buffer_m2.width = 4; buffer_m1.height = 4; buffer_m2.height = 4;

    fft_status = gfft(N / 4, 4 * B, FX_re, FX_im, buffer_m1, buffer_m2);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Execution error (recursively call gfft).\n");
        return FFT_FAILURE;
    }

    temp = FX_re.array; FX_re.array = buffer_m1.array; buffer_m1.array = temp;
    temp = FX_im.array; FX_im.array = buffer_m2.array; buffer_m2.array = temp;

    printf("_____After recursive______: \n");
    for (int j = 1; j <= B; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 0; i < N; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.array[i], FX_im.array[i]);
        }
    }


    // Multiplication with twiddle factors
    //// Set grid and block size
    dim3 threadsPerBlock(4, 16);
    dim3 numBlocks(1, (N + 63)/64); // Make sure blocks are enough

    //// Call kernel function
    for (int j = 0; j < B; j++){
        multiply_twiddle<<<numBlocks, threadsPerBlock>>>(N, N/4, 4, FX_re.array + j * N, FX_im.array + j * N);
    }

    printf("_____After combination______: \n");
    for (int j = 1; j <= B; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 0; i < N; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.array[i], FX_im.array[i]);
        }
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

    cudaDeviceSynchronize();

    printf("_____After Second Transpose______: \n");
    for (int j = 1; j <= B; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 0; i < N; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.array[i], FX_im.array[i]);
        }
    }


    // Call fft4
    buffer_m1.width =4; buffer_m2.width = 4; buffer_m1.height = 4; buffer_m2.height = 4;
    printf("Size: %d, %d, %d, %d, %d, %d, %d, %d\n", FX_re.height, FX_re.width, FX_im.height, FX_im.width, buffer_m1.height, buffer_m1.width, buffer_m2.height, buffer_m1.width);
    printf("_____Before final fft4______: \n");
    for (int j = 1; j <= B; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 1; i <= N; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.element(i, j), FX_im.element(i, j));
        }
    }
    fft_status = fft4(N / 4 * B, FX_re, FX_im, buffer_m1, buffer_m2);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Execution error (combine step calling fft4).\n");
        return FFT_FAILURE;
    }
    temp = FX_re.array; FX_re.array = buffer_m1.array; buffer_m1.array = temp;
    temp = FX_im.array; FX_im.array = buffer_m2.array; buffer_m2.array = temp;

    printf("_____After final fft4______: \n");
    for (int j = 1; j <= B; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 1; i <= N; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.element(i, j), FX_im.element(i, j));
        }
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

    cudaDeviceSynchronize();

    
    // Reshape back input and output matrix: (4*(N/4) --Reshape--> N) * B
    X_re.width = X_re.width * 4 / N; X_re.height = N;
    X_im.width = X_im.width * 4 / N; X_im.height = N;
    FX_re.width = FX_re.width * 4 / N; FX_re.height = N;
    FX_im.width = FX_im.width * 4 / N; FX_im.height = N;


    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

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
            input_re.element(i, j) = (float)1.0f;
            input_im.element(i, j) = (float)0.0f;
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

    
    buffer_m1.width = BATCH;
    buffer_m1.height = SIZE;
    mem_size = buffer_m1.width * buffer_m1.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(buffer_m1.array), mem_size));
    buffer_m2.width = BATCH;
    buffer_m2.height = SIZE;
    mem_size = buffer_m2.width * buffer_m2.height * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &(buffer_m2.array), mem_size));

    FFT_S status;
    
    status = init_F4();
    if (status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Matrix initialization error (init Fourier matrix).\n");
        return FFT_FAILURE;
    }

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
}
