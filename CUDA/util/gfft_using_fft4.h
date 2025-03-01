/*
 * Implementing the FFT algorithm for general input
 * Input should be fp32 vectors with size equals to the power of 4
 * Number of vectors is given by BATCH (B)
 * Recursive algorithm
 * Base case is fft4
 */

#ifndef FFT_GFFT_USING_FFT4_H
#define FFT_GFFT_USING_FFT4_H 

#include "my_include.h"

extern fft::MatrixH F4_re;
extern fft::MatrixH F4_im;

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

FFT_S gfft_recursion(int N, int B, fft::MatrixF& X_re, fft::MatrixF& X_im, fft::MatrixF& FX_re, fft::MatrixF& FX_im) 
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
    fft_status = gfft_recursion(N / 4, 4 * B, FX_re, FX_im, FX_re, FX_im);
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

FFT_S gfft(int SIZE, int BATCH, float* X_re, float* X_im, float* FX_re, float* FX_im) 
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

    // Copy input matrix from function parameter
    memcpy(input_re.array, (void*)X_re, SIZE * BATCH * sizeof(float));
    memcpy(input_im.array, (void*)X_im, SIZE * BATCH * sizeof(float));
    
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
    status = gfft_recursion(SIZE, BATCH, input_re, input_im, output_re, output_im);
    if (status != FFT_SUCCESS){
        printf("Error in running fft algorithm\n");
        return FFT_FAILURE;
    }

    
    // Copy result to output array
    memcpy(FX_re, (void*)(output_re.array), SIZE * BATCH * sizeof(float));
    memcpy(FX_im, (void*)(output_im.array), SIZE * BATCH * sizeof(float));
   
 
    checkCudaErrors(cudaFree(input_re.array));
    checkCudaErrors(cudaFree(input_im.array));
    checkCudaErrors(cudaFree(output_re.array));
    checkCudaErrors(cudaFree(output_im.array));

    return FFT_SUCCESS;
}


#endif /* FFT_GFFT_USING_FFT4_H */
