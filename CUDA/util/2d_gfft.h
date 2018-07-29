/*
 * Implementing the 2D FFT algorithm for general input
 * Input should be ONE fp32 matrix with both dimensions eqaul and are the power of 4
 * To be called by testing program
 * Calling 1D gfft
 */

#ifndef FFT_2D_GFFT_H
#define FFT_2D_GFFT_H

#include "improved_gfft_for_2D.h"

cublasStatus_t status;
cublasHandle_t handle;


int fft_2d(int M, int N, float* X_re, float* X_im, float* FX_re, float* FX_im, int BATCH = 1)
{
    FFT_S fft_status;

    // Allocate memory for input and output matrix
    float *input_re, *input_im, *output_re, *output_im;
    int mem_size = M * N * BATCH * sizeof(float);
    checkCudaErrors(cudaMalloc((void **) &input_re, mem_size));
    checkCudaErrors(cudaMalloc((void **) &input_im, mem_size));
    checkCudaErrors(cudaMalloc((void **) &output_re, mem_size));
    checkCudaErrors(cudaMalloc((void **) &output_im, mem_size));

    // Copy input matrix to device memory
    checkCudaErrors(cudaMemcpy(input_re, X_re, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(input_im, X_im, mem_size, cudaMemcpyHostToDevice));
    
    // Initialize memory for 1D gfft
    fft_status = gfft_init(M, N * BATCH);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! gFFT context initialization error.\n");
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

    float alpha = 1.0f, beta = 0.0f;

    // Call gfft function for the first time (Z = F * X)
    fft_status = gfft(M, input_re, input_im, output_re, output_im, N * BATCH);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! first gFFT execution error.\n");
        exit(1);
    }

    // Transpose the temporary result (transpose(Z))
    // Expect BATCH = 1
    // Reuse input_re & input_im as buffer
    for (int j = 0; j < BATCH; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, &alpha, output_re + j * M * N, M, 
                            &beta, output_re + j * M * N, M, input_re + j * M * N, N);
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, &alpha, output_im + j * M * N, M, 
                            &beta, output_im + j * M * N, M, input_im + j * M * N, N);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (first transpose).\n");
            return FFT_FAILURE;
        }
    }

    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Call gfft function for the second time (transpose(Y) = F * transpose(Z))
    // Use input_re & input_im as buffer
    fft_status = gfft(N, input_re, input_im, input_re, input_im, M * BATCH);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! second gFFT execution error.\n");
        exit(1);
    }

    // Transpose to get final result (Y) 
    // Expect BATCH = 1
    for (int j = 0; j < BATCH; j++){
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &alpha, input_re + j * M * N, N, 
                            &beta, input_re + j * M * N, N, output_re + j * M * N, M);
        status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, &alpha, input_im + j * M * N, N, 
                            &beta, input_im + j * M * N, N, output_im + j * M * N, M);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "!!!! CUBLAS kernel execution error (second transpose).\n");
            return FFT_FAILURE;
        }
    }

    // Wait for GPU to finish work
    cudaDeviceSynchronize();    

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS shutdown error.\n");
        exit(1);
    }

    // Destroy 1D gfft context
    fft_status = gfft_destroy();
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! gFFT context destroy error.\n");
        exit(1);
    }

    // Copy result from device to host
    mem_size = M * N * BATCH * sizeof(float);
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

#endif /* FFT_2D_GFFT_H */
