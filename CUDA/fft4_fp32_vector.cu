/*
 * Implementing fft4 algorithm
 * Input is one float32 vector
 * No spliting
 * It's not a complete FFT
 * To be used recursively by gfft
 */

// C includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas.h>

// Matrix and vector
#include <helper/my_vector.h>
#include <helper/my_matrix.h>
#include <helper/my_const.h>

float UPPER_BOUND = 1000.0f;


fft::MatrixF F4_re;
fft::MatrixF F4_im;

FFT_S init_F4()
{
    F4_re.width = 4;
    F4_re.height = 4;
    F4_re.array = (float*)malloc(F4_re.width * F4_re.height * sizeof(float));

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
    
    F4_im.width = 4;
    F4_im.height = 4;
    F4_im.array = (float*)malloc(F4_re.width * F4_re.height * sizeof(float));

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

FFT_S fft4(fft::VectorF X_re, fft::VectorF X_im, fft::VectorF FX_re, fft::VectorF FX_im) 
{
    cublasStatus_t status;
    cublasHandle_t handle;
    float* dev_FM, dev_input, dev_result1, dev_result2;
    float alpha = 1.0f, beta = 0.0f;

    // Initialize cublas and allocate device memory
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return FFT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_FM), 4 * 4 * sizeof(dev_FM[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate Fourier Matrix)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_input), 4 * 2 * sizeof(dev_input[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate Input Matrix)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_result1), 4 * 2 * sizeof(dev_result1[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate result 1 Matrix)\n");
        return EXIT_FAILURE;
    }
    if (cudaMalloc(reinterpret_cast<void **>(&dev_result2), 4 * 2 * sizeof(dev_result2[0])) !=
      cudaSuccess) {
        fprintf(stderr, "!!!! device memory allocation error (allocate result 2 Matrix)\n");
        return EXIT_FAILURE;
    }

    // F4_re * (X_re, X_im)
    //// Copy host data to device
    status = cublasSetVector(4 * 4, sizeof(F4_re.array[0]), F4_re.array, 1, dev_FM, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write F4_re)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(4, sizeof(X_re.array[0]), X_re.array, 1, dev_input, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write X_re)\n");
        return EXIT_FAILURE;
    }
    status = cublasSetVector(4, sizeof(X_im.array[0]), X_im.array, 1, dev_input + 4 * sizeof(dev_input[0]), 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write X_im)\n");
        return EXIT_FAILURE;
    }
    //// Call cublas gemm
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 2, 4, &alpha, dev_FM,
                       4, dev_input, 4, &beta, dev_result1, 4);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (a * (c, d)).\n");
        return FFT_FAILURE;
    }

    // F4_im * (X_re, X_im)
    //// Copy host data to device
    status = cublasSetVector(4 * 4, sizeof(F4_im.array[0]), F4_im.array, 1, dev_FM, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (write F4_im)\n");
        return EXIT_FAILURE;
    }
    //// Call cublas gemm
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, 2, 4, &alpha, dev_FM,
                       4, dev_input, 4, &beta, dev_result2, 4);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (b * (c, d)).\n");
        return FFT_FAILURE;
    }

    // Combine and get result, store in result1
    alpha = -1.0f;
    status = cublasSaxpy(handle, 4, result2 + 4 * sizeof(result2[0]), 1, result1, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (ac - bd).\n");
        return FFT_FAILURE;
    }
    alpha = 1.0f;
    status = cublasSaxpy(handle, 4, result2, 1, result1 + 4 * sizeof(result1[0]), 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (ad + bc).\n");
        return FFT_FAILURE;
    }

    // Copy device memory to host
    status = cublasGetVector(4, sizeof(FX_re.array[0]), dev_result1, 1, FX_re.array, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read FX_re)\n");
        return EXIT_FAILURE;
    }
    status = cublasGetVector(4, sizeof(FX_im.array[0]), dev_result1 + 4 * sizeof(dev_result1[0]), 1, FX_im.array, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! device access error (read FX_im)\n");
        return EXIT_FAILURE;
    }

    // Deallocate device memory and shutdown
    if (cudaFree(&dev_FM)) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free Fourier Matrix)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(&dev_input)) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free Input Matrix)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(&dev_result1)) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free result 1 Matrix)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(&dev_result2)) != cudaSuccess) {
        fprintf(stderr, "!!!! device memory free error (free result 2 Matrix)\n");
        return EXIT_FAILURE;
    }
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }

    return FFT_SUCCESS;
}

int main()
{
    FFT_S status;
    status = init_f4();
    if (status != FFT_SUCCESS){
        printf("Error in Fourier matrix initialization\n");
        exit(1);
    }

    fft::VectorF X_re;
    X_re.size = SIZE;
    X_re.array = (float*)malloc(X_re.size * sizeof(float));

    fft::VectorF X_im;
    X_im.size = SIZE;
    X_im.array = (float*)malloc(X_im.size * sizeof(float));

    fft::VectorF FX_re;
    FX_re.size = SIZE;
    FX_re.array = (float*)malloc(FX_re.size * sizeof(float));

    fft::VectorF FX_IM;
    FX_IM.size = SIZE;
    FX_IM.array = (float*)malloc(FX_IM.size * sizeof(float));

    // Setting input value
    srand(time(NULL));
    printf("The input is: \n");
    for (int i = 1; i <= 4; i++){
        X_re.element(i) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
        X_im.element(i) = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
        printf("X[%d] = (%.10f, %.10f) \n", i, X_re.element(i), X_im.element(i));
    }

    fft4(X_re,X_im, FX_re, FX_im);

    printf("Result: \n);
    for (int i = 0; i <= 4; i++){
        printf("FX[%d] = (%.10f, %.10f) \n", i, FX_re.element(i), FX_im.element(i));
    }


    free(X_re.array);
    free(X_im.array);
    free(FX_re.array);
    free(FX_im.array);
    return 0;
}
