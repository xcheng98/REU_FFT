/* 
 * A program that compare acceleration of gemm, cufft32, cufft16
 */

// C includes
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <vector>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
// CUFFT
#include <cufft.h>
#include <cufftXt.h>

// nvidia helper
#include "../checkCudaErrors.h"
#include "../helper_string.h"

typedef half2 Chalf;
typedef float2 Csingle;

const float NORM = 1.0f;
const int BATCH = 16;
const int SIZE = 256;
const int ITERATION = 10;
const int DISPLAY_DATA = 0;
const int DEVICE = 0;

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&duration, start, stop); _V.push_back(duration); cudaEventDestroy(start); cudaEventDestroy(stop);


cudaEvent_t start, stop;
std::vector<float> cuFFT32Run, cuFFT16Run, gemmRun;
float duration;


float show_mean(std::vector<float> v)
{
    float sum = 0;
    for (int i = 0; i < v.size(); i++)
        sum += v[i];
    return sum / v.size(); 
}

int cuFFT32(int N, Csingle* X, Csingle* FX, int B){
    // Allocate unified momory for input and output
    int mem_size = N * N * B *sizeof(Csingle);
    Csingle *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // Copy input data to memory
    checkCudaErrors(cudaMemcpy(d_idata, X, mem_size, cudaMemcpyHostToDevice));

    // cuFFT plan
    cufftResult result;
    cufftHandle plan;
    size_t workSize;
    long long int input_size_long[2] = {N, N};
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS)
    {
        fprintf(stderr, "In cuFFT32: cufftCreate plan returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 2, input_size_long, NULL, 1, 1, \
                         CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, B, \
                         &workSize, CUDA_C_32F);
    if (result != CUFFT_SUCCESS)
    {
        printf("In cuFFT32: cufftXtMakePlanMany returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }


    __START__
    // cuFFT execution
    result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("In cuFFT32: cufftExecC2C (execution) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    __STOP__(cuFFT32Run)

    // Copy Device memory to output
    checkCudaErrors(cudaMemcpy(FX, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Clean up content and memory
    cufftDestroy(plan);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return 0;
}

int cuFFT16(int N, Chalf* X, Chalf* FX, int B){
    // Allocate unified momory for input and output
    int mem_size = N * N * B *sizeof(Chalf);
    Chalf *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // Copy input data to memory
    checkCudaErrors(cudaMemcpy(d_idata, X, mem_size, cudaMemcpyHostToDevice));

    // cuFFT plan
    cufftResult result;
    cufftHandle plan;
    size_t workSize;
    long long int input_size_long[2] = {N, N};
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftCreate (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 2, input_size_long, NULL, 1, 1, \
                         CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, B, \
                         &workSize, CUDA_C_16F);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftXtMakePlanMany (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

    __START__
    // cuFFT execution
    result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftExecC2C (execution) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    __STOP__(cuFFT16Run)

    // Copy Device memory to output
    checkCudaErrors(cudaMemcpy(FX, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Clean up content and memory
    cufftDestroy(plan);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return 0;
}

int gemm(int N, half* X, half* FX, int B){
    // Allocate unified momory for input and output
    int mem_size = N * N * B *sizeof(half);
    Chalf *d_idata, *d_idata2, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_idata2, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // Copy input data to memory
    checkCudaErrors(cudaMemcpy(d_idata, X, mem_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_idata2, X, mem_size, cudaMemcpyHostToDevice));

    // cublas
    cublasStatus_t status;
    cublasHandle_t handle;
    half alpha = 1.0f, beta = 0.0f;
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
    
    __START__
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        d_idata, CUDA_R_16F, N, d_idata2, CUDA_R_16F, N, &beta,
        d_odata, CUDA_R_16F, N, CUDA_R_16F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error .\n");
        exit(-1);
    }
    __STOP__(gemmRun)

    // Copy Device memory to output
    checkCudaErrors(cudaMemcpy(FX, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Clean up content and memory
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS shutdown error.\n");
        exit(1);
    }
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return 0;
}


int get_parameters(int argc, char **argv, int& help_info, float& norm, int& n, int& batch, int& iter, int& display, int& device){
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?") ||
            checkCmdLineFlag(argc, (const char **)argv, "h")) {
        printf("Usage: -norm=upper_bound (Max norm of input elements)\n"
               " -n=size (Input vector size)\n"
               " -batch=batch_size (Number of input vectors)\n"
               " -iter=iteration (Times of experiments)\n"
               " -display=show_result (0 or 1) \n" 
               " -device=ID (ID >= 0 for deviceID)\n");
        help_info = 1;
        return 0;
    }

    // Get and set parameter 
    if (checkCmdLineFlag(argc, (const char **)argv, "norm")) {
        norm = getCmdLineArgumentFloat(argc, (const char **)argv, "norm");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "batch")) {
        batch = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
    }
    
    if (checkCmdLineFlag(argc, (const char **)argv, "iter")) {
        iter = getCmdLineArgumentInt(argc, (const char **)argv, "iter");
    }
    
    if (checkCmdLineFlag(argc, (const char **)argv, "display")) {
        display = getCmdLineArgumentInt(argc, (const char **)argv, "display");
    }
    
    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        device = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(device);
    }

    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&device);
    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    error = cudaGetDeviceProperties(&deviceProp, device);
    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        exit(EXIT_SUCCESS);
    }
    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    else
    {
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", device, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    return 0;
}


int main(int argc, char **argv)
{
    int help_info = 0;
    float norm = NORM;
    int n = SIZE;
    int batch = BATCH;
    int iter = ITERATION;
    int display = DISPLAY_DATA;
    int device = DEVICE;

    get_parameters(argc, argv, help_info, norm, n, batch, iter, display, device);

    if (help_info == 1){
        exit(EXIT_SUCCESS);
    }

    // Start program
    printf("Problem size = %d, batch size = %d, norm = %f, iteration = %d\n", n, batch, norm, iter);

    printf("[Testing acceleration] - Starting...\n");

    // Define and zero initialize input and output
    half* X_re = new half[n * n * batch]();
    half* FX_re = new half[n * n * batch]();
    Csingle* X_32 = new Csingle[n * n * batch]();
    Csingle* FX_32 = new Csingle[n * n * batch]();
    Chalf* X_16 = new Chalf[n * n * batch]();
    Chalf* FX_16 = new Chalf[n * n * batch]();

    // Warm up
    cuFFT32(n, X_32, FX_32, batch);
    cuFFT16(n, X_16, FX_16, batch);
    gemm(n, X_re, FX_re, batch);

    cuFFT32Run.pop_back();
    cuFFT16Run.pop_back();
    gemmRun.pop_back();

    printf("Warm up completed, start experiments...\n");
 
    // Run experiment
    for (int i = 0; i < iter; i++){
        // Initialize input
        srand(time(NULL));
        for (int j = 0; j < n * batch; j++){
            X_re[j] = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            FX_re[j] = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            X_32[j].x = X_re[j]; X_32[j].y = FX_re[j];
            X_16[j].x = (half)X_re[j]; X_16[j].y = (half)FX_re[j];
        }

        cuFFT32(n, X_32, FX_32, batch);
        cuFFT16(n, X_16, FX_16, batch);
        gemm(n, X_re, FX_re, batch);
    }

    // Print experiment result
    printf("Time of cuFFT32: %f milliseconds\n", show_mean(cuFFT32Run)); 
    printf("Time of cuFFT16: %f milliseconds\n", show_mean(cuFFT16Run)); 
    printf("Time of gemm: %f milliseconds\n", show_mean(gemmRun)); 

    // Free input and output memory
    delete [] X_re;
    delete [] FX_re;
    delete [] X_32;
    delete [] FX_32;
    delete [] X_16;
    delete [] FX_16;

    exit(0);
} 
