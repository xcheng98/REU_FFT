/* 
 * A program cuFFT testing
 * Test the speed and accuracy of FP16 and FP32 calculation
 */

// System includes
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuda_fp16.h>

// Helper
/* To process command line input */
# include "nvidia_helper/helper_string.h"

#define N 128
typedef half2 Chalf;
typedef float2 Csingle;

int run_test_FP32(int input_size){
    printf("[cuFFT32] is starting...\n");
    int mem_size = input_size*sizeof(Csingle);
    Csingle *h_idata = (Csingle *)malloc(mem_size);
    
    // Intialize the memory for the input data
    for (unsigned int i = 0; i < input_size; i++) {
        h_idata[i].x = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
        h_idata[i].y = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
    }
    h_idata[0].x = 1; h_idata[0].y = 2; h_idata[1].x = 0; h_idata[1].y = 0; 
    h_idata[2].x = 0; h_idata[2].y = 1; h_idata[3].x = -1; h_idata[3].y = 0;
    for (unsigned int i = 0; i < input_size; i++) {
        printf("x[%d]=(%f, %f); ", i, h_idata[i].x, h_idata[i].y);
    }
    printf("\n"); 

    // Allocate device momory for input and output
    Csingle *d_idata, *d_odata;
    cudaError_t error;
    error = cudaMalloc((void **) &d_idata, mem_size);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_idata returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
    error = cudaMalloc((void **) &d_odata, mem_size);
    if (error != cudaSuccess)
    {
        printf("cudaMalloc d_odata returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy host data to device
    error = cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (d_idata,h_idata) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // cuFFT plan
    cufftResult result;
    cufftHandle plan;
    size_t workSize;
    long long int input_size_long = input_size;
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftCreate (plan) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 1, &input_size_long, NULL, 1, 1, \
                         CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, 1, \
                         &workSize, CUDA_C_32F);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftXtMakePlanMany (plan) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("Temporary buffer size %li bytes\n", workSize);

    // cuFFT warm-up execution
    result = cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftExecC2C (plan) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Measure execution time
    cudaDeviceSynchronize();
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));


    // Copy Device memory to host
    Csingle *h_odata = (Csingle *)malloc(mem_size);
    error = cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess)
    {
        printf("cudaMemcpy (h_odata,d_odata) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Print result
    for (unsigned int i = 0; i < input_size; i++) {
        printf("x[%d]=(%f, %f); ", i, h_odata[i].x, h_odata[i].y);
    }
    printf("\n"); 

    // Clean up content and memory
    cufftDestroy(plan);
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);

    return 0;
}

int run_test_FP16(int input_size){
    printf("[cuFFT16] is starting...\n");
    Chalf *h_idata = (Chalf *)malloc(input_size*sizeof(Chalf));
    Chalf *d_idata, *d_odata;
    
    printf("size: %d\n", sizeof(Chalf));    
    return 0;
}

int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?")) {
        printf("Usage: -n=size (Input vector size) -device=ID (ID > 0 for deviceID)\n"); 
        exit(EXIT_SUCCESS);
    }
    
    // Set block size
    int block_size = 32;
 
    // Device ID by defualt is 0
    int devID = 0;
    if (checkCmdLineFlag(argc, (const char **)argv, "device")) {
        devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
        cudaSetDevice(devID);
    }
    cudaError_t error;
    cudaDeviceProp deviceProp;
    error = cudaGetDevice(&devID);
    if (error != cudaSuccess)
    {
        printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
    }
    error = cudaGetDeviceProperties(&deviceProp, devID);
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
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
    }

    // Input size by defualt is 8
    int n = block_size * 8;
    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }
    printf("Size = %d\n", n);
     

    printf("[Testing of cuFFT FP32 and FP16] - Starting...\n");
    
    int test32 = run_test_FP32(n);
    int test16 = run_test_FP16(n);

    exit(test32 || test16);
} 
