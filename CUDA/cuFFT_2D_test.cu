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
#include "nvidia_helper/helper_string.h"
/* To check cuda state */
#include "nvidia_helper/checkCudaErrors.h"

#define N 128
typedef half2 Chalf;
typedef float2 Csingle;

int DISPLAY_DATA = 1;

int run_test_1D(int input_size){
    printf("[cuFFT 1D] is starting...\n");
    int mem_size = input_size * input_size * sizeof(Csingle);
    Csingle *h_idata = (Csingle *)malloc(mem_size);
    
    // Intialize the memory for the input data
    for (unsigned int i = 0; i < input_size; i++) {    
        for (unsigned int j = 0; j < input_size; j++) {
            h_idata[i*input_size + j].x = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
            h_idata[i*input_size + j].y = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
        }
        if (input_size == 4) {
            h_idata[i*input_size + 0].x = 1; h_idata[i*input_size + 0].y = 2;
            h_idata[i*input_size + 1].x = 0; h_idata[i*input_size + 1].y = 0; 
            h_idata[i*input_size + 2].x = 0; h_idata[i*input_size + 2].y = 1;
            h_idata[i*input_size + 3].x = -1; h_idata[i*input_size + 3].y = 0;
        }
    }
    if (DISPLAY_DATA == 1) {
        printf("Input data: \n");
        for (unsigned int i = 0; i < input_size; i++) {    
            for (unsigned int j = 0; j < input_size; j++) {
                printf("x[%d, %d]=(%.2f, %.2f); \n", j, i, h_idata[i*input_size + j].x, h_idata[i*input_size + j].y);
            }
            printf("\n"); 
        }
    }
    // Allocate device momory for input and output
    Csingle *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // Copy host data to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // cuFFT plan
    cufftResult result;
    cufftHandle plan;
    size_t workSize;
    long long int input_size_long = input_size;
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftCreate (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 1, &input_size_long, NULL, 1, 1, \
                         CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, input_size, \
                         &workSize, CUDA_C_32F);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftXtMakePlanMany (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("Temporary buffer size %li bytes\n", workSize);

    // cuFFT warm-up execution
    result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftExecC2C (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Measure execution time
    cudaDeviceSynchronize();
    // Allocate CUDA events
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    // Repeatedly execute cuFFT
    int nIter = 300;
    for (int i = 0; i < nIter; i++){
        result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                              reinterpret_cast<cufftComplex *>(d_odata), \
                              CUFFT_FORWARD);
    }
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    // Calculate performance
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    float msecPerFFT = msecTotal / nIter;

    // Copy Device memory to host
    Csingle *h_odata = (Csingle *)malloc(mem_size);
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Print result
    if (DISPLAY_DATA == 1) {
        printf("FFT result: \n");
        for (unsigned int i = 0; i < input_size; i++) {
            for (unsigned int j = 0; j < input_size; j++) {
                printf("x[%d, %d]=(%.2f, %.2f); \n", j, i, h_odata[i*input_size + j].x, h_odata[i*input_size + j].y);
            }
            printf("\n");
        }
    }

    // Print the performance
    printf("Performance of cuFFT1D: Problem size= %d * %d, Time= %.5f msec\n", \
        input_size, input_size,
        msecPerFFT);

    // Clean up content and memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    cufftDestroy(plan);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
    free(h_idata);
    free(h_odata);

    return 0;
}

int run_test_2D(int input_size){
    printf("[cuFFT 2D] is starting...\n");
    int mem_size = input_size * input_size * sizeof(Csingle);
    Csingle *h_idata = (Csingle *)malloc(mem_size);
    
    // Intialize the memory for the input data
    for (unsigned int i = 0; i < input_size; i++) {    
        for (unsigned int j = 0; j < input_size; j++) {
            h_idata[i*input_size + j].x = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
            h_idata[i*input_size + j].y = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
        }
        if (input_size == 4) {
            h_idata[i*input_size + 0].x = 1; h_idata[i*input_size + 0].y = 2;
            h_idata[i*input_size + 1].x = 0; h_idata[i*input_size + 1].y = 0; 
            h_idata[i*input_size + 2].x = 0; h_idata[i*input_size + 2].y = 1;
            h_idata[i*input_size + 3].x = -1; h_idata[i*input_size + 3].y = 0;
        }
    }
    if (DISPLAY_DATA == 1) {
        printf("Input data: \n");
        for (unsigned int i = 0; i < input_size; i++) {    
            for (unsigned int j = 0; j < input_size; j++) {
                printf("x[%d, %d]=(%.2f, %.2f); \n", j, i, h_idata[i*input_size + j].x, h_idata[i*input_size + j].y);
            }
            printf("\n"); 
        }
    }
    // Allocate device momory for input and output
    Csingle *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

    // Copy host data to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // cuFFT plan
    cufftResult result;
    cufftHandle plan;
    size_t workSize;
    long long int input_size_long[2];
    result = cufftCreate(&plan);
    input_size_long[0] = input_size;
    input_size_long[1] = input_size;
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftCreate (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 2, input_size_long, NULL, 1, 1, \
                         CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, 1, \
                         &workSize, CUDA_C_32F);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftXtMakePlanMany (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    printf("Temporary buffer size %li bytes\n", workSize);

    // cuFFT warm-up execution
    result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftExecC2C (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Measure execution time
    cudaDeviceSynchronize();
    // Allocate CUDA events
    cudaEvent_t start;
    checkCudaErrors(cudaEventCreate(&start));
    cudaEvent_t stop;
    checkCudaErrors(cudaEventCreate(&stop));
    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));
    // Repeatedly execute cuFFT
    int nIter = 300;
    for (int i = 0; i < nIter; i++){
        result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                              reinterpret_cast<cufftComplex *>(d_odata), \
                              CUFFT_FORWARD);
    }
    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));
    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));
    // Calculate performance
    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
    float msecPerFFT = msecTotal / nIter;

    // Copy Device memory to host
    Csingle *h_odata = (Csingle *)malloc(mem_size);
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Print result
    if (DISPLAY_DATA == 1) {
        printf("FFT result: \n");
        for (unsigned int i = 0; i < input_size; i++) {
            for (unsigned int j = 0; j < input_size; j++) {
                printf("x[%d, %d]=(%.2f, %.2f); \n", j, i, h_odata[i*input_size + j].x, h_odata[i*input_size + j].y);
            }
            printf("\n");
        }
    }

    // Print the performance
    printf("Performance of cuFFT2D: Problem size= %d * %d, Time= %.5f msec\n", \
        input_size, input_size,
        msecPerFFT);

    // Clean up content and memory
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    cufftDestroy(plan);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));
    free(h_idata);
    free(h_odata);

    return 0;
}



int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?") ||
            checkCmdLineFlag(argc, (const char **)argv, "h")) {
        printf("Usage: -n=size (Input vector size)"
	       " -device=ID (ID > 0 for deviceID)"
               " -display=show_result (0 or 1) \n"); 
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
     
    // Set display mode
    if (checkCmdLineFlag(argc, (const char **)argv, "display")) {
        int entered_mode = getCmdLineArgumentInt(argc, (const char **)argv, "display");
        if (entered_mode == 0)  DISPLAY_DATA = 0;
    }

    printf("Problem size = %d * %d\n", n, n);

    printf("[Testing of cuFFT 1D and 2D] - Starting...\n");
    
    int test1D = run_test_1D(n);
    int test2D = run_test_2D(n);

    exit(test1D || test2D);
} 
