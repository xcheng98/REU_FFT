/* 
 * A program that compare performance of gfft and cuFFT library
 * Test the speed and accuracy of FP16 and FP32 calculation
 */

// C library, CUDA runtime, helpers, and utilities
#include "../util/my_include.h"
#include <vector>

// gfft
#include "../util/gfft_using_fft4.h"

// CUFFT
#include <cufft.h>
#include <cufftXt.h>

typedef half2 Chalf;
typedef float2 Csingle;

const float NORM = 1.0f;
const int BATCH = 16;
const int SIZE = 256;
const int DISPLAY_DATA = 0;


int cuFFT32(int N, int B, Csingle* X, Csingle* FX){
    // Allocate unified momory for input and output
    int mem_size = N * B *sizeof(Csingle);
    Csingle *d_idata, *d_odata;
    checkCudaErrors(cudaMallocManaged((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &d_odata, mem_size));

    // Copy input data to memory
    checkCudaErrors(cudaMemcpy(d_idata, X, mem_size, cudaMemcpyHostToDevice));

    // cuFFT plan
    cufftResult result;
    cufftHandle plan;
    size_t workSize;
    long long int input_size_long = N;
    result = cufftCreate(&plan);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftCreate (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 1, &input_size_long, NULL, 1, 1, \
                         CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, B, \
                         &workSize, CUDA_C_32F);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftXtMakePlanMany (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

    // cuFFT execution
    result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftExecC2C (execution) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Copy Device memory to output
    checkCudaErrors(cudaMemcpy(FX, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Clean up content and memory
    cufftDestroy(plan);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return 0;
}


int main(int argc, char **argv)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
            checkCmdLineFlag(argc, (const char **)argv, "?") ||
            checkCmdLineFlag(argc, (const char **)argv, "h")) {
        printf("Usage: -norm=upper_bound (Max norm of input elements)\n"
               " -n=size (Input vector size)\n"
               " -batch=batch_size (Number of input vectors)\n"
               " -display=show_result (0 or 1) \n"); 
        exit(EXIT_SUCCESS);
    }
 

    // Get and set parameter 
    //// Norm
    float norm = NORM;
    if (checkCmdLineFlag(argc, (const char **)argv, "norm")) {
        norm = getCmdLineArgumentInt(argc, (const char **)argv, "norm");
    }

    //// Input size
    int n = SIZE;
    if (checkCmdLineFlag(argc, (const char **)argv, "n")) {
        n = getCmdLineArgumentInt(argc, (const char **)argv, "n");
    }

    //// Batch size
    int batch = BATCH;
    if (checkCmdLineFlag(argc, (const char **)argv, "batch")) {
        batch = getCmdLineArgumentInt(argc, (const char **)argv, "batch");
    }
    
    //// Result display mode
    int display = DISPLAY_DATA;
    if (checkCmdLineFlag(argc, (const char **)argv, "display")) {
        display = getCmdLineArgumentInt(argc, (const char **)argv, "display");
    }
    
    // Start program
    printf("Problem size = %d, batch size = %d, norm = %f\n", n, batch, norm);

    printf("[Testing of cuFFT FP32] - Starting...\n");

    // Define input and output
    Csingle X_32[n * batch], FX_32[n * batch];
   
    // Run experiment
    for (int i = 0; i < 1; i++){
        // Initialize input
        srand(time(NULL));
        for (int j = 0; j < n * batch; j++){
            X_32[j].x = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            X_32[j].y = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            if (display == 1){
                printf("X[%d] = (%.10f, %.10f) \n", j, X_32[j].x, X_32[j].y);
            }
  
        }
        // Call cuFFT32
        cuFFT32(n, batch, X_32, FX_32);
    }

    exit(0);
}
