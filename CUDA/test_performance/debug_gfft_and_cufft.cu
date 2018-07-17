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
const int BATCH = 1;
const int SIZE = 4;
const int BLOCK_SIZE = 32;
const int DISPLAY_DATA = 1;
const int DEVICE = 0;

#define __START__ cudaEventCreate(&start); cudaEventCreate(&stop); cudaEventRecord(start, 0);
#define __STOP__(_V) cudaEventRecord(stop, 0); cudaEventSynchronize(stop); cudaEventElapsedTime(&duration, start, stop); _V.push_back(duration); cudaEventDestroy(start); cudaEventDestroy(stop);

float show_mean(std::vector<float> v)
{
    float sum = 0;
    for (int i = 0; i < v.size(); i++)
        sum += v[i];
    return sum / v.size(); 
}

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

int cuFFT16(int N, int B, Chalf* X, Chalf* FX){
    // Allocate unified momory for input and output
    int mem_size = N * B *sizeof(Chalf);
    Chalf *d_idata, *d_odata;
    checkCudaErrors(cudaMallocManaged((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &d_odata, mem_size));

    // Copy input data to memory
    if (DISPLAY_DATA == 1){
        printf("___fft16____input:\n");
        for (int j = 0; j < N * B; j++){
            printf("FX[%d] = (%.10f, %.10f) \n", j, (float)X[j].x, (float)X[j].y);
        }
    } 
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
                         CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, B, \
                         &workSize, CUDA_C_16F);
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

    if (DISPLAY_DATA == 1){
        printf("___fft16____output:\n");
        for (int j = 0; j < N * B; j++){
            printf("FX[%d] = (%.10f, %.10f) \n", j, (float)FX[j].x, (float)FX[j].y);
        }
    } 
    // Clean up content and memory
    cufftDestroy(plan);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    return 0;
}

int run_test_FP16(int input_size){
    printf("[cuFFT16] is starting...\n");
    
    // Initialize the memory for the input data
    int mem_size = input_size*sizeof(Chalf);
    Chalf *h_idata = (Chalf *)malloc(mem_size);
    for (unsigned int i = 0; i < input_size; i++) {
        h_idata[i].x = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
        h_idata[i].y = rand() / (0.5 * static_cast<float>(RAND_MAX)) - 1;
    }
    if (input_size == 4) {
        h_idata[0].x = 1.0f; h_idata[0].y = 2.0f;
        h_idata[1].x = 0.0f; h_idata[1].y = 0.0f; 
        h_idata[2].x = 0.0f; h_idata[2].y = 1.0f;
        h_idata[3].x = -1.0f; h_idata[3].y = 0.0f;
    }
    if (DISPLAY_DATA == 1) {
        printf("Input data: \n");
        for (unsigned int i = 0; i < input_size; i++) {
            printf("x[%d]=(%.2f, %.2f); \n", i, (float)h_idata[i].x, (float)h_idata[i].y);
        }
        printf("\n"); 
    }

    // Allocate device momory for input and output
    Chalf *d_idata, *d_odata;
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
                         CUDA_C_16F, NULL, 1, 1, CUDA_C_16F, 1, \
                         &workSize, CUDA_C_16F);
    if (result != CUFFT_SUCCESS)
    {
        printf("cufftXtMakePlanMany (plan) returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

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
    Chalf *h_odata = (Chalf *)malloc(mem_size);
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    // Print result
    if (DISPLAY_DATA == 1) {
        printf("FFT result: \n");
        for (unsigned int i = 0; i < input_size; i++) {
            printf("x[%d]=(%.2f, %.2f); \n", i, (float)h_odata[i].x, (float)h_odata[i].y);
        }
        printf("\n");
    }
    // Print the performance
    printf("Performance of cuFFT16: Problem size= %d, Time= %.5f msec\n", \
        input_size,
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
        printf("Usage: -norm=upper_bound (Max norm of input elements)"
               " -n=size (Input vector size)\n"
               " -batch=batch_size (Number of input vectors)\n"
               " -bs=block_size (Number of threads in a block)\n"
               " -display=show_result (0 or 1) \n" 
	       " -device=ID (ID >= 0 for deviceID)\n");
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
    
    //// Block size
    int bs = BLOCK_SIZE;
    if (checkCmdLineFlag(argc, (const char **)argv, "bs")) {
        bs = getCmdLineArgumentInt(argc, (const char **)argv, "bs");
    }
    
    //// Result display mode
    int display = DISPLAY_DATA;
    if (checkCmdLineFlag(argc, (const char **)argv, "display")) {
        display = getCmdLineArgumentInt(argc, (const char **)argv, "display");
    }
    
    //// Device ID by defualt is 0
    int device = DEVICE;
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


    // Start program
    printf("Problem size = %d, batch size = %d\n", n, batch);

    printf("[Testing of gfft and cuFFT] - Starting...\n");

    // Define event, result data structure
    cudaEvent_t start, stop;
    std::vector<float> cuFFT32Run, cuFFT16Run, gfftRun;
    std::vector<float> cuFFT16Error, gfftError;
    float duration, error1, error2;

    // Define input and output
    float X_re[n * batch], X_im[n * batch], FX_re[n * batch], FX_im[n * batch];
    Csingle X_32[n * batch], FX_32[n * batch];
    Chalf X_16[n * batch], FX_16[n * batch];
   
    // Run experiment
    for (int i = 0; i < 1; i++){
        // Initialize input
        srand(time(NULL));
        for (int j = 0; j < n * batch; j++){
            X_re[j] = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            X_im[j] = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            X_re[j] = (float)j;
            X_im[j] = 0.0f;
            X_32[j].x = X_re[j]; X_32[j].y = X_im[j];
            X_16[j].x = (half)X_re[j]; X_16[j].y = (half)X_im[j];
            if (display == 1){
                printf("X[%d] = (%.10f, %.10f) \n", j, X_re[j], X_im[j]);
            }
  
        }
        // Call cuFFT32
        __START__
        cuFFT32(n, batch, X_32, FX_32);
        __STOP__(cuFFT32Run)


        // Call cuFFT16
        __START__
        cuFFT16(n, batch, X_16, FX_16);
        __STOP__(cuFFT16Run)


        // Call gfft
        __START__
        gfft(n, batch, X_re, X_im, FX_re, FX_im);
        __STOP__(gfftRun)

        if (display == 1){
            printf("Result of cuFFT32:\n");
            for (int j = 0; j < n * batch; j++){
                printf("FX[%d] = (%.10f, %.10f) \n", j, FX_32[j].x, FX_32[j].y);
            }
            printf("Result of cuFFT16:\n");
            for (int j = 0; j < n * batch; j++){
                printf("FX[%d] = (%.10f, %.10f) \n", j, (float)FX_16[j].x, (float)FX_16[j].y);
            }
            printf("Result of gfft:\n");
            for (int j = 0; j < n * batch; j++){
                printf("FX[%d] = (%.10f, %.10f) \n", j, FX_re[j], FX_im[j]);
            }
        }

        // Calculate error
        for (int j = 0; j < n * batch; j++){
            error1 += (float)fabs((float)(FX_16[j].x) - FX_32[j].x);
            error1 += (float)fabs((float)(FX_16[j].y) - FX_32[j].y);
            error2 += (float)fabs(FX_re[j] - FX_32[j].x);
            error2 += (float)fabs(FX_im[j] - FX_32[j].y);
        }
        cuFFT16Error.push_back(error1 / (n * batch));
        gfftError.push_back(error2 / (n * batch));
    }

    printf("Time of cuFFT32: %f milliseconds\n", show_mean(cuFFT32Run)); 
    printf("Time of cuFFT16: %f milliseconds, error = %.10f\n", show_mean(cuFFT16Run), show_mean(cuFFT16Error)/norm); 
    printf("Time of gfft: %f milliseconds, error = %.10f\n", show_mean(gfftRun), show_mean(gfftError)/norm); 

    exit(0);
} 
