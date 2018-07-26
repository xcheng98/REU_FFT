/* 
 * Somehow this file does not work for gfft without splitting
 * A program that compare performance of the optimized gfft and cuFFT library
 * Test the speed and accuracy of FP16 and FP32 calculation
 * Try to avoid the impact of device warming up
 */

// C library, CUDA runtime, helpers, and utilities
#include "../util/my_include_combined.h"
#include <vector>

// gfft
#include "../alternative/32_gfft_magma.h"
#include "../alternative/improved_gfft_magma.h"

// CUFFT
#include <cufft.h>
#include <cufftXt.h>

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

float show_mean(std::vector<float> v)
{
    float sum = 0;
    for (int i = 0; i < v.size(); i++)
        sum += v[i];
    return sum / v.size(); 
}

int cuFFT32(int N, Csingle* X, Csingle* FX, int B){
    // Allocate unified momory for input and output
    int mem_size = N * B *sizeof(Csingle);
    Csingle *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

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
        fprintf(stderr, "In cuFFT32: cufftCreate plan returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }
    result = cufftXtMakePlanMany(plan, 1, &input_size_long, NULL, 1, 1, \
                         CUDA_C_32F, NULL, 1, 1, CUDA_C_32F, B, \
                         &workSize, CUDA_C_32F);
    if (result != CUFFT_SUCCESS)
    {
        printf("In cuFFT32: cufftXtMakePlanMany returned error code %d, line(%d)\n", result, __LINE__);
        exit(EXIT_FAILURE);
    }

    // cuFFT execution
    result = cufftXtExec(plan, reinterpret_cast<cufftComplex *>(d_idata), \
                          reinterpret_cast<cufftComplex *>(d_odata), \
                          CUFFT_FORWARD);
    if (result != CUFFT_SUCCESS)
    {
        printf("In cuFFT32: cufftExecC2C (execution) returned error code %d, line(%d)\n", result, __LINE__);
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

int cuFFT16(int N, Chalf* X, Chalf* FX, int B){
    // Allocate unified momory for input and output
    int mem_size = N * B *sizeof(Chalf);
    Chalf *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));

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

    // Clean up content and memory
    cufftDestroy(plan);
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

    printf("[Testing of gfft and cuFFT] - Starting...\n");

    // Define error, event, result data structure
    cudaEvent_t start, stop;
    std::vector<float> cuFFT32Run, cuFFT16Run, gfft32Run, gfftRun;
    std::vector<float> cuFFT16Error, gfft32Error, gfftError;
    float duration, error1, error2, error3;

    // Define and zero initialize input and output
    float* X_re = new float[n * batch]();
    float* X_im = new float[n * batch]();
    float* FX_re = new float[n * batch]();
    float* FX_im = new float [n * batch]();
    float* FX_re_32 = new float[n * batch]();
    float* FX_im_32 = new float [n * batch]();
    Csingle* X_32 = new Csingle[n * batch]();
    Csingle* FX_32 = new Csingle[n * batch]();
    Chalf* X_16 = new Chalf[n * batch]();
    Chalf* FX_16 = new Chalf[n * batch]();

    // Warm up
    cuFFT32(n, X_32, FX_32, batch);
    cuFFT16(n, X_16, FX_16, batch);
    // gfft_32(n, X_re, X_im, FX_re_32, FX_im_32, batch);
    gfft(n, X_re, X_im, FX_re, FX_im, batch);
  
 
    // Run experiment
    for (int i = 0; i < iter; i++){
        // Initialize input
        srand(time(NULL));
        for (int j = 0; j < n * batch; j++){
            X_re[j] = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            X_im[j] = (float)rand() / (float)(RAND_MAX) * 2 * norm - norm;
            X_32[j].x = X_re[j]; X_32[j].y = X_im[j];
            X_16[j].x = (half)X_re[j]; X_16[j].y = (half)X_im[j];
            if (display == 1){
                printf("X[%d] = (%.10f, %.10f) \n", j, X_re[j], X_im[j]);
            }
  
        }

        // Call cuFFT32
        __START__
        cuFFT32(n, X_32, FX_32, batch);
        __STOP__(cuFFT32Run)

        // Call cuFFT16
        __START__
        cuFFT16(n, X_16, FX_16, batch);
        __STOP__(cuFFT16Run)

        // Call gfft32
       // __START__
       // gfft_32(n, X_re, X_im, FX_re_32, FX_im_32, batch);
       // __STOP__(gfft32Run)

        // Call gfft
        __START__
        gfft(n, X_re, X_im, FX_re, FX_im, batch);
        __STOP__(gfftRun)

        // Calculate error
        for (int j = 0; j < n * batch; j++){
            error1 += (float)fabs((float)(FX_16[j].x) - FX_32[j].x);
            error1 += (float)fabs((float)(FX_16[j].y) - FX_32[j].y);
            error2 += (float)fabs(FX_re[j] - FX_32[j].x);
            error2 += (float)fabs(FX_im[j] - FX_32[j].y);
            error3 += (float)fabs(FX_re_32[j] - FX_32[j].x);
            error3 += (float)fabs(FX_im_32[j] - FX_32[j].y);
        }
        cuFFT16Error.push_back(error1 / (n * batch));
        gfftError.push_back(error2 / (n * batch));
        gfft32Error.push_back(error3 / (n * batch));
    }

    // Print experiment result
    printf("Time of cuFFT32: %f milliseconds\n", show_mean(cuFFT32Run)); 
    printf("Time of cuFFT16: %f milliseconds, error = %.10f\n", show_mean(cuFFT16Run), show_mean(cuFFT16Error)/norm); 
    printf("Time of gfft32: %f milliseconds, error = %.10f\n", show_mean(gfft32Run), show_mean(gfft32Error)/norm); 
    printf("Time of gfft: %f milliseconds, error = %.10f\n", show_mean(gfftRun), show_mean(gfftError)/norm); 

    // Free input and output memory
    delete [] X_re;
    delete [] X_im;
    delete [] FX_re;
    delete [] FX_im;
    delete [] FX_re_32;
    delete [] FX_im_32;
    delete [] X_32;
    delete [] FX_32;
    delete [] X_16;
    delete [] FX_16;

    exit(0);
} 
