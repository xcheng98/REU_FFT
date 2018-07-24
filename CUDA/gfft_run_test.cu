/*
 * Implementing the FFT algorithm for general input
 * Input should be fp32 vectors with size equals to the power of 4
 * Number of vectors is given by BATCH (B)
 * Recursive algorithm, base case is fft4
 * Combine all components in one file
 * Version after multiple optimizations
 * This implementation is without matrix and vector
 */

#include "my_include_combined.h"


#define PI 3.14159265
#define EPS 0.0000001192f


const float UPPER_BOUND = 1.0f;
const int BATCH = 16;
const int SIZE = 256;


FFT_S gfft(int N, float* X_re, float* X_im, float* FX_re, float* FX_im, int B);
 
__global__ void myTranspose(int m, int n, float* input, float* output, int B);

__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im, int B);

FFT_S init_F4();

FFT_S fft4(float* X_re, float* X_im, float* FX_re, float* FX_im, int B);

__global__ void mySplit(int N, float* X, half* Xhi, half* Xlo, float* s1, float* s2, int B, float* Xtemp);

__global__ void myAccumulate(int N, float* X1, float* X2, float* alpha, float* R1, float* R2, int B);

FFT_S fft4_transposed(int M, float* X_re, float* X_im, float* FX_re, float* FX_im, int B);

__global__ void mySplit_transposed(int n, int M, float* X, half* Xhi, half* Xlo, float* s1, float* s2, int B, float* Xtemp);

__global__ void myAccumulate_transposed(int n, int M, float* X1, float* X2, float* alpha, float* R1, float* R2, int B);


half* F4_re;
half* F4_im;
float* buffer;


int main()
{
    int mem_size;
    FFT_S status;

    // Allocate unified memory for input and output matrix
    float* input_re, input_im, output_re, output_im;
    mem_size = BATCH * SIZE * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &input_re, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &input_im, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &output_re, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &output_im, mem_size));

    // Initialize the input data
    srand(time(NULL));
    printf("The input is: \n");
    for (int j = 0; j < BATCH; j++){
        printf("Vector %d: \n", j);
        for (int i = 0; i < SIZE; i++){
            input_re[i + j * SIZE] = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            input_im[i + j * SIZE] = (float)rand() / (float)(RAND_MAX) * 2 * UPPER_BOUND - UPPER_BOUND;
            input_re[i + j * SIZE] = (float)i;
            input_im[i + j * SIZE] = 0.0f;
            printf("X[%d] = (%.10f, %.10f) \n", i, input_re[i + j * SIZE], input_re[i + j * SIZE]);
        }
        printf("\n");
    }
    
    // Allocate unified memory for the buffer (array of float)
    mem_size = SIZE * BATCH * sizeof(float);
    checkCudaErrors(cudaMallocManaged((void **) &buffer, mem_size));

    // Initialize Fourier matrix
    status = init_F4();
    if (status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Matrix initialization error (Fourier matrix).\n");
        exit(1);
    }

    // Call gfft function
    status = gfft(SIZE, input_re, input_im, output_re, output_im, BATCH);
    if (status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! gFFT execution error.\n");
        exit(1);
    }

    // Print result
    printf("Result: \n");
    for (int j = 0; j < BATCH; j++){
        printf("Resulting vector %d: \n", j);
        for (int i = 0; i < SIZE; i++){
            printf("FX[%d] = (%.10f, %.10f) \n", i, output_re[i + j * SIZE], output_im[i + j * SIZE]);
        }
    }

    // Deallocate unified memory
    checkCudaErrors(cudaFree(input_re));
    checkCudaErrors(cudaFree(input_im));
    checkCudaErrors(cudaFree(output_re));
    checkCudaErrors(cudaFree(output_im));

    exit(0);
}


FFT_S gfft(int N, float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{
    // Base case
    if (N == 4) {
        return fft4(X_re, X_im, FX_re, FX_im, B);
    }

    // Status variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    FFT_S fft_status;
    cudaError_t cerror;

    // Declare temp variable for buffer swapping
    float* temp;

    // Initialize cublas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS initialization error.\n");
        return FFT_FAILURE;
    }

    // Transpose input matrix: 4 * (N/4*B) --> (N/4) * (4*B)
    // First store the result in buffer to avoid racing condition
    //// Set grid and block size
    dim3 threadsPerBlock1(4, 16);
    dim3 blockPerGrid1(B, (N / 4 + 15)/16); // Make sure blocks are enough

    //// Transpose real matrix
    myTranspose<<<blockPerGrid1, threadsPerBlock1>>>(4, N / 4, X_re, buffer, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during transposition of real matrix.\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    //// Swap FX_re and buffer to store the transposition result in FX_re
    temp = FX_re; FX_re = buffer; buffer = temp;

    //// Transpose imaginary matrix
    myTranspose<<<blockPerGrid1, threadsPerBlock1>>>(4, N / 4, X_im, buffer, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during transposition of imaginary matrix.\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
    ////// Swap FX_im and buffer to store the transposition result in FX_im
    temp = FX_im; FX_im = buffer; buffer = temp;

    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Recursively call gfft function, not! using buffer matrix
    fft_status = gfft(N / 4, 4 * B, FX_re, FX_im, FX_re, FX_im);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Function execution error (recursively call gfft).\n");
        return FFT_FAILURE;
    }

    // Wait for GPU to finish work
    cudaDeviceSynchronize();

    // Multiplicate each element with twiddle factor
    //// Set grid and block size
    dim3 threadsPerBlock2(4, 16);
    dim3 blockPerGrid2(B, (N / 4 + 15)/16); // Make sure blocks are enough

    //// Call kernel function
    multiply_twiddle<<<blockPerGrid2, threadsPerBlock2>>>(N, N/4, 4, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during twiddle factor multiplication.\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    // Call the optimized fft4 function to avoid transposition
    fft_status = fft4_transposed(N / 4, FX_re, FX_im, FX_re, FX_im, B);
    if (fft_status != FFT_SUCCESS){
        fprintf(stderr, "!!!!! Function execution error (calling fft4_transposed).\n");
        return FFT_FAILURE;
    } 

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! shutdown error.\n");
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

/* 
 * Transpose every input matrix of size m * n
 * Number of matrices is given by B 
 * Every matrix in a batch is transposed independently
 * Input is expected to be matrix of size m * (n * B)
 * Output is expected to be matrix of size n * (m * B)
 * The grid size is expected to be B in horizontal dimension
 * Usage: transpose a matrix of size 4 * (N/4 * B) to (N/4) * (4 * B)
 * */
__global__ void myTranspose(int m, int n, float* input, float* output, int B)
{
    // Calculate position in the OUTPUT matrix (0 based)
    int j = threadIdx.x; // Column number within a matrix, expected to be 0, 1, 2, 3
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row number
    int matrix_id = blockIdx.x; // The index of matrix in the batch

    if (i < n && j < m && matrix_id < B){
        output[matrix_id * m * n + j * n + i] = input[matrix_id * m * n + i * m + j];
    }
}

/* 
 * Multifly every element of the input matrix with the twiddle factor
 * Every matrix in a batch is processed independently
 * Block and thread layout should be 2D, and the total dimension is expected to be (m, n * B)
 * n is expected to be 4
 * result.re(i, j) [0 based] = xre(i, j) * cos(2pi/N * i * j) + xim(i, j) * sin(2pi/N * i * j)
 * result.im(i, j) [0 based] = -xre(i, j) * sin(2pi/N * i * j) + xim(i, j) * cos(2pi/N * i * j)
 * ONLY that thread will access the particular matrix_re and matrix_im, so buffer is not needed
 * */
__global__ void multiply_twiddle(int N, int m, int n, float* matrix_re, float* matrix_im, int B)
{
    // Calculate position
    int j = threadIdx.x; // Column number within a matrix, 0 to 3 in radix 4
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row number within a matrix
    int matrix_id = blockIdx.x; // Index of matrix in the batch

    if (i < m && j < n && matrix_id < B){
        // Per-thread local variables
        int index = matrix_id * N + j * m + i;
        float tw_re = cos(2 * PI / N * i * j);
        float tw_im = sin(2 * PI / N * i * j);
        float result_re = matrix_re[index] * tw_re + matrix_im[index] * tw_im;
        float result_im = -1.0f * matrix_re[index] * tw_im + matrix_im[index] * tw_re;

        matrix_re[index] = result_re;
        matrix_im[index] = result_im;
    }
}


/* 
 * Perform fft on every length-4 vector
 * Batch size is given by B
 * Internally split every FP32 input into two FP16 vectors
 * Combine them together after FFT
 * */
FFT_S fft4(float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{
    // Variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaError_t cerror;
    float alpha = 1.0f, beta = 0.0f; 
    float* scales; // = re_s1, re_s2, im_s1, im_s2;
    half* X_split; // = X_re_hi, X_re_lo, X_im_hi, X_im_lo;
    float *result1, *result2; // F4_re * X_split, F4_im * X_split

    // Initialize cublas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS initialization error\n");
        return FFT_FAILURE;
    }
    status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // allow Tensor Core
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS setting math mode error\n");
        return FFT_FAILURE;
    }

    //  Allocate unified memory
    checkCudaErrors(cudaMallocManaged((void **) &scales, B * 4 * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **) &X_split, 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &result1, 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMallocManaged((void **) &result2, 4 * B * 4 * sizeof(result2[0])));

    // Split input
    //// Define segmentation pointers for convenience
    half* X_re_hi = X_split + 4 * B * 0;
    half* X_re_lo = X_split + 4 * B * 1;
    half* X_im_hi = X_split + 4 * B * 2;
    half* X_im_lo = X_split + 4 * B * 3;
    float* re_s1 = scales + B * 0;
    float* re_s2 = scales + B * 1;
    float* im_s1 = scales + B * 2;
    float* im_s2 = scales + B * 3;

    //// Call the splitting kernel
    int numThreads = 64;
    int numBlocks = (B + 63) / 64;
    mySplit<<<numBlocks, numThreads>>>(4, X_re, X_re_hi, X_re_lo, re_s1, re_s2, B, buffer);
    mySplit<<<numBlocks, numThreads>>>(4, X_im, X_im_hi, X_im_lo, im_s1, im_s2, B, buffer);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during fft4 splitting\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
  
    // Matrix multiplication with Fourier matrix
    //// Call cublas gemm on F4_re
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 4, 4, &alpha, 
        F4_re,CUDA_R_16F, 4, X_split, CUDA_R_16F, 4, &beta,
        result1, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_re * X_split).\n");
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, B * 4, 4, &alpha,
        F4_im, CUDA_R_16F, 4, X_split, CUDA_R_16F, 4, &beta,
        result2, CUDA_R_32F, 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! CUBLAS kernel execution error (F4_im * X_split).\n");
        return FFT_FAILURE;
    }

    // Rescale the result and combine them together
    //// Set grid and block size
    dim3 threadsPerBlock(16, 4);
    dim3 blocksPerGrid((B+15)/16, 1);

    //// call kernel function (FX_re and FX_im will be zero-initialized)
    myAccumulate<<<blocksPerGrid, threadsPerBlock>>>(4, result1, result2, scales, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        fprintf(stderr, "!!!!! CUDA error: %s during fft4 accumulation\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }

    // Deallocate unified memory
    checkCudaErrors(cudaFree(scales));
    checkCudaErrors(cudaFree(X_split));
    checkCudaErrors(cudaFree(result1));
    checkCudaErrors(cudaFree(result2));

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!!! shutdown error\n");
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

/* 
 * Initialize Fourier matrix
 * Allocate unified memory and set value for F4_re and F4_im
 * */
FFT_S init_F4()
{
    // Allocate unified memory for Fourier Matrix
    int mem_size = 16 * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &F4_re, mem_size));
    checkCudaErrors(cudaMallocManaged((void **) &F4_im, mem_size));

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

/* 
 * float* X (N*B), half* Xhi (N*B), half* Xlo (N*B)
 * float* s1, float* s2
 * int N, int B. N is always 4
 * Grid and dim size should be 1D, total size = B
 * All data should be in unified memory or device memory
 * */
__global__ void mySplit(int N, float* X, half* Xhi, half* Xlo, float* s1, float* s2, int B, float* Xtemp)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < B){
        // Calculate scaling factor 1
        float scale1 = 0.0f;
        for (int i = 0; i < N; i++){
            float norm = (float) fabs(X[i + idx * N]);
            if (norm > scale1) scale1 = norm;
        }
        
        // If all number are zero, skip
        if (scale1 == 0.0f){
            s1[idx] = 0.0f;
            s2[idx] = 0.0f;
            for (int i = 0; i < N; i++){
                Xhi[i + idx * N] = Xlo[i + idx * N] = 0.0f;
            }
        }
        else
        {
            // Restrict scale range
            if (scale1 < EPS) scale1 = EPS;
            if (scale1 > 1.0f/EPS) scale1 = 1.0f/EPS;
            s1[idx] = scale1;

            // Scale the high half
            for (int i = 0; i < N; i++){
                Xtemp[i + idx * N] = X[i + idx * N]/scale1;
                Xhi[i + idx * N] = (half)(Xtemp[i + idx * N]);
                // Use Xtemp to store the residual
                Xtemp[i + idx * N] = X[i + idx * N] - scale1 * (float)(Xhi[i + idx * N]);
            }

           // Calculate the lower scaling factor
            float scale2 = 0.0f;
            for (int i = 0; i < N; i++){
                float norm = (float) fabs(Xtemp[i + idx * N]);
                if (norm > scale2) scale2 = norm;
            }
        
            // If all number are zero, skip
            if (scale2 == 0.0f){
                s2[idx] = 0.0f;
                for (int i = 0; i < N; i++){
                    Xlo[i + idx * N] = 0.0f;
                }
            }
            else
            {
                // Restrict scale range
                if (scale2 < EPS) scale2 = EPS;
                if (scale2 > 1.0f/EPS) scale2 = 1.0f/EPS;
                s2[idx] = scale2;

                for (int i = 0; i < N; i++){
                Xlo[i + idx * N] = (half) (Xtemp[i + idx * N] / scale2);
                }
            }
        }
    }
}

/* 
 * N is number of elements in one column (expected to be 4)
 * X1, X2 are 4 * (B * 4) column-major matrix. Inner order is by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
 * alpha is B * 4 array. Inner order is by batch. Outer order is re_s1, re_s2, im_s1, im_s2
 * R1, R2 are 4 * B matrix
 * B is batch size
 * */
__global__ void myAccumulate(int N, float* X1, float* X2, float* alpha, float* R1, float* R2, int B)
{

    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < N && j < B){
        R1[i + j * N] = R2[i + j * N] = 0.0f;
        R1[i + j * N] += alpha[j] * X1[i + j * N];
        R1[i + j * N] += alpha[j + B] * X1[i + j * N + N * B];
        R1[i + j * N] += -1.0f * alpha[j + 2*B] * X2[i + j * N + N * 2 * B];
        R1[i + j * N] += -1.0f * alpha[j + 3*B] * X2[i + j * N + N * 3 * B];
        R2[i + j * N] += alpha[j] * X2[i + j * N];
        R2[i + j * N] += alpha[j + B] * X2[i + j * N + N * B];
        R2[i + j * N] += alpha[j + 2*B] * X1[i + j * N + N * 2 * B];
        R2[i + j * N] += alpha[j + 3*B] * X1[i + j * N + N * 3 * B];
    }
}



/* 
 * Perform fft4 assuming the input is in the transposed layout
 * M is the number of rows
 * 4 * B is the number of columns
 * Note that the fourier matrix is symmetric
 */
FFT_S fft4_transposed(int M, float* X_re, float* X_im, float* FX_re, float* FX_im, int B) 
{


    // Variable declaration
    cublasStatus_t status;
    cublasHandle_t handle;
    cudaError_t cerror;

    //// Unified variables
    float *scales; // = *re_s1, *re_s2, *im_s1, *im_s2;
    half *X_split; // = *X_re_hi, *X_re_lo, *X_im_hi, *X_im_lo;
    float *result1, *result2; // Store the intermediate result
    //// Scaling variables
    float alpha = 1.0f, beta = 0.0f; 

    // Initialize cublas
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return FFT_FAILURE;
    }

    status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH); // allow Tensor Core
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS setting math mode error\n");
        return FFT_FAILURE;
    }

    //  Allocate unified memory with 0 initialization
    checkCudaErrors(cudaMallocManaged((void **) &scales, M * B * 4 * sizeof(float)));
    checkCudaErrors(cudaMemset(scales, 0.0f, M * B * 4 * sizeof(float)));
    checkCudaErrors(cudaMallocManaged((void **) &X_split, M * 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMemset(X_split, 0.0f, M * 4 * B * 4 * sizeof(half)));
    checkCudaErrors(cudaMallocManaged((void **) &result1, M * 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMemset(result1, 0.0f, M * 4 * B * 4 * sizeof(result1[0])));
    checkCudaErrors(cudaMallocManaged((void **) &result2, M * 4 * B * 4 * sizeof(result2[0])));
    checkCudaErrors(cudaMemset(result2, 0.0f, M * 4 * B * 4 * sizeof(result2[0])));

    // Split input
    //// Initialize Matrix and Vector data structure to store split result
    half* X_re_hi;
    X_re_hi.width = 4 * B;
    X_re_hi.height = M;
    X_re_hi = X_split + M * 4 * B * 0;

    half* X_re_lo;
    X_re_lo.width = 4 * B;
    X_re_lo.height = M;
    X_re_lo = X_split + M * 4 * B * 1;

    half* X_im_hi;
    X_im_hi.width = 4 * B;
    X_im_hi.height = M;
    X_im_hi = X_split + M * 4 * B * 2;

    half* X_im_lo;
    X_im_lo.width = 4 * B;
    X_im_lo.height = M;
    X_im_lo = X_split + M * 4 * B * 3;

    float* re_s1;
    re_s1.size = M * B;
    re_s1 = scales + M * B * 0;

    float* re_s2;
    re_s2.size = M * B;
    re_s2 = scales + M * B * 1;

    float* im_s1;
    im_s1.size = M * B;
    im_s1 = scales + M * B * 2;

    float* im_s2;
    im_s2.size = M * B;
    im_s2 = scales + M * B * 3;

    //// Call splitting function
    dim3 threadsPerBlock1(4, 16);
    dim3 BlocksPerGrid1((B + 3)/4, (M + 15)/16);
    mySplit_transposed<<<4, M, BlocksPerGrid1, threadsPerBlock1>>>(X_re, X_re_hi, X_re_lo, re_s1, re_s2, B, buffer);
    mySplit_transposed<<<4, M, BlocksPerGrid1, threadsPerBlock1>>>(X_im, X_im_hi, X_im_lo, im_s1, im_s2, B, buffer);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during splitting in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }
  
 
    // Call cublas function and finish Matrix multiplication calculation
    // The order of multiplicands are reversed
    //// Define batched offset
    long long int stride = M * 4;

    //// Call cublas batched gemm on F4_re

    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, X_split,
                        CUDA_R_16F, M, stride, F4_re, CUDA_R_16F, 4, 0, &beta, result1, CUDA_R_32F, M, stride, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error: %d in fft4_transposed ((c, d) * a).\n", status);
        printf("Error is: %d\n", status == CUBLAS_STATUS_NOT_SUPPORTED);
        return FFT_FAILURE;
    }

    //// Call cublas gemm on F4_im
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, 4, 4, &alpha, X_split,
                        CUDA_R_16F, M, stride, F4_im, CUDA_R_16F, 4, 0, &beta, result2, CUDA_R_32F, M, stride, B * 4, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error in fft4_transposed ((c, d) * b).\n");
        return FFT_FAILURE;
    }


    // Scale, combine and get result, add to output
    //// Set grid and block size
    dim3 threadsPerBlock2(16, 16);
    dim3 BlocksPerGrid2((4 * B + 15)/16, (M + 15)/16);

    //// call kernel function (buffer is zero-initialized inside)
    myAccumulate_transposed<<<BlocksPerGrid2, threadsPerBlock2>>>(4, M, result1, result2, scales, FX_re, FX_im, B);
    cerror = cudaGetLastError();
    if (cerror != cudaSuccess)
    {
        printf("CUDA error: %s during accumulation in fft4_transposed\n", cudaGetErrorString(cerror));
        return FFT_FAILURE;
    }


    // Deallocate unified memory
    if (cudaFree(scales) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free scales vector)\n");
        return FFT_FAILURE;
    }

    if (cudaFree(X_split) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free split result matrix)\n");
        return FFT_FAILURE;
    }

    if (cudaFree(result1) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free result 1 Matrix)\n");
        return FFT_FAILURE;
    }

    if (cudaFree(result2) != cudaSuccess) {
        fprintf(stderr, "!!!! unified memory free error (free result 2 Matrix)\n");
        return FFT_FAILURE;
    }

    // Shutdown cublas
    status = cublasDestroy(handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return FFT_FAILURE;
    }

    return FFT_SUCCESS;
}

/* 
 * float* X (M * (n * B)), half* Xhi (M * (n * B)), half* Xlo (M * (n * B))
 * float* s1 of size M * B, float* s2 of size M * B
 * int n, int M, int B. n is expected to be 4, M = N / 4
 * Grid and dim size should be 2D, total size = M * B
 * All data should be in unified memory or device memory
 * */
__global__ void mySplit_transposed(int n, int M, float* X, half* Xhi, half* Xlo, float* s1, float* s2, int B, float* Xtemp)
{

    int rowIdx = blockIdx.y * blockDim.y + threadIdx.y; // Row number (max M)
    int blockNum = blockIdx.x * blockDim.x + threadIdx.x; // 'Column number' (max B)

    if (rowIdx < M && blockNum < B){
        /* Data to be manipulated:
         *  X, Xhi, Xlo (rowIdx, blockIdx * n +0+1+2+3) = X, Xhi, Xlo[rowIdx + blockIdx * n * M + 0/1/2/3 * M]
         * s1, s2 (rowIdx, blockIdx) = s1, s2[rowIdx + blockIdx * M]
         */
        int offset = rowIdx + blockNum * n * M;
        int stride = M;
        int factor_idx = rowIdx + blockNum * M;

        // Calculate scaling factor 1
        float scale1 = 0.0f;
        for (int i = 0; i < n; i++){
            float norm = (float) fabs(X[offset + i * stride]);
            if (norm > scale1) scale1 = norm;
        }
        
        // If all number are zero, skip
        if (scale1 == 0.0f){
            s1[factor_idx] = 0.0f;
            s2[factor_idx] = 0.0f;
            for (int i = 0; i < n; i++){
                Xhi[offset + i * stride] = Xlo[offset + i * stride] = 0.0f;
            }
        }
        else
        {
            // Restrict scale range
            if (scale1 < EPS) scale1 = EPS;
            if (scale1 > 1.0f/EPS) scale1 = 1.0f/EPS;
            s1[factor_idx] = scale1;

            // Scale the high half
            for (int i = 0; i < n; i++){
                Xtemp[offset + i * stride] = X[offset + i * stride]/scale1;
                Xhi[offset + i * stride] = (half)(Xtemp[offset + i * stride]);
                // Use Xtemp to store the residual
                Xtemp[offset + i * stride] = X[offset + i * stride] - scale1 * (float)(Xhi[offset + i * stride]);
            }

           // Calculate the lower scaling factor
            float scale2 = 0.0f;
            for (int i = 0; i < n; i++){
                float norm = (float) fabs(Xtemp[offset + i * stride]);
                if (norm > scale2) scale2 = norm;
            }
        
            // If all number are zero, skip
            if (scale2 == 0.0f){
                s2[factor_idx] = 0.0f;
                for (int i = 0; i < n; i++){
                    Xlo[offset + i * stride] = 0.0f;
                }
            }
            else
            {
                // Restrict scale range
                if (scale2 < EPS) scale2 = EPS;
                if (scale2 > 1.0f/EPS) scale2 = 1.0f/EPS;
                s2[factor_idx] = scale2;

                for (int i = 0; i < n; i++){
                Xlo[offset + i * stride] = (half) (Xtemp[offset + i * stride] / scale2);
                }
            }
        }
    }
}


/* 
 * X1, X2 are M * (4 * B * 4) matrix. The inner-most column order is by element in a unit. Then by batch. Outer order is Re_hi, Re_lo, Im_hi, Im_lo
 * alpha is a M * B * 4 array. Inner most order is by rows. Then by batch. Outer order is re_s1, re_s2, im_s1, im_s2
 * R1, R2 are M * (4 * B) matrix
 * n is number of elements in one unit (expected to be 4)
 * M is number of rows, B is batch size
 * */
__global__ void myAccumulate_transposed(int n, int M, float* X1, float* X2, float* alpha, float* R1, float* R2, int B)
{

    int i = blockIdx.y * blockDim.y + threadIdx.y; // row number
    int j = blockIdx.x * blockDim.x + threadIdx.x; // column number

    if (i < M && j < 4 * B){
        int result_idx = i + j * M;
        int e_stride = M * 4 * B; // Stride for elements, e.g. from Re_hi to Re_lo
        int factor_idx = i + j / 4 * M;
        int f_stride = M * B; // Stride for factors, e.g. from re_s1 to re_s2
        R1[result_idx] = R2[result_idx] = 0.0f;

        R1[result_idx] += alpha[factor_idx] * X1[result_idx];
        R1[result_idx] += alpha[factor_idx + f_stride] * X1[result_idx + e_stride];
        R1[result_idx] += -1.0f * alpha[factor_idx + 2*f_stride] * X2[result_idx + 2*e_stride];
        R1[result_idx] += -1.0f * alpha[factor_idx + 3*f_stride] * X2[result_idx + 3*e_stride];
        R2[result_idx] += alpha[factor_idx] * X2[result_idx];
        R2[result_idx] += alpha[factor_idx + f_stride] * X2[result_idx + e_stride];
        R2[result_idx] += alpha[factor_idx + 2*f_stride] * X1[result_idx + 2*e_stride];
        R2[result_idx] += alpha[factor_idx + 3*f_stride] * X1[result_idx + 3*e_stride];
    }
}
