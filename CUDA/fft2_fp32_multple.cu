
/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>

/* Matrix size */
#define N  (2)
#define nvec 4

/* FFT2 */
int main(int argc, char **argv)
{
    cublasStatus_t status;
    float *X_re;
    float *FX2_re;
    float *X_re_FX2_re;
//  float *h_C_ref;
    float *X_im;
    float *FX2_im;
    float *X_re_FX2_im;
    float *X_im_FX2_re;
    float *X_im_FX2_im;
    float *FX_re;
    float *FX_im;
    float *d_X_re = 0;
    float *d_X_im = 0;
    float *d_FX2_re = 0;
    float *d_FX2_im = 0;
    float *d_X_re_FX2_re = 0;
    float *d_X_re_FX2_im = 0;
    float *d_X_im_FX2_re = 0;
    float *d_X_im_FX2_im = 0;
    float *d_FX_re;
    float *d_FX_im;
    float alpha = 1.0f;
    float beta = 0.0f;
    int n2 = N * nvec;
    int i;
    cublasHandle_t handle;

    int dev = findCudaDevice(argc, (const char **) argv);

    if (dev == -1)
    {
        return EXIT_FAILURE;
    }

    /* Initialize CUBLAS */
    printf("simpleCUBLAS test running..\n");

    status = cublasCreate(&handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! CUBLAS initialization error\n");
        return EXIT_FAILURE;
    }

   /* Allocate host memory for the matrices */
   X_re = (float *)malloc(n2 * sizeof(X_re[0]));

    if (X_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (X_re)\n");
        return EXIT_FAILURE;
    }

   X_im = (float *)malloc(n2 * sizeof(X_im[0]));

    if (X_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (X_im)\n");
        return EXIT_FAILURE;
    }

    FX2_re = (float *)malloc(4 * sizeof(FX2_re[0]));

    if (FX2_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (FX2_re)\n");
        return EXIT_FAILURE;
    }

    FX2_im = (float *)malloc(4 * sizeof(FX2_im[0]));

    if (FX2_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (FX2_im)\n");
        return EXIT_FAILURE;
    }

    X_re_FX2_re = (float *)malloc(n2 * sizeof(X_re_FX2_re[0]));

    if (X_re_FX2_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (X_re_FX2_re)\n");
        return EXIT_FAILURE;
    }

    X_re_FX2_im = (float *)malloc(n2 * sizeof(X_re_FX2_im[0]));

    if (X_re_FX2_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (X_re_FX2_im)\n");
        return EXIT_FAILURE;
    }

    X_im_FX2_re = (float *)malloc(n2 * sizeof(X_im_FX2_re[0]));

    if (X_im_FX2_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (X_im_FX2_re)\n");
        return EXIT_FAILURE;
    }

    X_im_FX2_im = (float *)malloc(n2 * sizeof(X_im_FX2_im[0]));

    if (X_im_FX2_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (X_im_FX2_im)\n");
        return EXIT_FAILURE;
    }

    FX_re = (float *)malloc(n2 * sizeof(FX_re[0]));

    if (FX_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (FX_re)\n");
        return EXIT_FAILURE;
    }

    FX_im = (float *)malloc(n2 * sizeof(FX_im[0]));

    if (FX_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (FX_im)\n");
        return EXIT_FAILURE;
    }


    /* Fill the matrices with test data */
    printf("\n X_re = [");
    for (i = 0; i < n2; i++)
    {
       X_re[i] = rand() / (float)RAND_MAX;
        printf("%f, ", X_re[i]);
    }
    printf("]; \n X-im = [");
    for (i=0; i<n2; i++)
    {
       X_im[i] = rand() / (float)RAND_MAX;
        printf("%f, ", X_im[i]);
       X_re_FX2_re[i] = 0;
    }
    printf("];");
        FX2_re[0] = 1;
        FX2_re[1] = 1;
        FX2_re[2] = 1;
        FX2_re[3] = -1;
        FX2_im[0] = 0;
        FX2_im[1] = 0;
        FX2_im[2] = 0;
        FX2_im[3] = 0;

   /* Allocate device memory for the matrices */
    if (cudaMalloc((void **)&d_X_re, n2 * sizeof(d_X_re[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Are)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_X_im, n2 * sizeof(d_X_im[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Aim)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_FX2_re, 4 * sizeof(d_FX2_re[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Bre)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_FX2_im, 4 * sizeof(d_FX2_im[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Bim)\n");
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_X_re_FX2_re, n2 * sizeof(d_X_re_FX2_re[0])) != cu                                                                                        daSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Crere)\n"                                                                                        );
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_X_re_FX2_im, n2 * sizeof(d_X_re_FX2_im[0])) != cu                                                                                        daSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Creim)\n"                                                                                        );
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_X_im_FX2_re, n2 * sizeof(d_X_im_FX2_re[0])) != cu                                                                                        daSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Cimre)\n"                                                                                        );
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_X_im_FX2_im, n2 * sizeof(d_X_im_FX2_im[0])) != cu                                                                                        daSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate Cimim)\n"                                                                                        );
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_FX_im, n2 * sizeof(d_FX_im[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate FXim)\n")                                                                                        ;
        return EXIT_FAILURE;
    }

    if (cudaMalloc((void **)&d_FX_re, n2 * sizeof(d_FX_re[0])) != cudaSuccess)
    {
        fprintf(stderr, "!!!! device memory allocation error (allocate FXre)\n")                                                                                        ;
        return EXIT_FAILURE;
    }

      /* Initialize the device matrices with the host matrices */
    status = cublasSetVector(n2, sizeof(X_re[0]), X_re, 1, d_X_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write A)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(X_im[0]), X_im, 1, d_X_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write Aim)\n");
        return EXIT_FAILURE;
    }
   status = cublasSetVector(4, sizeof(FX2_re[0]), FX2_re, 1, d_FX2_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

   status = cublasSetVector(4, sizeof(FX2_im[0]), FX2_im, 1, d_FX2_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write B)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(X_re_FX2_re[0]), X_re_FX2_re, 1, d_X_re_                                                                                        FX2_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(X_re_FX2_im[0]), X_re_FX2_im, 1, d_X_re_                                                                                        FX2_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(X_im_FX2_re[0]), X_im_FX2_re, 1, d_X_im_                                                                                        FX2_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(X_im_FX2_im[0]), X_im_FX2_im, 1, d_X_im_                                                                                        FX2_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write C)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(FX_im[0]), FX_im, 1, d_FX_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write FX im)\n");
        return EXIT_FAILURE;
    }

    status = cublasSetVector(n2, sizeof(FX_re[0]), FX_re, 1, d_FX_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (write FX re)\n");
        return EXIT_FAILURE;
    }

    /* Performs multiply  operation using cublas */
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, nvec,N, &alpha, d_                                                                                        FX2_re, N, d_X_re, N, &beta, d_X_re_FX2_re, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, nvec,N, &alpha, d_                                                                                        FX2_im, N, d_X_re, N, &beta, d_X_re_FX2_im, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, nvec,N, &alpha, d_                                                                                        FX2_re, N, d_X_im, N, &beta, d_X_im_FX2_re, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, nvec,N, &alpha, d_                                                                                        FX2_im, N, d_X_im, N, &beta, d_X_im_FX2_im, N);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! kernel execution error.\n");
        return EXIT_FAILURE;
    }

/* Copy into Result Matrix */
    status = cublasScopy(handle, n2,d_X_re_FX2_re, 1,d_FX_re,1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"!!! Cublas Kernal Exececution Error copy .\n");
        return EXIT_FAILURE;
    }

    status = cublasScopy(handle, n2,d_X_re_FX2_im, 1,d_FX_im,1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,"!!! Cublas Kernal Exececution Error copy .\n");
        return EXIT_FAILURE;
    }

/* ac-bd */
    alpha = -1.0f;
    status = cublasSaxpy(handle, n2, &alpha, d_X_im_FX2_im, 1, d_FX_re, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (ac - bd).\n");
        return EXIT_FAILURE;
    }

    alpha = 1.0f;
    status = cublasSaxpy(handle, n2, &alpha, d_X_im_FX2_re, 1, d_FX_im, 1);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "!!!! CUBLAS kernel execution error (ad + bc).\n");
        return EXIT_FAILURE;
    }


    /* Allocate host memory for reading back the result from device memory */
    X_re_FX2_re = (float *)malloc(n2 * sizeof(X_re_FX2_re[0]));

    if (X_re_FX2_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    X_re_FX2_im = (float *)malloc(n2 * sizeof(X_re_FX2_im[0]));

    if (X_re_FX2_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    X_im_FX2_re = (float *)malloc(n2 * sizeof(X_im_FX2_re[0]));

    if (X_im_FX2_re == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    X_im_FX2_im = (float *)malloc(n2 * sizeof(X_im_FX2_im[0]));

    if (X_im_FX2_im == 0)
    {
        fprintf(stderr, "!!!! host memory allocation error (C)\n");
        return EXIT_FAILURE;
    }

    /* Read the result back */
    status = cublasGetVector(n2, sizeof(X_re_FX2_re[0]), d_X_re_FX2_re, 1, X_re_                                                                                        FX2_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    status = cublasGetVector(n2, sizeof(X_re_FX2_im[0]), d_X_re_FX2_im, 1, X_re_                                                                                        FX2_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }
    status = cublasGetVector(n2, sizeof(X_im_FX2_re[0]), d_X_im_FX2_re, 1, X_im_                                                                                        FX2_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }
    status = cublasGetVector(n2, sizeof(X_im_FX2_im[0]), d_X_im_FX2_im, 1, X_im_                                                                                        FX2_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    status = cublasGetVector(n2, sizeof(FX_re[0]), d_FX_re, 1, FX_re, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

    status = cublasGetVector(n2, sizeof(FX_im[0]), d_FX_im, 1, FX_im, 1);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! device access error (read C)\n");
        return EXIT_FAILURE;
    }

   printf("\nResult: \n");
   printf(" F_re = [");
   for(int k=0;k<n2;k++)
   {
        printf(" %f,", FX_re[k]);
   }
   printf("]; \n F_im = [");
   for(int k=0;k<n2;k++)
   {
        printf("%f,", FX_im[k]);
   }
   printf("];\n");

    /* Memory clean up */

    free(X_re);
    free(FX2_re);
    free(X_re_FX2_re);
//  free(h_C_ref);
    free(X_im);
    free(FX2_im);
    free(X_re_FX2_im);
    free(X_im_FX2_re);
    free(X_im_FX2_im);
    free(FX_re);
    free(FX_im);

    if (cudaFree(d_X_re) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_FX2_re) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_X_re_FX2_re) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

   if (cudaFree(d_X_im) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (A)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_FX2_im) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (B)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_X_re_FX2_im) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_X_im_FX2_re) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_X_im_FX2_im) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    if (cudaFree(d_FX_re) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }
    if (cudaFree(d_FX_im) != cudaSuccess)
    {
        fprintf(stderr, "!!!! memory free error (C)\n");
        return EXIT_FAILURE;
    }

    /* Shutdown */
    status = cublasDestroy(handle);

    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "!!!! shutdown error (A)\n");
        return EXIT_FAILURE;
    }



}
