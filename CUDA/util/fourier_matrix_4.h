/*
 * Define F4_re and F4_im matrix
 * data type is fp16
 * store in unified memory
 */

#ifndef FFT_FM_4_H
#define FFT_FM_4_H

// C includes
#include <stdio.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Matrix 
#include "../helper/my_matrix.h"

// CUDA helper: to check error
#include "../nvidia_helper/checkCudaErrors.h"

fft::MatrixH F4_re;
fft::MatrixH F4_im;

FFT_S init_F4()
{
    // Allocate unified memory for Fourier Matrix
    int mem_size;

    F4_re.width = 4;
    F4_re.height = 4;
    mem_size = F4_re.width * F4_re.height * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &(F4_re.array), mem_size));

    F4_im.width = 4;
    F4_im.height = 4;
    mem_size = F4_im.width * F4_im.height * sizeof(half);
    checkCudaErrors(cudaMallocManaged((void **) &(F4_im.array), mem_size));

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

#endif /* FFT_FM_4_H */
