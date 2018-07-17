/*
 * A general header file that includes libraries and helpers
 * To be used by gfft and testing program
 */

#ifndef FFT_MY_INCLUDE_H
#define FFT_MY_INCLUDE_H


// C includes
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// CUDA includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Helper
/// Checking execution error, getting command line input
#include "../nvidia_helper/checkCudaErrors.h"
#include "../nvidia_helper/helper_string.h"

/// Matrix and vector
#include "../helper/my_vector.h"
#include "../helper/my_matrix.h"

/// Define program state
#include "../helper/my_const.h"

/// Utility programs
#include "fp32_to_fp16.h"
#include "fourier_matrix_4.h"
#include "fft4.h"


#define PI 3.14159265


#endif /* FFT_MY_INCLUDE_H */
