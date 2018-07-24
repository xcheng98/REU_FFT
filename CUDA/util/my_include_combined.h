/*
 * A general header file that includes libraries and helpers
 * To be used by the optimized gfft and testing program
 * No need to include matrix and vector
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

/// Define program state
#include "../helper/my_const.h"


#endif /* FFT_MY_INCLUDE_H */
