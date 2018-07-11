#ifndef FFT_FP32_TO_FP16
#define FFT_FP32_TO_FP16 

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define EPS 0.0000001192f

#include "../helper/my_vector.h"
#include "../helper/my_matrix.h"
#include "../helper/my_const.h"

FFT_S split_32_to_16(fft::MatrixF X, fft::MatrixH Xhi, fft::MatrixH Xlo, fft::VectorF s1, fft::VectorF s2, int N, int B)
{
    // Calculate scaling factor 1
    for (int j = 1; j <= B; j++){
        float scale1 = 0.0f;
        for (int i = 1; i <= N; i++){
            float norm = (float) fabs(X.element(i, j));
            if (norm > scale1) scale1 = norm;
        }
        
        // If all number are zero, skip
	if (scale1 == 0.0f){
            s1.element(j) = 0.0f;
            continue;
	}

        // Restrict scale range
        if (scale1 < EPS){
            scale1 = EPS;
        }
        if (scale1 > 1.0f/EPS){
            scale1 = 1.0f/EPS;
        }

        s1.element(j) = scale1;
    }

    // Initialize temporary matrix
    fft::MatrixF Xtemp;
    Xtemp.width = B;
    Xtemp.height = N;
    Xtemp.array = (float*)malloc(Xtemp.width * Xtemp.height * sizeof(float));

    // Get the normalized Xhi
    for (int j = 1; j <= B; j++){ 
        // If all number are zero, skip
        if (s1.element(j) == 0.0f){
            continue;
        }

        for (int i = 1; i <= N; i++) {
            Xtemp.element(i, j) = X.element(i, j) / s1.element(j);
            Xhi.element(i, j) = (half)(Xtemp.element(i, j));
            // Using Xtemp to store the residual
            Xtemp.element(i, j) = X.element(i, j) - s1.element(j) * (float)Xhi.element(i, j);
        }
    }

    // Calculate lower scaling factor
    for (int j = 1; j <= B; j++){
        // If all number are zero, skip
        if (s1.element(j) == 0.0f){
            continue;
        }

        float scale2 = 0.0f;
        for (int i = 1; i <= N; i++){
            float norm = (float)fabs(Xtemp.element(i, j));
            if (norm > scale2) scale2 = norm;
        }

        // If all remainders are zero, skip
	if (scale2 == 0.0f){
            s2.element(j) = 0.0f;
            continue;
	}

        if (scale2 < EPS){
            scale2 = EPS;
        }
        if (scale2 > 1.0f/EPS){
            scale2 = 1.0f/EPS;
        }
        s2.element(j) = scale2;
    }

    // Normalize lower part
    for (int j = 1; j <= B; j++){
        // If all number are zero, skip
        if (s1.element(j) == 0.0f){
            continue;
        }

        // If all remainders are zero, set X_lo to zero
	if (s2.element(j) == 0.0f){
            for (int i = 1; i <= N; i++){
                Xlo.element(i, j) = (half) 0.0f;
            }
            continue;
	}

        for (int i = 1; i <= N; i++){
            Xtemp.element(i, j) = Xtemp.element(i, j) / s2.element(j);
            Xlo.element(i, j) = (half) (Xtemp.element(i, j));
        }
    }

    free(Xtemp.array);   
    
    // Deal with zero case
    for (int j = 1; j <= B; j++){
        if (s1.element(j) == 0.0f){
            s2.element(j) == 0.0f;
            for (int i = 0; i <= N; i++){
                Xhi.element(i, j) = (half) 0.0f;
                Xlo.element(i, j) = (half) 0.0f;
            }
        }
    }

    return FFT_SUCCESS;
}

#endif /* FFT_FP32_TO_FP16 */

