/*
 * Define matrix:
 ** Matrix A;
 ** A.width = w;
 ** A.height = h;
 ** A.array = (float*)malloc(A.width * A.height * sizeof(float));
 * Access elements:
 ** A.element(i, j) = x;
 ** Free memory:
 ** free(A.array)
 */

// To suppory half type
#include <math.h>
#include <cuda_fp16.h>


namespace fft {
    typedef class {
    public:
        int width;
        int height;
        float* array;
        float& element(int i, int j){
            return array[i-1 + (j-1) * height];
        }
    } MatrixF;

    typedef class {
    public:
        int width;
        int height;
        half* array;
        half& element(int i, int j){
            return array[i-1 + (j-1) * height];
        }
    } MatrixH;
}


