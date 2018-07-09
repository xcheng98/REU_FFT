/*
 * To support 1-based indexing
 * Define vector:
 ** fft::Vector A;
 ** A.size = n;
 ** A.array = (float*)malloc(A.size * sizeof(float));
 * Access elements:
 ** A.element(i) = x;
 * Free memory:
 ** free(A.array)
 */
#include <math.h>
#include <cuda_fp16.h>

namespace fft {
    typedef class {
    public:
        int size;
        float* array;
        float& element(int i){
            return array[i - 1];
        }
    } VectorF;

    typedef class {
    public:
        int size;
        half* array;
        half& element(int i){
            return array[i - 1];
        }
    } VectorH;
}


