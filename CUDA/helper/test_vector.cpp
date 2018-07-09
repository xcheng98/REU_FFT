#include <stdio.h>
#include <stdlib.h>

#include "my_vector.h"

int main(int argc, char *argv[])
{
    fft::VectorH A;
    A.size = 128;
    A.array = (half*)malloc(A.size * sizeof(half));
    for (int i = 1; i <= A.size; i++){
        A.element(i) = (float) 80 * i;
    }
    printf("A[1] = %f, A[24] = %f, A[128] = %f\n", (double)A.element(1), (double)A.element(24), (double)A.element(128));
    return 0;
}

