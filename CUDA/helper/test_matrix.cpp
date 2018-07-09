#include <stdio.h>
#include <stdlib.h>

#include "my_matrix.h"
using namespace fft;

int main(int argc, char *argv[])
{
    MatrixF A;
    A.height = 64;
    A.width = 32;
    A.array = (float*)malloc(A.width * A.height * sizeof(float));
    for (int i = 1; i <= A.height; i++){
        for (int j = 1; j <= A.width; j++) {
            A.element(i, j) = i*j;
        }
    }
    printf("A[1][1] = %f, A[64][32] = %f\n", A.element(1,1), A.element(64, 32));
    return 0;
}

