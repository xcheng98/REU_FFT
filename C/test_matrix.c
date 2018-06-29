#include <stdio.h>
#include <stdlib.h>

#include "matrix_support.h"

int main(int argc, char *argv[])
{
    Matrix A;
    A.height = 64;
    A.width = 32;
    A.elements = (float*)malloc(A.width * A.height * sizeof(float));
    for (int i = 1; i <= A.height; i++){
        for (int j = 1; j <= A.width; j++) {
            A.elements[getIndex(i, j, A.width, A.height)] = i*j;
        }
    }
    printf("A[1][1] = %f, A[64][32] = %f\n", A.elements[getIndex(1,1, A.width, A.height)], A.elements[getIndex(64,32, A.width, A.height)]);
    return 0;
}

