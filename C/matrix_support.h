/*
 * Define matrix:
 ** Matrix A;
 ** A.width = w;
 ** A.height = h;
 ** A.elements = (float*)malloc(A.width * A.height * sizeof(float));
 * Access elements:
 ** A.elements[getIndex(i, j, A.width, A.height)] = x;
 */

int getIndex(int i, int j, int width, int height){
    return i-1 + (j-1) * height;
}    

typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;


