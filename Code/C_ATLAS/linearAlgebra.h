#ifndef LINEARALGEBRA_H_INCLUDED
#define LINEARALGEBRA_H_INCLUDED

float* mallocMatrix(int rows, int columns);
float* callocMatrix(int rows, int columns);
float* mallocVector(int rows);
float* callocVector(int rows);

void printMatrix(float *A);
void printVector(float *x);

float* matMatMult(float *A, float *B);
float* matVecMult(float *A, float *x);
float* vecScalarMult(float *x, float scalar);
float* diagMatInvert(float *A);

#endif
