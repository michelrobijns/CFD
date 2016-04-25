#ifndef LINEARALGEBRA_H_INCLUDED
#define LINEARALGEBRA_H_INCLUDED

float* mallocMatrix(int rows, int columns);
float* callocMatrix(int rows, int columns);
float* mallocVector(int rows);
float* callocVector(int rows);

void printMatrix(float *A);
void printVector(float *x);
void storeMatrix(float *A, char *fileName);
void storeVector(float *x, char *fileName);

float* matMatMult(float *A, float *B);
float* matVecMult(float *A, float *x);
float* vecScalarMult(float *x, float scalar);
float* diagMatInvert(float *A);
float* vecInvert(float *x);

#endif
