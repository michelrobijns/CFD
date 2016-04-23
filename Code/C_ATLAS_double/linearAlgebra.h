#ifndef LINEARALGEBRA_H_INCLUDED
#define LINEARALGEBRA_H_INCLUDED

double* mallocMatrix(int rows, int columns);
double* callocMatrix(int rows, int columns);
double* mallocVector(int rows);
double* callocVector(int rows);

void printMatrix(double *A);
void printVector(double *x);
void storeMatrix(double *A, char *fileName);
void storeVector(double *x, char *fileName);

double* matMatMult(double *A, double *B);
double* matVecMult(double *A, double *x);
double* vecScalarMult(double *x, double scalar);
double* diagMatInvert(double *A);
double* vecInvert(double *x);

#endif
