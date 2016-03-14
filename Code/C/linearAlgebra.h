#ifndef LINEARALGEBRA_H_INCLUDED
#define LINEARALGEBRA_H_INCLUDED

double** callocMatrix(int rows, int columns);
void freeMatrix(double **matrix, int rows);
double** matMatMult(double **A, int rowsA, int columnsA,
                    double **B, int rowsB, int columnsB);
double* matVecMult(double **A, int rowsA, int columnsA, double *x, int rowsX);
void matScalarMult(double **matrix, double scalar, int rows, int columns);
void vecScalarMult(double *vector, int rows, double scalar);
double** matTranspose(double **A, int rowsA, int columnsA);
double** matInvert(double **A, int rowsA, int columnsA);
void printMatrix(double **matrix, int rows, int columns);
void printVector(double *vector, int rows);
void LUFactorization(double **A, int rowsA, int columnsA);
double* LUSubstitution(double **A, int rowsA, int columnsA, double *f,
                       int rowsF);

#endif
