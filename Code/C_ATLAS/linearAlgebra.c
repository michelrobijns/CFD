#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "linearAlgebra.h"

float* mallocMatrix(int rows, int columns)
{
    if (rows <= 0 || columns <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    float *A = malloc((2 + rows * columns) * sizeof(float));
    
    if (A == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    A[0] = rows;
    A[1] = columns;
    
    return A;
}

float* callocMatrix(int rows, int columns)
{
    if (rows <= 0 || columns <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    float *A = calloc(2 + rows * columns, sizeof(float));
    
    if (A == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    A[0] = rows;
    A[1] = columns;
    
    return A;
}

float* mallocVector(int rows)
{
    if (rows <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    float *x = malloc((1 + rows) * sizeof(float));
    
    if (x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    x[0] = rows;
    
    return x;
}

float* callocVector(int rows)
{
    if (rows <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    float *x = calloc(1 + rows, sizeof(float));
    
    if (x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    x[0] = rows;
    
    return x;
}

void printMatrix(float *A)
{
    int rows = A[0];
    int columns = A[1];
    float value;
    
    fprintf(stdout, "The matrix below has %d rows and %d columns\n", rows,
        columns);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            value = A[2 + i * columns + j];
            
            if (value != 0) {
                fprintf(stdout, "% 3.2f ", value);
            } else {
                fprintf(stdout, "  .   ");
            }
        }
        
        fprintf(stdout, "\n\n");
    }
}

void printVector(float *x)
{
    int rows = x[0];
    float value;
    
    fprintf(stdout, "The vector below has %d rows\n", rows);
    
    for (int i = 0; i < rows; i++) {
        value = x[1+i];
        
        if (value != 0) {
            fprintf(stdout, "% f\n", value);
        } else {
            fprintf(stdout, " .\n");
        }
    }
}

float* matMatMult(float *A, float *B)
{
    float *C = callocMatrix(A[0], B[1]);
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A[0], B[1], A[1], 1.0, &A[2], A[1], &B[2], B[1], 0.0, &C[2], C[1]);
    
    return C;
}

float* matVecMult(float *A, float *x)
{
    float *y = callocVector(A[0]);
    
    cblas_sgemv(CblasRowMajor, CblasNoTrans, A[0], A[1], 1.0, &A[2], A[1], &x[1], 1, 0.0, &y[1], 1);
    
    return y;
}

float* vecScalarMult(float *x, float scalar)
{
    float *y = mallocVector(x[0]);
    
    cblas_scopy(x[0], &x[1], 1, &y[1], 1);
    cblas_sscal(x[0], scalar, &y[1], 1);
    
    return y;
}

float* diagMatInvert(float *A)
{
    int rows = A[0];
    int columns = A[1];
    
    if (rows != columns) {
        fprintf(stderr, "Matrix must be a diagonal for inversion.\n");
        exit(EXIT_FAILURE);
    }
    
    float *B = callocMatrix(rows, columns);
        
    for (int i = 0; i < rows; i++) {
        B[2 + i * columns + i] = 1 / A[2 + i * columns + i];
    }
    
    return B;
}
