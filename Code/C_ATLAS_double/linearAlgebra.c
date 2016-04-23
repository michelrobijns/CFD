#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
//#include <omp.h>
#include "linearAlgebra.h"

double* mallocMatrix(int rows, int columns)
{
    if (rows <= 0 || columns <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    double *A = malloc((2 + rows * columns) * sizeof(double));
    
    if (A == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    A[0] = rows;
    A[1] = columns;
    
    return A;
}

double* callocMatrix(int rows, int columns)
{
    if (rows <= 0 || columns <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    double *A = calloc(2 + rows * columns, sizeof(double));
    
    if (A == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    A[0] = rows;
    A[1] = columns;
    
    return A;
}

double* mallocVector(int rows)
{
    if (rows <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    double *x = malloc((1 + rows) * sizeof(double));
    
    if (x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    x[0] = rows;
    
    return x;
}

double* callocVector(int rows)
{
    if (rows <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        exit(EXIT_FAILURE);
    }
    
    double *x = calloc(1 + rows, sizeof(double));
    
    if (x == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(EXIT_FAILURE);
    }
    
    // Set dimensions
    x[0] = rows;
    
    return x;
}

void printMatrix(double *A)
{
    int rows = A[0];
    int columns = A[1];
    double value;
    
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

void printVector(double *x)
{
    int rows = x[0];
    double value;
    
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

void storeMatrix(double *A, char *fileName)
{
    int rows = A[0];
    int columns = A[1];
    
    FILE *filePointer = fopen(fileName, "w");
    
    if (filePointer != NULL) {
        char str[80];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                sprintf(str, "%.6e", A[2 + i * columns + j]);
                fputs(str, filePointer);
                
                if (j != columns - 1) {
                    fputs("\t", filePointer);
                }
            }
            fputs("\n", filePointer);
        }
        
        fclose(filePointer);
    } else {
        fprintf(stderr, "Could not open %s.\n", fileName);
    }
}

void storeVector(double *x, char *fileName)
{
    int rows = x[0];
    
    FILE *filePointer = fopen(fileName, "w");
    
    if (filePointer != NULL) {
        char str[80];
        
        for (int i = 0; i < rows; i++) {
            sprintf(str, "%.6e\n", x[1+i]);
            fputs(str, filePointer);
        }
        
        fclose(filePointer);
    } else {
        fprintf(stderr, "Could not open %s.\n", fileName);
    }
}

double* matMatMult(double *A, double *B)
{
    double *C = callocMatrix(A[0], B[1]);
    
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, A[0], B[1], A[1], 1.0, &A[2], A[1], &B[2], B[1], 0.0, &C[2], C[1]);
    
    return C;
}

double* matVecMult(double *A, double *x)
{
    double *y = callocVector(A[0]);
    
    cblas_dgemv(CblasRowMajor, CblasNoTrans, A[0], A[1], 1.0, &A[2], A[1], &x[1], 1, 0.0, &y[1], 1);
    
    return y;
}

double* vecScalarMult(double *x, double scalar)
{
    double *y = mallocVector(x[0]);
    
    cblas_dcopy(x[0], &x[1], 1, &y[1], 1);
    cblas_dscal(x[0], scalar, &y[1], 1);
    
    return y;
}

double* diagMatInvert(double *A)
{
    int rows = A[0];
    int columns = A[1];
    
    if (rows != columns) {
        fprintf(stderr, "Matrix must be a diagonal for inversion.\n");
        exit(EXIT_FAILURE);
    }
    
    double *B = callocMatrix(rows, columns);
        
    for (int i = 0; i < rows; i++) {
        B[2 + i * columns + i] = 1 / A[2 + i * columns + i];
    }
    
    return B;
}

double* vecInvert(double *x)
{
    int rows = x[0];
    
    double *y = callocVector(rows);
    
    for (int i = 0; i < rows; i++) {
        y[1 + i] = 1.0 / x[1 + i];
    }
    
    return y;
}
