#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "linearAlgebra.h"

double** callocMatrix(int rows, int columns)
{
    double **matrix = calloc(rows * columns, sizeof(double));
    
    if (matrix == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    for (int i = 0; i < rows; i++) {
        matrix[i] = calloc(columns, sizeof(double));
        
        if (matrix[i] == NULL) {
            fprintf(stderr, "Out of memory.\n");
            exit(-1);
        }
    }
    
    return matrix;
}

void freeMatrix(double **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    
    free(matrix);
}

double** matMatMult(double **A, int rowsA, int columnsA,
                    double **B, int rowsB, int columnsB)
{
    if (columnsA != rowsB) {
        fprintf(stderr, "Incompatible matrix dimensions.\n");
        exit(-1);
    }
    
    double **C = callocMatrix(rowsA, columnsB);
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columnsB; j++) {
            for (int k = 0; k < columnsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    
    return C;
}

double* matVecMult(double **A, int rowsA, int columnsA, double *x, int rowsX)
{
    if (columnsA != rowsX) {
        fprintf(stderr, "Incompatible matrix and vector dimensions.\n");
        exit(-1);
    }
    
    double *b = calloc(rowsA, sizeof(double));
    
    if (b == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columnsA; j++) {
            b[i] += A[i][j] * x[j];
        }
    }
    
    return b;
}

void matScalarMult(double **matrix, double scalar, int rows, int columns)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            matrix[i][j] *= scalar;
        }
    }
}

void vecScalarMult(double *vector, int rows, double scalar)
{
    for (int i = 0; i < rows; i++) {
        vector[i] *= scalar;
    }
}

double** matTranspose(double **A, int rowsA, int columnsA)
{
    double **B = callocMatrix(columnsA, rowsA);
    
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < columnsA; j++) {
            B[j][i] = A[i][j];
        }
    }
    
    return B;
}

double** matInvert(double **A, int rowsA, int columnsA)
{
    if (rowsA != columnsA) {
        fprintf(stderr, "Matrix must be a diagonal for inversion.\n");
        exit(-1);
    }
    
    double **B = callocMatrix(rowsA, columnsA);
        
    for (int i = 0; i < rowsA; i++) {
        B[i][i] = 1 / A[i][i];
    }
    
    return B;
}

void printMatrix(double **matrix, int rows, int columns)
{
    fprintf(stdout, "The matrix below has %d rows and %d columns\n", rows,
        columns);   

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (matrix[i][j] != 0) {
                fprintf(stdout, "%5.2f ", matrix[i][j]);
            } else {
                fprintf(stdout, "  .   ", matrix[i][j]);
            }
        }
        
        fprintf(stdout, "\n");
    }
}

void printVector(double *vector, int rows)
{
    fprintf(stdout, "The vector below has %d rows\n", rows);   

    for (int i = 0; i < rows; i++) {
        if (vector[i] != 0) {
            fprintf(stdout, "%f\n", vector[i]);
        } else {
            fprintf(stdout, "  .\n", vector[i]);
        }
    }
}

void LUFactorization(double **A, int rowsA, int columnsA)
{    
    int N = rowsA;
    
    for (int c = 0; c < N - 1; c++) {
        for (int r = c + 1; r < N; r++) {
            A[r][c] = A[r][c] / A[c][c];

            for (int cc = c + 1; cc < N; cc++) {
                A[r][cc] = A[r][cc] - A[r][c] * A[c][cc];
            }
        }
    }
}

double* LUSubstitution(double **A, int rowsA, int columnsA, double *f,
                       int rowsF)
{
    if (columnsA != rowsF) {
        fprintf(stderr, "Incompatible matrix and vector dimensions.\n");
        exit(-1);
    }
            
    double *u = calloc(rowsA, sizeof(double));
    
    if (u == NULL) {
        fprintf(stderr, "Out of memory.\n");
        exit(-1);
    }
    
    int N = rowsA;
    double sum;
    
    // Forward substitution step
    for (int r = 0; r < N; r++)    {
        sum = 0;

        for (int c = 0; c < r; c++)
            sum += A[r][c] * u[c];

        u[r] = f[r] - sum;
    }

    // Backward substitution step
    for (int r = N - 1; r >= 0; r--) {
        sum = 0;

        for (int c = r + 1; c < N; c++)
            sum += A[r][c] * u[c];

        u[r] = (u[r] - sum) / A[r][r];
    }
    
    return u;
}
